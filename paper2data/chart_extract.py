"""柱状图数值提取 —— 借鉴 thu-digitizer 的「标定 + 颜色分割 + 反算」。

设计（三段式，vision 与代码分工）：
1. vision 预检：报「子图区域 + 图类型 + 柱子颜色」；
2. 路由：图类型 → 对应提取器（本模块只做柱状图）；
3. 提取器：AxisCalibration 鲁棒标定 + 颜色分割检测柱子 + 反算。

标定锚点 = 代码检测刻度线像素位置 + vision 报刻度数值。
"""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
from PIL import Image


# 轴/刻度检测的默认颜色容差：locate_plot_bounds / detect_axis_ticks / figure_to_text
# 三处必须一致，否则轴框定位和标定会基于不同的像素掩码各算各的
AXIS_TOL = 20


class AxisCalibration:
    """像素 → 数值的仿射标定（鲁棒）。

    借鉴 thu-digitizer：至少两个锚点，多个锚点时用 Huber 权重迭代，
    防止单个刻度读错把整条拟合线拉偏。像素 y 与数值常反比（图像 y
    向下、数值向上），线性拟合自动处理。
    """

    def __init__(self, anchors: Sequence[tuple[float, float]], huber_delta: float = 1.5):
        pairs = [(float(px), float(val)) for px, val in anchors]
        if len(pairs) < 2:
            raise ValueError("至少需要两个轴锚点 (像素, 数值)")
        self.pixels = np.array([px for px, _ in pairs], dtype=float)
        self.values = np.array([val for _, val in pairs], dtype=float)

        design = np.column_stack([self.pixels, np.ones_like(self.pixels)])
        slope, intercept = np.linalg.lstsq(design, self.values, rcond=None)[0]

        for _ in range(8):
            residuals = self.values - (slope * self.pixels + intercept)
            scale = max(1.4826 * float(np.median(np.abs(residuals - np.median(residuals)))), 1e-9)
            z = np.abs(residuals) / scale
            weights = np.where(z > huber_delta, huber_delta / np.maximum(z, 1e-9), 1.0)
            slope, intercept = np.linalg.lstsq(
                design * weights[:, None], self.values * weights, rcond=None
            )[0]

        self.slope = float(slope)
        self.intercept = float(intercept)

    def value_at_pixel(self, pixel: float) -> float:
        return self.slope * pixel + self.intercept

    def pixel_at_value(self, value: float) -> float:
        return (value - self.intercept) / self.slope


def detect_bars(
    img: Image.Image,
    color: tuple[int, int, int],
    tol: int = 20,
    min_area: int = 200,
    min_side: int = 0,
    with_fill_ratio: bool = False,
) -> list:
    """颜色分割 + 连通组件，检测柱子矩形。

    color 是参数（由 vision 报，不写死）。返回 [(x0, y0, x1, y1)]；
    with_fill_ratio=True 时返回 [(x0, y0, x1, y1, fill_ratio)]。
    min_side > 0 时排除宽或高小于该值的细长线（轴线/刻度线/图框线）。
    """
    arr = np.array(img.convert("RGB")).astype(int)
    r, g, b = color
    mask = (
        (abs(arr[:, :, 0] - r) <= tol)
        & (abs(arr[:, :, 1] - g) <= tol)
        & (abs(arr[:, :, 2] - b) <= tol)
    )

    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    bars: list = []
    for i in range(h):
        for j in range(w):
            if not mask[i, j] or visited[i, j]:
                continue
            q = deque([(i, j)])
            visited[i, j] = True
            min_i = max_i = i
            min_j = max_j = j
            pixel_count = 1
            while q:
                y, x = q.popleft()
                min_i, max_i = min(min_i, y), max(max_i, y)
                min_j, max_j = min(min_j, x), max(max_j, x)
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
                        pixel_count += 1
            area = (max_i - min_i + 1) * (max_j - min_j + 1)
            width = max_j - min_j + 1
            height = max_i - min_i + 1
            if area >= min_area and width >= min_side and height >= min_side:
                fill_ratio = pixel_count / area
                if with_fill_ratio:
                    bars.append((min_j, min_i, max_j, max_i, fill_ratio))
                else:
                    bars.append((min_j, min_i, max_j, max_i))
    return bars


def detect_bars_baseline(
    img: Image.Image,
    color: tuple[int, int, int],
    tol: int = 50,
    min_area: int = 200,
    min_side: int = 15,
    min_fill_ratio: float = 0.55,
    baseline_tol_px: int = 5,
) -> tuple[list[tuple[int, int, int, int]], float | None]:
    """检测柱子（几何过滤 + 填充率 + 共同基线），不依赖外部 baseline。

    先按宽高过滤（min_side）排除轴线/刻度线/图框线，再按填充率筛掉文字
    笔画（稀疏），再聚类实心块的底端找共同基线，保留底端贴基线的块。
    y 轴不从 0 开始（截断轴）也能正确找基线。返回 (柱子列表, 基线像素)。
    """
    blocks = detect_bars(
        img, color, tol=tol, min_area=min_area, min_side=min_side, with_fill_ratio=True
    )
    solid = [b for b in blocks if b[4] >= min_fill_ratio]
    if not solid:
        return [], None
    if len(solid) == 1:
        return [solid[0][:4]], float(solid[0][3])

    # 聚类实心块底端（y1）找共同基线：柱子底端都贴同一水平线，图例底端分散
    bottoms = sorted(b[3] for b in solid)
    groups: list[list[int]] = []
    for y in bottoms:
        if groups and y - groups[-1][-1] <= baseline_tol_px:
            groups[-1].append(y)
        else:
            groups.append([y])
    best = max(groups, key=len)
    baseline = float(sum(best) / len(best))
    bars = [
        (b[0], b[1], b[2], b[3])
        for b in solid
        if abs(b[3] - baseline) <= baseline_tol_px
    ]
    return bars, baseline


def detect_axis_ticks(
    img: Image.Image,
    tick_color: tuple[int, int, int] = (0, 0, 0),
    tol: int = 60,
    min_run_len: int = 3,
    max_run_len: int = 40,
    search_radius: int = 30,
    with_len: bool = False,
    axis_hint: int | None = None,
):
    """检测 y 轴刻度线的 y 像素位置（从上到下）。

    先找 y 轴（每列最长垂直连续段 + 轴的实际宽度），再在轴两侧（不含轴列）
    找短水平段。刻度线是「从轴向外延伸、远端不触搜索窗口边界」的短段；网格线
    横贯绘图区会触到远端边界，据此排除。方向不再假设刻度线只在轴左侧。

    with_len=True 时返回 [(y, 线长)]——**主刻度线通常比次刻度线长**，标定那边
    靠「线长排候选次序 + 间距定主刻度」来选锚点（见 `figure_extract._anchors_by_major_grid`）。
    默认 False，返回 list[int]，与旧行为逐位一致。
    """
    arr = np.array(img.convert("RGB")).astype(int)
    r, g, b = tick_color
    mask = (
        (abs(arr[:, :, 0] - r) <= tol)
        & (abs(arr[:, :, 1] - g) <= tol)
        & (abs(arr[:, :, 2] - b) <= tol)
    )
    h, w = mask.shape

    # 1. 找 y 轴列。给了 axis_hint（代码切框的轴框左沿）就**只在它附近选**：
    #    外扩裁剪会把左邻面板卷进来，而邻面板的右边框往往比本块 y 轴还长 —— 全局
    #    那套「最左的合格竖线」会选中它（实测 24_fig3 块4：选到 x=37，本块轴在 178，
    #    结果刻度检出 0 条、标定失败）。没给 hint 才退回全局判定。
    if axis_hint is None:
        axis = _find_y_axis_col(mask, min_axis_len=30)
    else:
        axis = _find_y_axis_col_near(mask, axis_hint, min_axis_len=30)
    if axis is None:
        return []
    axis_x = axis[0]

    # 2. 轴的实际宽度：以 axis_x 为中心向两侧扩展（连续的垂直长线列）
    col_runs = np.zeros(w, dtype=int)
    for x in range(w):
        run = mx = 0
        for v in mask[:, x]:
            if v:
                run += 1
                if run > mx:
                    mx = run
            else:
                run = 0
        col_runs[x] = mx
    axis_left = axis_x
    while axis_left - 1 >= 0 and col_runs[axis_left - 1] >= col_runs[axis_x] * 0.5:
        axis_left -= 1
    axis_right = axis_x
    while axis_right + 1 < w and col_runs[axis_right + 1] >= col_runs[axis_x] * 0.5:
        axis_right += 1

    # 3. 轴两侧（不含轴列）找短水平段；触「远端」边界的是贯穿线（网格线）。
    #    刻度线近轴端贴着轴边缘是正常的，只查远离轴的那一端是否触界。
    def scan_len(y: int, x0: int, x1: int, far_is_left: bool) -> int:
        """该行里「合格刻度段」的最长长度；没有合格段返回 0。"""
        if x1 < x0:
            return 0
        best = 0
        x = x0
        while x <= x1:
            if not mask[y, x]:
                x += 1
                continue
            start = x
            while x <= x1 and mask[y, x]:
                x += 1
            end = x - 1
            touches_far = start == x0 if far_is_left else end == x1
            # 刻度线从轴伸出，近轴端必然贴轴边缘（硬几何约束，排柱子顶碎片）
            touches_axis = (end >= x1 - 1) if far_is_left else (start <= x0 + 1)
            if (
                touches_axis
                and not touches_far
                and min_run_len <= end - start + 1 <= max_run_len
            ):
                best = max(best, end - start + 1)
        return best

    tick_rows: dict[int, int] = {}
    for y in range(h):
        L = scan_len(y, max(0, axis_left - search_radius), axis_left - 1, True)
        if not L:
            L = scan_len(y, axis_right + 1, min(w - 1, axis_right + search_radius), False)
        if L:
            tick_rows[y] = L

    # 4. 合并相邻行（一条刻度线占 2~3 像素）；合并时线长取最长的那条
    merged: list[tuple[int, int]] = []
    for y in sorted(tick_rows):
        if merged and y - merged[-1][0] <= 3:
            py, pl = merged[-1]
            merged[-1] = ((py + y) // 2, max(pl, tick_rows[y]))
        else:
            merged.append((y, tick_rows[y]))
    return merged if with_len else [y for y, _ in merged]


def filter_equidistant_ticks(
    ticks: list[int], n: int | None = None, tol: int = 5
) -> list[int]:
    """用等距先验过滤刻度候选，返回等距的真实刻度子集。

    真实刻度一定等距，噪声（柱子顶/网格线）不落在等距网格上。n 是目标刻度
    数（vision 读的数量），指定时找「支持数量最接近 n」的网格；否则找支持
    数量最多的网格。枚举所有候选间距 + 基准，用等距网格投票。
    """
    if len(ticks) < 3:
        return sorted(ticks)
    ticks = sorted(ticks)
    steps = sorted({
        ticks[j] - ticks[i]
        for i in range(len(ticks))
        for j in range(i + 1, len(ticks))
        if ticks[j] - ticks[i] >= 20
    })
    if not steps:
        return ticks
    best: list[int] = ticks[:1]
    best_score: tuple | None = None
    for step in steps:
        for base in ticks:
            inliers = sorted(
                t for t in ticks
                if abs((t - base) - round((t - base) / step) * step) <= tol
            )
            cnt = len(inliers)
            if cnt < 3 or (n is not None and cnt > n + 1):  # 至少3个，容忍 vision 少读1个
                continue
            # 相邻 inlier 可能隔着 1、2、3… 个 step（真实刻度被漏检是常事），
            # 偏差要对着「最近的整数倍 step」算，不能直接比 step 本身——否则
            # 一个离群点造成的大间隔会把整条正确网格误杀（实测：图2(a) 的
            # 64 网格含一个 127≈2×64 的间隔被整条丢掉，只剩稀疏的 128 网格）
            gaps = [inliers[i + 1] - inliers[i] for i in range(cnt - 1)]
            dev = max((abs(g - round(g / step) * step) for g in gaps), default=0)
            rel_dev = dev / step if step > 0 else 0
            if rel_dev > 0.05:  # 相对间距偏差>5% → 噪声网格，排除
                continue
            score = (-cnt, rel_dev)  # 支持数量优先，tie 时相对偏差小优先
            if best_score is None or score < best_score:
                best_score = score
                best = inliers
    return sorted(best)


def _longest_true_run(column, bridge: int = 2) -> tuple[int, int]:
    """一维布尔序列里最长连续 True 段的 (start, end)（闭区间）；无则 (0, -1)。

    bridge>0 时先桥接 ≤bridge 像素的小缝隙（抗锯齿 / 交叉线 / mask 容差没盖住的
    1~2px 断点）再取最长段——否则 y 轴一处断开就只返回半条轴，bounds 全偏。
    只桥接极小缝隙，不会把远处的邻图内容并进来。
    """
    n = len(column)
    best_start, best_end, best_len = 0, -1, 0
    i = 0
    while i < n:
        if not column[i]:
            i += 1
            continue
        start = last = i
        j = i + 1
        gap = 0
        while j < n:
            if column[j]:
                last, gap = j, 0
            elif gap < bridge:
                gap += 1
            else:
                break
            j += 1
        if last - start > best_len:
            best_len, best_start, best_end = last - start, start, last
        i = last + 1
    return best_start, best_end


def _find_y_axis_col(
    mask: "np.ndarray", min_axis_len: int, rel: float = 0.6
) -> tuple[int, int, int] | None:
    """找 y 轴列，返回 (x, y_top, y_bottom)；找不到返回 None。

    取**最左**的「足够长 + 底端接上 x 轴」的垂直线：

    - 不能取「最长」——实测：一条比 y 轴长十几像素的**竖网格线**就能把它骗走；
    - 光取「最左」也不行——实测：左边缘一条杂散竖线会被误选；
    - 所以还得要求它的**底端附近有一条长水平线（x 轴）从它这里起步**——y 轴和
      x 轴本就相交，这是硬几何约束，杂散线接不上 x 轴就落选。
    """
    h, w = mask.shape
    runs: list[tuple[int, int, int]] = []
    best_len = 0
    for x in range(w):
        s, e = _longest_true_run(mask[:, x])
        runs.append((x, s, e))
        best_len = max(best_len, e - s + 1)
    if best_len < min_axis_len:
        return None
    thresh = max(min_axis_len, round(rel * best_len))
    cands = [(x, s, e) for x, s, e in runs if e - s + 1 >= thresh]
    # 首选：底端接上 x 轴（两轴相交是硬几何事实）。从最左往右，第一个接上的就是。
    for x, s, e in cands:
        for y in range(max(0, e - 6), min(h, e + 7)):
            hs, he = _longest_true_run(mask[y, :])
            if he - hs + 1 >= min_axis_len and abs(hs - x) <= 12:
                return x, s, e
    # 兜底：x 轴被裁掉/不清晰、没人接得上时，退化为「最左的足够长垂直线」。
    # （不能因为找不到 x 轴就放弃——那样会把真 y 轴让给某个恰好接上的柱边框）
    return cands[0] if cands else None


def _find_y_axis_col_near(
    mask: "np.ndarray", hint: int, min_axis_len: int, window: int = 10
) -> tuple[int, int, int] | None:
    """在 `hint` 附近找 y 轴列，返回 (x, y_top, y_bottom)；找不到返回 None。

    `_find_y_axis_col` 取「最左的合格竖线」，在**只含本块**的裁剪图上没问题；但外扩
    裁剪把左邻面板卷进来之后，邻面板的**右边框**可能比本块 y 轴还长，于是它被选中
    （实测 24_fig3 块4：选到 x=37，本块轴在 178）→ 刻度一条都检不出、标定失败。
    这里改成只在 hint ± window 里选**最长**的那条竖线，绕开邻面板。
    """
    h, w = mask.shape
    best: tuple[int, int, int] | None = None
    for x in range(max(0, hint - window), min(w, hint + window + 1)):
        s, e = _longest_true_run(mask[:, x])
        if best is None or (e - s + 1) > (best[2] - best[1] + 1):
            best = (x, s, e)
    if best is None or best[2] - best[1] + 1 < min_axis_len:
        return None
    return best


def _estimate_bg(arr: "np.ndarray", quant: int = 16) -> tuple[int, int, int]:
    """估背景色：量化后取整图出现次数最多的颜色（柱状图背景占绝大多数像素）。

    比写死「>240 算背景」通用——底色偏灰/带色（扫描件、截图、JPEG 伪影）也不会误判。
    """
    q = (arr // quant).astype(np.int32)
    key = q[:, :, 0] * 65536 + q[:, :, 1] * 256 + q[:, :, 2]
    vals, counts = np.unique(key, return_counts=True)
    k = int(vals[int(counts.argmax())])
    half = quant // 2
    return ((k >> 16 & 255) * quant + half, (k >> 8 & 255) * quant + half, (k & 255) * quant + half)


def _h_line_thickness(not_bg: "np.ndarray", y: int, min_run: int) -> int:
    """量 y 那一行的长横线在垂直方向有多厚（即 x 轴的线宽，用于清基线）。"""
    def is_long(yy: int) -> bool:
        s, e = _longest_true_run(not_bg[yy, :])
        return e - s + 1 >= min_run

    if not is_long(y):
        return 1
    t = 1
    yy = y - 1
    while yy >= 0 and is_long(yy):
        t += 1
        yy -= 1
    yy = y + 1
    while yy < not_bg.shape[0] and is_long(yy):
        t += 1
        yy += 1
    return t


def locate_plot_bounds(
    img: Image.Image,
    tick_color: tuple[int, int, int] = (0, 0, 0),
    tol: int = AXIS_TOL,
    min_axis_len: int = 30,
) -> tuple[int, int, int, int] | None:
    """定位绘图区（轴框）精确边界 (x0, y0, x1, y1)。

    y 轴 = 「最长连续垂直段」所在列（不是整列的 min/max——那样会被挤在同一
    裁剪框里的下方邻图/子图污染，把 x 轴找错）。该段顶端 = 绘图区上边界，
    底端 = x 轴位置；x 轴取其底端附近（±8px）的最长水平线。找不到返回 None。
    """
    arr = np.array(img.convert("RGB")).astype(int)
    r, g, b = tick_color
    mask = (
        (abs(arr[:, :, 0] - r) <= tol)
        & (abs(arr[:, :, 1] - g) <= tol)
        & (abs(arr[:, :, 2] - b) <= tol)
    )
    h, w = mask.shape

    # y 轴：最左的「足够长」垂直线（不能取最长——竖网格线/图例框边可能比它长）
    axis = _find_y_axis_col(mask, min_axis_len)
    if axis is None:
        return None
    axis_x, y_top, y_bottom = axis

    # x 轴：y 轴底端附近（±8px）、且**左端接上 y 轴**的最长水平线。
    # 不要求「接上 y 轴」的话，可能选中左侧无关的长横线（表格框/邻图边），
    # 其右端甚至落在 y 轴左边 → 裁出 right < left 的非法框（实测直接 crash）
    axis_y, best_x_len, x_right = y_bottom, 0, 0
    for y in range(max(0, y_bottom - 8), min(h, y_bottom + 8)):
        s, e = _longest_true_run(mask[y, :])
        # 只要求「右端在 y 轴右侧」就能防住非法框；不能要求「左端贴 y 轴」——
        # 轴中间有断点时最长那段左端会离 y 轴很远，会挑错线（实测图4 因此柱数归零）
        if e > axis_x and e - s + 1 > best_x_len:
            best_x_len, axis_y, x_right = e - s + 1, y, e
    if best_x_len < min_axis_len:
        return None

    return (axis_x + 1, y_top, int(x_right), axis_y)


def _split_by_coltop(
    col_top: dict[int, int],
    min_width: int,
    max_width: int,
    step_tol: int,
    trace: list | None = None,
) -> list[tuple[int, int, int]]:
    """把连通域的「每列最高像素」阶梯按跳变拆成单根柱子。

    相邻柱子粘成一个连通域时，柱顶高度不同 → col_top 在交界处跳变，据此
    切回单根。返回 [(x0, x1, y_top)]。
    """
    xs = sorted(col_top)
    if not xs:
        return []
    segments: list[tuple[int, int]] = []
    seg_start = prev_x = xs[0]
    prev_top = col_top[xs[0]]
    for x in xs[1:]:
        if x - prev_x > 3 or abs(col_top[x] - prev_top) > step_tol:
            segments.append((seg_start, prev_x))
            seg_start = x
        prev_x, prev_top = x, col_top[x]
    segments.append((seg_start, prev_x))
    out: list[tuple[int, int, int]] = []
    for s, e in segments:
        w = e - s + 1
        cols = [col_top[x] for x in range(s, e + 1) if x in col_top]
        if min_width <= w <= max_width and cols:
            out.append((s, e, min(cols)))
            if trace is not None:
                trace.append({"stage": "split_coltop", "action": "keep",
                              "box": (s, min(cols), e), "w": w,
                              "why": f"宽 {w} 在 [{min_width},{max_width}]"})
        elif trace is not None:
            trace.append({"stage": "split_coltop", "action": "drop",
                          "box": (s, (min(cols) if cols else None), e), "w": w,
                          "why": f"宽 {w} 不在 [{min_width},{max_width}]" if not (min_width <= w <= max_width) else "段内无列"})
    return out


def _runs_mask(m: "np.ndarray", length: int, axis: int) -> "np.ndarray":
    """「属于长度 ≥ length 的连续 True 段」的像素掩码（窗口和法，纯 numpy）。"""
    n = m.shape[1] if axis == 1 else m.shape[0]
    if length > n:
        return np.zeros_like(m)
    if axis == 1:
        cs = np.cumsum(m, axis=1)
        win = cs[:, length:] - cs[:, :-length]
        st = np.zeros_like(m)
        st[:, : n - length] = win == length
        cs2 = np.cumsum(st, axis=1)
        left = np.zeros_like(cs2)
        left[:, length:] = cs2[:, :-length]
        return (cs2 - left) > 0
    cs = np.cumsum(m, axis=0)
    win = cs[length:, :] - cs[:-length, :]
    st = np.zeros_like(m)
    st[: n - length, :] = win == length
    cs2 = np.cumsum(st, axis=0)
    up = np.zeros_like(cs2)
    up[length:, :] = cs2[:-length, :]
    return (cs2 - up) > 0


def detect_plot_frames(
    img: Image.Image,
    h_ratio: float = 0.06,
    v_ratio: float = 0.05,
    min_pixels: int = 500,
    min_height_ratio: float = 0.03,
) -> list[tuple[int, int, int, int]]:
    """用「长直线」找绘图区框（纯代码、确定性），替代让 vision 报 bbox。

    柱状图/折线图的轴 = 长横线 + 长竖线；正文文字只有短笔画，接不成长线。表格横线
    虽长，但**又扁又薄**（几像素高）→ 靠「最小高度」一筛就掉。所以「长横线 ∪ 长竖线」
    的连通域，天然就是每个图的轴框。

    为什么要代码来切：vision 对图文混排的多面板页会**按格子均分**，把表格切进来、
    把图竖切两半、面板张冠李戴（实测论文2 图1-2）。几何是它的软肋，是代码的强项。
    返回 [(x0, y0, x1, y1)]，按从上到下、从左到右；找不到返回 []（调用方回退 vision）。
    """
    arr = np.array(img.convert("RGB")).astype(int)
    h, w = arr.shape[:2]
    nonbg = np.abs(arr - np.array(_estimate_bg(arr))).max(axis=2) > 25
    frame = _runs_mask(nonbg, max(8, round(h_ratio * w)), 1) | _runs_mask(
        nonbg, max(8, round(v_ratio * h)), 0
    )

    # 像素底只用来滤碎屑；**滤表格线靠最小高度**——表格横线又长又薄（几像素高），
    # 拿面积当门槛会连真图框一起滤掉（实测 0.01×整页 就把 4 个真框全滤没了）
    min_h = max(8, round(min_height_ratio * h))
    seen = np.zeros_like(frame, dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []
    for i in range(h):
        for j in range(w):
            if not frame[i, j] or seen[i, j]:
                continue
            q = deque([(i, j)])
            seen[i, j] = True
            mi = ma = i
            mj = mja = j
            n = 1
            while q:
                y, x = q.popleft()
                mi, ma = min(mi, y), max(ma, y)
                mj, mja = min(mj, x), max(mja, x)
                n += 1
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and frame[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            if n >= min_pixels and (ma - mi + 1) >= min_h:
                boxes.append((mj, mi, mja, ma))
    boxes.sort(key=lambda b: (b[1], b[0]))
    # 相对尺寸过滤：图例小框/装饰框的面积远小于真图框，按最大框的比例一刀切掉
    # （真面板之间不会差一个数量级）
    if boxes:
        biggest = max((b[2] - b[0]) * (b[3] - b[1]) for b in boxes)
        boxes = [b for b in boxes if (b[2] - b[0]) * (b[3] - b[1]) >= 0.25 * biggest]
        boxes.sort(key=lambda b: (b[1], b[0]))
    return boxes


def _strip_error_bars(
    segs: list[tuple[int, int, int]], step_tol: int, trace: list | None = None
) -> list[tuple[int, int, int]]:
    """把误差棒从柱子里摘掉，并把被它劈开的柱子两半合回来。

    误差棒（柱顶中央那根细竖线 + 横帽）在 col_top 上形成「**更窄、更高**」的一段，
    把一根柱子劈成左右两半——两半的柱顶**完全相同**（都是真柱顶）。所以：

    ① 丢掉「宽度 < 中位宽一半，且柱顶比左右邻居都高」的段 → 那是误差棒本身；
    ② 合并柱顶相同(±step_tol)、且紧挨着(间隙≤3)的相邻段 → 那是同一根柱的两半。

    纯结构判据，不看颜色——所以对彩色柱、白柱一样管用。借自 thu-digitizer 的
    「误差棒当独立几何」：它更高更窄，能和柱体分开。
    """
    if len(segs) < 2:
        return segs
    segs = sorted(segs)
    widths = sorted(e - s + 1 for s, e, _ in segs)
    med_w = widths[len(widths) // 2]

    kept: list[tuple[int, int, int]] = []
    for i, (s, e, top) in enumerate(segs):
        w = e - s + 1
        if w < 0.5 * med_w:  # 窄段：看它是不是比左右邻居都高（更高=top 更小）
            left = segs[i - 1] if i > 0 else None
            right = segs[i + 1] if i + 1 < len(segs) else None
            higher_than = [
                nb[2] for nb in (left, right) if nb is not None and nb[2] > top + step_tol
            ]
            if higher_than:
                if trace is not None:
                    trace.append({"stage": "strip_errorbar", "action": "drop",
                                  "box": (s, top, e), "w": w,
                                  "why": "窄且比左右邻居都高 → 判为误差棒"})
                continue  # 误差棒：丢
        kept.append((s, e, top))

    # 合并柱顶相同的相邻段（被误差棒劈开的两半）。间隙阈值按**段宽的半**算：
    # 两半之间的空隙 = 误差棒宽度（约半根柱宽的 1/3），而不同柱子之间是**柱间距**
    # （明显更宽）。写死 3px 会漏（实测两半隔了 16px，误差棒自己的宽度）。
    out: list[tuple[int, int, int]] = []
    for s, e, top in kept:
        if out:
            ps, pe, ptop = out[-1]
            if abs(top - ptop) <= step_tol and s - pe <= 0.5 * (pe - ps + 1):
                if trace is not None:
                    trace.append({"stage": "strip_errorbar", "action": "merge",
                                  "box": (s, top, e), "w": e - s + 1,
                                  "why": f"柱顶与上一段相同(±{step_tol})且紧邻 → 合并（同一根柱被误差棒劈开）"})
                out[-1] = (ps, e, ptop)
                continue
        out.append((s, e, top))
    return out


def _split_wide_segments(
    bars: list[tuple[int, int, int]], min_width: int, max_width: int,
    trace: list | None = None,
) -> list[tuple[int, int, int]]:
    """把疑似「多根等高柱粘连」的宽 segment 按单根宽等分。

    等高相邻柱柱顶齐平 → col_top 无跳变 → 高度跳变切不开，只能靠宽度先验。

    单根宽取**出现次数最多的宽度**（众数，tie 取较小），**不能用最小宽**——一个
    窄噪声段会把单根宽估小，导致所有柱被过度等分（实测图2(c) 曾因此炸成 21 根）。
    宽度 ≥1.5×单根宽才拆。

    局限：整图全是粘连柱、找不到单根参照时拆不了。
    """
    if len(bars) < 2:
        return bars
    counts: dict[int, int] = {}
    for b in bars:
        w = b[1] - b[0] + 1
        counts[w] = counts.get(w, 0) + 1
    top_count = max(counts.values())
    unit = min(w for w, c in counts.items() if c == top_count)
    if trace is not None:
        trace.append({"stage": "split_wide", "action": "info", "box": None, "w": unit,
                      "why": f"单根宽估计 unit={unit}（众数），入口 {len(bars)} 段"})
    if unit < min_width:
        if trace is not None:
            trace.append({"stage": "split_wide", "action": "skip", "box": None, "w": unit,
                          "why": f"unit={unit} < min_width={min_width} → 单根宽不可信，整体不拆"})
        return bars  # 单根宽不可信，宁可不拆
    out: list[tuple[int, int, int]] = []
    for x0, x1, top in bars:
        w = x1 - x0 + 1
        k = round(w / unit)
        if k >= 2 and w >= 1.5 * unit and w <= max_width * k:
            step = w / k
            for i in range(k):
                out.append((
                    int(round(x0 + i * step)),
                    int(round(x0 + (i + 1) * step)) - 1,
                    top,
                ))
            if trace is not None:
                trace.append({"stage": "split_wide", "action": "split",
                              "box": (x0, top, x1), "w": w,
                              "why": f"宽 {w} ≈ {k}×unit({unit}) → 切成 {k} 根"})
        else:
            out.append((x0, x1, top))
            if trace is not None:
                trace.append({"stage": "split_wide", "action": "keep",
                              "box": (x0, top, x1), "w": w,
                              "why": f"宽 {w} / unit({unit}) = {w / unit:.2f} → 不拆"})
    return out


def detect_gray_bars(
    img: Image.Image,
    bounds: tuple[int, int, int, int],
    bg_tol: int = 20,
    min_width: int | None = None,
    max_width: int | None = None,
    min_height: int | None = None,
    step_tol: int | None = None,
    baseline_gap: int | None = None,
    trace: list | None = None,
) -> list[tuple[int, int, int]]:
    """灰色系柱状图柱体检测（不依赖柱身颜色，兼容白柱/深柱/灰度渐变）。

    柱子必然底端贴 x 轴；用「背景色外的连通域 + 底端贴基线」找柱体——表格说明
    框、坐标文字、图题都不贴基线，天然被滤除。相邻柱子粘成一个连通域时，用柱顶
    阶梯跳变拆回单根。返回 [(x0, x1, y_top)]，按 x 排序。

    阈值默认**按绘图区尺寸自适应**（宽/高/步长 = 绘图区的比例；背景色从整图估；
    基线厚度实测），不写死像素——换分辨率/DPI 的图也能用。
    """
    x0, y0, x1, y1 = bounds
    arr = np.array(img.convert("RGB")).astype(int)
    h, w = arr.shape[:2]

    # 背景色：量化后取众数（背景占绝大多数像素），比写死「>240」通用
    bg = np.array(_estimate_bg(arr))
    not_bg = ~(np.abs(arr - bg) <= bg_tol).all(axis=2)

    plot_w, plot_h = max(1, x1 - x0), max(1, y1 - y0)
    if min_width is None:
        min_width = max(4, round(0.02 * plot_w))
    if max_width is None:
        max_width = round(0.9 * plot_w)
    if min_height is None:
        min_height = max(4, round(0.03 * plot_h))
    if step_tol is None:
        step_tol = max(3, round(0.015 * plot_h))
    if baseline_gap is None:
        baseline_gap = max(2, _h_line_thickness(not_bg, y1, max(10, plot_w // 2)))

    m = not_bg.copy()
    m[max(0, y1 - baseline_gap): y1 + baseline_gap + 1, :] = False  # 清基线，断 x 轴
    m[:, : x0 + 3] = False                                          # 清 y 轴列
    m[:y0, :] = False
    m[y1:, :] = False
    m[:, x1 + 1:] = False   # 右侧也清：BFS 种子虽限于 [x0,x1]，组件却能向右溢出，
                            # 把裁剪框里右侧的邻图/表格框并进柱子（裁剪越紧越容易中）

    visited = np.zeros_like(m, dtype=bool)
    bars: list[tuple[int, int, int]] = []
    for i in range(y0, y1):
        for j in range(x0, min(x1 + 1, w)):
            if not m[i, j] or visited[i, j]:
                continue
            q = deque([(i, j)])
            visited[i, j] = True
            col_top: dict[int, int] = {}
            max_y = i
            while q:
                y, x = q.popleft()
                if y > max_y:
                    max_y = y
                if x not in col_top or y < col_top[x]:
                    col_top[x] = y
                for dy, dx in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and m[ny, nx] and not visited[ny, nx]:
                        visited[ny, nx] = True
                        q.append((ny, nx))
            cx0, cx1 = min(col_top), max(col_top)
            ctop = min(col_top.values())
            if trace is not None:
                trace.append({"stage": "component",
                              "action": "keep" if (y1 - 8 <= max_y <= y1 + 2) else "drop",
                              "box": (cx0, ctop, cx1, max_y), "w": cx1 - cx0 + 1,
                              "h": max_y - ctop + 1, "px": len(col_top),
                              "why": "底端贴基线" if (y1 - 8 <= max_y <= y1 + 2)
                                     else f"底端 {max_y} 不贴基线({y1})"})
            if not (y1 - 8 <= max_y <= y1 + 2):  # 底端不贴基线 → 非柱子
                continue
            segs = _strip_error_bars(
                _split_by_coltop(col_top, min_width, max_width, step_tol, trace),
                step_tol, trace,
            )
            for b in segs:
                if y1 - b[2] >= min_height:  # 太矮 → 邻图碎片/噪声，不是柱子
                    bars.append(b)
                elif trace is not None:
                    trace.append({"stage": "min_height", "action": "drop",
                                  "box": (b[0], b[2], b[1]), "w": b[1] - b[0] + 1,
                                  "h": y1 - b[2],
                                  "why": f"高 {y1 - b[2]} < min_height({min_height})"})
    bars.sort()
    return _split_wide_segments(bars, min_width, max_width, trace)
