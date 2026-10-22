"""多模态图数据提取 —— vision 分类/OCR + 代码反算的集成入口。

路由逻辑（能力路由，避开 vision 估读几何的软肋）：
① vision 分析图（类型 + 有无数字标签 + 系列颜色）
② 柱顶/扇区印了数字 → vision OCR 读数值（100% 准）
③ 无数字标签的数据图 → 代码标定 + 反算（精确，后续接刻度检测）
④ 谱图/显微图 → vision 定性描述
"""

from __future__ import annotations

import base64
import json
import logging
import math
import os
import re
from pathlib import Path

import numpy as np
from PIL import Image
from openai import OpenAI

from .config import settings

logger = logging.getLogger(__name__)


def _vision_call(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    client = OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)
    img_b64 = base64.b64encode(image_bytes).decode()
    response = client.chat.completions.create(
        model=settings.vision_model,
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:{mime_type};base64,{img_b64}"}},
            ],
        }],
        temperature=settings.temperature,  # 默认 0：vision 结果可复现（否则每次跑都抖）
    )
    return response.choices[0].message.content or ""


def _mime(img_path: str) -> str:
    return "image/png" if str(img_path).lower().endswith(".png") else "image/jpeg"


# 代码切框给的框 = 「长横线 ∪ 长竖线」的连通域**外接矩形**，它把轴线的像素也算在
# 里面；而下游 `detect_gray_bars` 要的是 `locate_plot_bounds` 的语义——绘图区
# **内部**（y 轴线右侧、x 轴线上方）。两者差一个「线宽」，所以往里收这几像素。
#
# 实测（24_fig3 四块）：安全余量很宽（x0 可偏 +1~+8、y1 可偏 -4~0 都不变），
# 但**收到 0 就会出错**——块3 那根紧贴 y 轴的柱组被连根切掉，15 根掉成 12 根。
_FRAME_INSET = (2, 2, 1, 2)  # 左 / 上 / 右 / 下


def _frame_to_plot_bounds(
    frame: tuple[int, int, int, int], crop_box: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    """把 `detect_plot_frames` 的框（**原图坐标**）换算成 crop 内的绘图区 bounds。

    这是整条链上**唯一一次**坐标换算：输入原图坐标，输出 crop 坐标，之后再往下传
    的全程都是 crop 坐标（下游拿到的是同一张裁剪图），不许再加偏移。
    """
    x0, y0, x1, y1 = frame
    left, top, right, bottom = _FRAME_INSET
    return (x0 - crop_box[0] + left, y0 - crop_box[1] + top,
            x1 - crop_box[0] - right, y1 - crop_box[1] - bottom)


def _parse_json(raw: str) -> dict:
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]
    return json.loads(raw)


def analyze_figure(img_path: str) -> dict:
    """vision 分析：图类型 + 有无数字标签 + 系列数 + 系列颜色。"""
    prompt = (
        "分析这张图，返回 JSON（只输出 JSON，不要其他文字）：\n"
        '{"type": "bar/pie/line/scatter/spectrum/microscopy/other", '
        '"has_labels": true/false, "series_count": 整数, "colors": ["#RRGGBB"], '
        '"description": "一句话"}\n'
        "type 是图类型；has_labels 表示数据点/柱子/扇区上是否直接印了数值标签；"
        "series_count 是**数据系列数**（分组柱状图里一个 x 位置上有几根柱子/图例有几项；"
        "单系列填 1）；colors 是数据系列颜色（柱状图/折线图填柱子或线颜色，谱图/显微图填空数组）。"
    )
    raw = _vision_call(Path(img_path).read_bytes(), _mime(img_path), prompt)
    return _parse_json(raw)


def detect_subfigures(img_path: str) -> list[dict]:
    """vision 识别一张图里的多个子图，返回 [{label, type, bbox(归一化 0~1)}]。"""
    prompt = (
        "这张图可能包含多个子图（几个小图拼在一起）。请识别每个子图，返回 JSON（只输出 JSON）：\n"
        '{"subfigures": [{"label": "子图标签", "type": '
        '"bar/line/scatter/pie/table/spectrum/microscopy/other", "bbox": [左, 上, 右, 下]}]}\n'
        "bbox 用归一化坐标（0~1，相对整张图的宽和高，左上角 (0,0)）。"
        "**按真实图块边界框，不要按格子均分**：表格标 table、正文段落标 other"
        "（都不是数据图，后续会跳过）；同一图号下的多个小面板（如 (a)(b)(c)）分开列。"
        "按从上到下、从左到右的顺序列出。若只有一个图，返回该图一个子图。"
    )
    raw = _vision_call(Path(img_path).read_bytes(), _mime(img_path), prompt)
    try:
        data = _parse_json(raw)
    except json.JSONDecodeError:
        return []
    return data.get("subfigures", [])


def figure_to_text(img_path: str, figure_label: str = "") -> str:
    """把一张图（可能含多个子图）转成文本描述，供 LLM 抽取时「融入」全文。

    先 vision 报粗 bbox 分割子图（解决错位布局），柱状图再代码找轴框精定位
    裁剪（轴框不受布局影响）；谱图/显微图无轴框，直接 vision 定性描述。
    """
    import tempfile

    from .chart_extract import detect_plot_frames, locate_plot_bounds

    img = Image.open(img_path)
    w, h = img.size
    # **代码切优先**：长直线找轴框，确定性、不会「按格子均分」。切成 ≥2 块就用它。
    # 切不出（<2 块：单图，或图里没有长直线）才回退 vision 报 bbox。
    frames = detect_plot_frames(img)
    code_cut = len(frames) >= 2
    if code_cut:
        blocks: list[tuple[list[float], str | None, str]] = [
            ([(x0) / w, (y0) / h, (x1) / w, (y1) / h], None, f"子图{i + 1}")
            for i, (x0, y0, x1, y1) in enumerate(frames)
        ]
    else:
        subfigs = detect_subfigures(img_path)
        if len(subfigs) <= 1:
            return _single_figure_to_text(img_path, figure_label)
        blocks = [
            (sf.get("bbox", [0, 0, 1, 1]), sf.get("type"), sf.get("label") or f"子图{i + 1}")
            for i, sf in enumerate(subfigs)
        ]

    texts: list[str] = []
    tmpdir = Path(tempfile.mkdtemp())
    for i, (bbox, stype, label) in enumerate(blocks):
        try:
            x0, y0, x1, y1 = (float(v) for v in bbox)
        except (ValueError, TypeError):
            continue
        # vision bbox 偏紧（常切掉 y 轴刻度数字/柱子），先外扩再精定位
        pad_ratio = 0.3
        dx = (x1 - x0) * pad_ratio
        dy = (y1 - y0) * pad_ratio
        ex_x0 = max(0.0, x0 - dx)
        ex_y0 = max(0.0, y0 - dy)
        ex_x1 = min(1.0, x1 + dx)
        ex_y1 = min(1.0, y1 + dy)
        crop_box = (int(ex_x0 * w), int(ex_y0 * h), int(ex_x1 * w), int(ex_y1 * h))
        crop = img.crop(crop_box)
        # **代码切的块**：detect_plot_frames 给的框本身就是轴框，换算一次直接往下传，
        # 不再让 locate_plot_bounds 重新定位——实测它会被左邻面板的右边框骗走
        # （那条竖线比本块 y 轴还长），把 24_fig3 块4 从 15 根炸成 31 根。
        # vision 报的块没有框 → plot_bounds=None → 下游照旧自己定位。
        plot_bounds = _frame_to_plot_bounds(frames[i], crop_box) if code_cut else None
        # vision 报的粗 bbox 偏紧，用轴框精定位修正；代码切的块**本身就是轴框**，免修
        if stype == "bar":
            tick_color = _detect_tick_color(crop)
            bounds = locate_plot_bounds(crop, tick_color=tick_color)
            if bounds:
                bx0, by0, bx1, by1 = bounds
                # 左侧须留足空间：y 轴刻度数字（如 "32"/"24"）在轴的左边，
                # 切掉会让 vision 把刻度值读错（0~32 读成 0~8 → 反算差 4 倍）。
                # 按图宽比例给（写死像素在小图上会把邻图/表格框也框进来）。
                pad_left = max(80, min(160, int(0.08 * crop.width)))
                pad_y = 40
                left = max(0, bx0 - pad_left)
                top = max(0, by0 - pad_y)
                right = min(crop.width, bx1 + pad_y)
                bottom = min(crop.height, by1 + pad_y)
                if right > left and bottom > top:  # 防非法框（PIL 会直接抛异常，整张图作废）
                    crop = crop.crop((left, top, right, bottom))
        tmp_path = tmpdir / f"subfig_{i}.png"
        crop.save(tmp_path)
        t = _single_figure_to_text(str(tmp_path), f"{figure_label}{label}", plot_bounds)
        tmp_path.unlink(missing_ok=True)
        if t:
            texts.append(t)
    return "\n\n".join(texts)


def _single_figure_to_text(
    img_path: str,
    figure_label: str = "",
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> str:
    """处理单张图：无标签柱状图走代码反算，其余 vision 读图转文本。

    plot_bounds 是**这张裁剪图内**的绘图区边界（代码切框时由调用方算好）；
    None 表示没有框先验（vision 报的块 / 单图），下游自己定位。
    """
    analysis = analyze_figure(img_path)
    ftype = analysis.get("type")
    # type=other/table（表格块 / 正文块，不是数据图）→ 直接跳过。否则会发「提取数值」
    # 的兜底 prompt，vision 就把表格数字当柱子值吐出来（实测：表4 的行被读成
    # `12: [55, 45, 35]` 混进结果）。分类是 vision 的活，这里只是尊重它的结论。
    if ftype in ("other", "table"):
        return ""
    # 无标签柱状图 → 代码反算（避免 vision 估读几何）
    if ftype == "bar" and not analysis.get("has_labels", True):
        colors = analysis.get("colors") or []
        if colors:
            series = extract_unlabeled_bar_multi(img_path, colors, plot_bounds=plot_bounds)
            # **系列数自检 —— 只警告，不否决。**
            # 本想拿它当一票否决（对不上就不输出），但实测**误伤太狠**：论文1 图4 那
            # 20 个值均误才 0.64%，却因为「代码数出 4 系列 vs vision 说 6」被全扔掉；
            # 论文2 图3-4 更冤——它是「单系列、4 根柱」，vision 报的"系列数"是 1 还是 4
            # **本来就说不清**。根因是 **「系列」的定义两边不一致**，判据本身有歧义，
            # 拿它一票否决 = 制造新的静默错误。所以只留作**信号**（日志里看得见）。
            expected = analysis.get("series_count")
            if isinstance(expected, int) and expected > 0 and len(series) != expected:
                logger.warning(
                    "图 %s：系列数对不上（代码 %d 个 vs vision 说 %d 个）——仅提示，不否决",
                    figure_label, len(series), expected,
                )
            usable = [s for s in series if s["confidence"] != "low"]
            if usable:  # 至少一个系列反算可信
                y_label = analysis.get("description", "")
                return _format_bar_text_multi(figure_label, y_label, usable)
            logger.warning("图 %s 代码反算失败（无可信系列），落回 vision 估读", figure_label)

    img_bytes = Path(img_path).read_bytes()
    prompt = (
        "分析这张图，提取数据。返回 JSON（只输出 JSON）：\n"
        '{"type": "bar/pie/line/scatter/spectrum/microscopy/other", '
        '"y_label": "纵轴标签(含义+单位)", "x_labels": ["横轴类别1", ...], '
        '"values": [数值1, ...], "description": "一句话描述"}\n'
        "柱顶/扇区/数据点上直接印了数值，就精确读出填进 values；"
        "没印数值但能看出数值，就估读填（标注约）；"
        "谱图/显微图 type 填 spectrum/microscopy，y_label/x_labels/values 填空。"
    )
    raw = _vision_call(img_bytes, _mime(img_path), prompt)
    try:
        data = _parse_json(raw)
    except json.JSONDecodeError:
        return ""
    return _to_text(data, figure_label)


def _to_text(data: dict, figure_label: str) -> str:
    """把图数据字典转成一段文本。"""
    ftype = data.get("type", "other")
    prefix = f"{figure_label} " if figure_label else ""

    if ftype in ("spectrum", "microscopy"):
        desc = data.get("description", "").strip()
        return f"{prefix}{desc}" if desc else ""

    y_label = data.get("y_label", "").strip()
    x_labels = data.get("x_labels", []) or []
    values = data.get("values", []) or []
    lines = []
    head = f"{prefix}纵轴: {y_label}".strip() if y_label else prefix.strip()
    if head:
        lines.append(head)
    for x, v in zip(x_labels, values):
        lines.append(f"- {x}: {v}")
    return "\n".join(lines) if lines else ""


def _vision_read_axis_values(img_path: str) -> list[float]:
    """vision 读 y 轴刻度数值（从下到上）。"""
    prompt = (
        "这是数据图。请读出 y 轴（纵轴）的刻度数值，从下到上依次列出"
        "（例如 0, 6, 12, 18, 24）。只输出数字列表（逗号分隔），不要其他文字。"
    )
    for _ in range(2):  # 读不到时重试一次（vision 读小数字不稳定）
        raw = _vision_call(Path(img_path).read_bytes(), _mime(img_path), prompt)
        raw = raw.replace("，", ",").replace("、", ",")
        try:
            vals = [float(x) for x in raw.split(",") if x.strip()]
            if vals:
                return vals
        except ValueError:
            continue
    return []  # 两次都读不到，返回空让标定失败落回兜底


def _vision_read_x_labels(img_path: str) -> list[str]:
    """vision 读横轴标签（组号/类别），从左到右。"""
    prompt = (
        "这是数据图。请读出横轴（x 轴）每个类别/组的标签，从左到右依次列出"
        "（例如 组1, 组2, 组3）。只输出标签列表（逗号分隔），不要其他文字。"
    )
    raw = _vision_call(Path(img_path).read_bytes(), _mime(img_path), prompt)
    raw = raw.replace("，", ",").replace("、", ",")
    return [x.strip() for x in raw.split(",") if x.strip()]


def _anchors_by_major_grid(
    ticks_len: list[tuple[int, int]],
    baseline_px: float | None,
    values: list[float],
    snap: int = 5,
    trace: dict | None = None,
) -> list[tuple[int, float]] | None:
    """按「主刻度网格」造标定锚点；造不出来返回 None（调用方回退老路径）。

    **为什么要它**：`detect_axis_ticks` 检出的线是「主刻度 + 次刻度」混在一起，
    老的 `filter_equidistant_ticks` 按「目标条数」抽网格时**相位会抽错**——实测
    24_fig3 四块全踩：抽到次刻度网格，锚点整体错开一整档，反算值偏一个刻度
    （±5% 命中 0/60）。这里换成：以 **baseline 为相位起点**、用主刻度候选的
    **最小间距**外推成网格。

    **线长只用来排候选次序**（取线长前一半），不当硬门槛——实测有一条真主刻度
    线长只有 5、其余是 8，按阈值会被误杀，按排序则自然留在前一半里。
    **判定主刻度靠间距规律**：缺项会把某个间距撑成 2 倍，取 `min(gaps)` 才不会把
    2 倍当步长。网格点对不上检出刻度的位置**推断补上**，并在 trace 里留痕——
    不把推断出来的位置悄悄当成实测刻度。
    """
    if not ticks_len or not values or baseline_px is None:
        return None
    base = float(baseline_px)
    rows = [(int(y), int(L)) for y, L in ticks_len if y < base - 5]
    if len(rows) < 2:
        return None
    by_len = sorted(rows, key=lambda t: -t[1])
    # 线长必须**真的分成两组**才敢「取长的一半」：只画主刻度的图（没有次刻度）线长
    # 全都一样，这时按长度排序是**任意的**，随便取一半会把相邻间距估成 2 倍，反算值
    # 直接减半（实测 论文2_图1-2 子图2：均误 1.88% → 12.89%）。分不开就取全部刻度，
    # 让间距自己去定步长。
    lens = sorted({L for _, L in rows})
    separated = (len(lens) >= 2
                 and max(lens[i + 1] - lens[i] for i in range(len(lens) - 1))
                 >= max(2, 0.25 * lens[-1]))
    cand = (sorted(y for y, _ in by_len[: max(1, len(by_len) // 2)])
            if separated else sorted(y for y, _ in rows))
    gaps = [cand[i + 1] - cand[i] for i in range(len(cand) - 1)]
    if not gaps:
        return None
    step = min(gaps)
    if step < 5:
        return None

    pts: list[tuple[int, str]] = []
    k = 1
    while len(pts) < len(values) - 1:
        target = base - k * step
        near = [y for y, _ in rows if abs(y - target) <= snap]
        pts.append((min(near, key=lambda y: abs(y - target)), "实测")
                   if near else (round(target), "推断"))
        k += 1
    inferred = [y for y, src in pts if src == "推断"]
    if len(inferred) > 0.5 * len(pts):  # 推断过半 → 步长估错了，别硬用
        return None
    ys = [int(base)] + [y for y, _ in pts]
    if trace is not None:
        trace.update({"原始刻度候选": rows, "主刻度候选": cand, "步长": step,
                      "网格": pts, "推断位置": inferred})
    return list(zip(ys, values))


def _calibrate_axis(
    img: Image.Image,
    img_path: str,
    tick_color: tuple[int, int, int] | None,
    baseline_px: float | None = None,
    plot_bounds: tuple[int, int, int, int] | None = None,
    trace: dict | None = None,
):
    """检测刻度线 + vision 读刻度数值 → 标定。

    返回 (calibration, tick_color)，标定失败时 calibration 为 None
    （刻度线或数值不足 2 个锚点）。baseline_px 是柱子底端像素（=最小刻度
    位置），若检测漏了底部刻度（被柱底边遮挡），用它补上底部锚点。

    锚点**优先走 `_anchors_by_major_grid`**（主刻度网格，理由见该函数）；
    造不出网格（刻度太少 / 没基线 / 步长不可信）才回退老的「等距过滤」路径。

    plot_bounds 有值时，把它的左沿当作 **y 轴列先验** 传给刻度检测——
    外扩裁剪会卷进左邻面板，邻面板的右边框可能比本块 y 轴还长，全局找轴会选错
    （实测 24_fig3 块4 因此刻度检出 0 条）。没有先验（vision 报的块）就照旧全局找。
    """
    from .chart_extract import AxisCalibration, detect_axis_ticks, filter_equidistant_ticks

    if tick_color is None:
        tick_color = _detect_tick_color(img)
    values = _vision_read_axis_values(img_path)
    axis_hint = (plot_bounds[0] - 1) if plot_bounds else None
    ticks_len = detect_axis_ticks(
        img, tick_color=tick_color, tol=20, min_run_len=3, with_len=True,
        axis_hint=axis_hint,
    )
    anchors = _anchors_by_major_grid(ticks_len, baseline_px, values, trace=trace)
    if anchors:
        if trace is not None:
            trace["锚点来源"] = "主刻度网格"
        return AxisCalibration(anchors), tick_color

    # ---- 回退：老的等距过滤路径（证据不足时的将就，**可信度显著更低**）----
    if trace is not None:
        trace["锚点来源"] = "回退等距过滤"
    ticks = sorted((y for y, _ in ticks_len), reverse=True)
    ticks = filter_equidistant_ticks(ticks, n=len(values), tol=5)  # 目标刻度数约束
    ticks = sorted(ticks, reverse=True)  # 过滤后恢复「从下到上」顺序
    # 补底部基线锚点：柱底贴基线（最小刻度），漏底部刻度时用 baseline 补上
    if baseline_px is not None and ticks and baseline_px > max(ticks) + 5:
        ticks = [baseline_px] + ticks
    # 漏刻度检查：检测到的刻度数明显少于 vision 读的数值数，标定锚点不完整
    if len(ticks) >= 3 and len(values) - len(ticks) >= 2:
        if trace is not None:
            trace["锚点来源"] = "标定失败"
        return None, tick_color
    n = min(len(ticks), len(values))
    if n < 2:
        if trace is not None:
            trace["锚点来源"] = "标定失败"
        return None, tick_color
    # 刻度间距均匀性：相邻间距不应有异常小值（重复检测同一刻度线会混入噪声）
    if n >= 3:
        gaps = [abs(ticks[i + 1] - ticks[i]) for i in range(n - 1)]
        median_gap = sorted(gaps)[len(gaps) // 2]
        if min(gaps) < 0.3 * median_gap:
            if trace is not None:
                trace["锚点来源"] = "标定失败"
            return None, tick_color  # 混入噪声刻度，标定不可信
    anchors_fb = list(zip(ticks[:n], values[:n]))
    return AxisCalibration(anchors_fb), tick_color


def _assess_confidence(
    results: list[tuple[float, float]],
    labels: list[str],
) -> str:
    """反算结果置信度：值几乎全相同/空=low，柱子数与标签数不符=medium，否则 high。"""
    if not results:
        return "low"
    nums = [v for _, v in results]
    lo, hi = min(nums), max(nums)
    spread = hi - lo
    scale = max(abs(lo), abs(hi))
    if spread <= 1e-6 or (scale > 0 and spread / scale < 0.02):
        return "low"  # 反算值几乎全相同 → 假柱子/文字笔画
    if labels and len(results) != len(labels):  # 柱子数 ≠ 标签数 → 漏检/混入噪声
        return "medium"
    return "high"


def _extract_one_series(
    img: Image.Image,
    calibration,
    rgb: tuple[int, int, int],
    labels: list[str],
) -> tuple[list[tuple[str | None, float]], str]:
    """单色柱子检测 + 反算 + 置信度。返回 ([(label, value)...], confidence)。"""
    from .chart_extract import detect_bars_baseline

    bars, baseline = detect_bars_baseline(img, rgb, tol=50)
    if baseline is None:
        return [], "low"
    results: list[tuple[float, float]] = []
    for x0, y0, x1, y1 in bars:
        x_center = (x0 + x1) / 2
        top_dist = abs(y0 - baseline)
        bottom_dist = abs(y1 - baseline)
        data_end = y0 if top_dist >= bottom_dist else y1
        results.append((x_center, calibration.value_at_pixel(float(data_end))))
    results.sort(key=lambda t: t[0])
    confidence = _assess_confidence(results, labels)
    if len(labels) == len(results):
        labeled = [(label, v) for label, (_, v) in zip(labels, results)]
    else:
        labeled = [(None, v) for _, v in results]
    return labeled, confidence


def _extract_gray_series(
    img: Image.Image,
    img_path: str,
    tick_color: tuple[int, int, int] | None,
    plot_bounds: tuple[int, int, int, int] | None = None,
    trace: dict | None = None,
) -> list[dict]:
    """灰色系单系列柱状图提取：几何法柱体检测 + 标定反算。

    用于柱身颜色不可分的图（白柱、灰度渐变，如论文2 图3/图4）——颜色法对
    它们无解，改走几何：轴框定位 → 触基线连通域检柱 → 反算。返回单元素
    series 列表（标定/检柱失败返回空）。

    plot_bounds 非空时**直接用**（代码切框给的先验，已是本裁剪图的坐标），
    不再 `locate_plot_bounds` 重新定位——那次重新定位会被左邻面板的右边框骗走。
    """
    from .chart_extract import detect_gray_bars, locate_plot_bounds

    tc = tick_color or _detect_tick_color(img)
    bounds = plot_bounds or locate_plot_bounds(img, tick_color=tc)
    if bounds is None:
        return []
    labels = _vision_read_x_labels(img_path)
    bars = detect_gray_bars(img, bounds)
    if not bars:
        return []
    calibration, _ = _calibrate_axis(img, img_path, tc, float(bounds[3]), bounds, trace)
    if calibration is None:
        return []
    results = [
        ((b[0] + b[1]) / 2, calibration.value_at_pixel(float(b[2]))) for b in bars
    ]
    results.sort(key=lambda t: t[0])
    confidence = _assess_confidence(results, labels)
    if len(labels) == len(results):
        labeled = [(label, v) for label, (_, v) in zip(labels, results)]
    else:
        labeled = [(None, v) for _, v in results]
    return [{"color": "gray", "values": labeled, "confidence": confidence}]


def _geometric_fallback_enabled() -> bool:
    """结果驱动降级的开关。**默认关**——正式输出保持原样，方便对照与回滚。

    开启（`PAPER2DATA_GEOMETRIC_FALLBACK=on`）时：逐色法一根都没捞到、但**有有效色**
    → 改试几何法；几何法也拿不出可用结果，才照旧落回 vision。

    ⚠️ **默认关的理由**（这三条都还没解决）：
      · 「非柱状图 + 有有效色 + 逐色零输出」这个负样本类**还没收集到** → 误触发风险未验证
      · 下游 LLM 层的收益**实测为零**（配对实验：±5% 49→50、精确 21→16）
      · 「标定走回退路径」的块会被守卫拒收（如 24_fig3 子图1）—— 拒得对，但根因未解
    """
    return os.getenv("PAPER2DATA_GEOMETRIC_FALLBACK", "off").lower() in (
        "on", "1", "true", "yes",
    )


def _maybe_geometric_fallback(
    img: Image.Image,
    img_path: str,
    tick_color: tuple[int, int, int] | None,
    plot_bounds: tuple[int, int, int, int] | None,
    n_colors: int,
    why: str,
) -> list[dict]:
    """逐色法没拿到结果时，按「开关 + 守卫」决定要不要改走几何法。返回 series（可能为空）。

    **每一次判断都留痕**（logger），四种结局：开关关 / 没有有效色 / 试了但拿不出结果 /
    **被守卫拒收**。这样以后动 `_calibrate_axis` 时，能看出某块是不是悄悄进了几何输出。

    为什么要守卫：**「标定成功」不等于「标定可信」**。实测 24_fig3 子图1 的标定走的是
    **回退路径**（刻度只检出 3 条、主刻度网格造不出来），照样返回了锚点——但那是 2 个
    锚点、间距 45 与 238 混在一起，算出来的 15 个值全错。所以接管许可只发给**主刻度
    网格**造出来的标定；证据不足时宁可拒收，也不把错值送去下游。

    触发条件刻意不写成「图是不是灰的」——那个判据把「用哪把工具」和「图长什么样」绑死
    了，两边各有死区。这里用**实际结果**触发，再用**标定证据强度**把门。
    """
    if not _geometric_fallback_enabled():
        logger.info("图 %s：逐色法无结果（%s），几何兜底开关关闭 → 落回 vision", img_path, why)
        return []
    if not n_colors:
        logger.info("图 %s：逐色法无结果（%s）且无有效色 → 不试几何兜底", img_path, why)
        return []

    trace: dict = {}
    img_obj = Image.open(img_path)
    tc = tick_color or _detect_tick_color(img_obj)
    series = _extract_gray_series(img_obj, img_path, tc, plot_bounds, trace)
    usable = [s for s in series if s["confidence"] != "low"]
    if not usable:
        logger.info("图 %s：几何兜底尝试失败（标定/检柱无结果）→ 落回 vision", img_path)
        return []
    vals = [v for s in usable for _, v in s["values"]]
    if len(vals) < 2:
        logger.info("图 %s：几何兜底被拒（只出 %d 个值，不成图）", img_path, len(vals))
        return []
    if any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in vals):
        logger.info("图 %s：几何兜底被拒（出现非有限值）", img_path)
        return []
    if len({round(float(v), 6) for v in vals}) < 2:
        logger.info("图 %s：几何兜底被拒（值全相同 → 多半是假柱子/笔画）", img_path)
        return []
    if trace.get("锚点来源") != "主刻度网格":
        logger.warning(
            "图 %s：几何兜底被拒——标定走的是「%s」，证据不足，拒收以免把错值送去下游",
            img_path, trace.get("锚点来源", "未知"),
        )
        return []
    logger.info("图 %s：逐色法无结果（%s）→ 几何兜底接管，给出 %d 个值", img_path, why, len(vals))
    return usable


def extract_unlabeled_bar_multi(
    img_path: str,
    colors: list[str],
    tick_color: tuple[int, int, int] | None = None,
    plot_bounds: tuple[int, int, int, int] | None = None,
) -> list[dict]:
    """多色分组柱状图提取：标定共享，每色检测柱子 + 反算，带置信度。

    colors 是 hex 颜色字符串列表（analyze_figure 报的）。返回
    [{"color": hex, "values": [(label, value)...], "confidence": "high/medium/low"}...]；
    标定失败返回空列表。颜色全为灰色（单系列灰度渐变，如论文2 图3/图4）时
    走几何法——柱身颜色不可分（白柱/深柱），靠「触基线连通域」检柱。

    plot_bounds 是**这张裁剪图内**的绘图区边界（代码切框时算好的先验），
    传下去给几何法直接用；None 时几何法自己 `locate_plot_bounds` 定位。
    """
    from .chart_extract import _estimate_bg

    img = Image.open(img_path)
    # 背景色实测（不写死「近白 = 240」）：底色偏灰/带色的图也能正确排掉背景色
    bg = _estimate_bg(np.array(img.convert("RGB")).astype(int))

    def _near_bg(rgb: tuple[int, int, int]) -> bool:
        return max(abs(rgb[i] - bg[i]) for i in range(3)) <= 20

    rgbs = [
        rgb for rgb in (_hex_to_rgb(c) for c in colors)
        if rgb is not None and not _near_bg(rgb)
    ]
    if len(rgbs) >= 2 and all(
        abs(rgb[0] - rgb[1]) < 20 and abs(rgb[1] - rgb[2]) < 20 for rgb in rgbs
    ):
        return _extract_gray_series(img, img_path, tick_color, plot_bounds)

    # 先用第一个有效颜色检测柱子底端 baseline（=最小刻度位置，用于补漏底部刻度）
    from .chart_extract import detect_bars_baseline

    baseline_px = None
    for hex_color in colors:
        rgb = _hex_to_rgb(hex_color)
        if rgb is None or _near_bg(rgb):
            continue
        _, base = detect_bars_baseline(img, rgb, tol=50)
        if base is not None:
            baseline_px = base
            break
    calibration, _ = _calibrate_axis(img, img_path, tick_color, baseline_px, plot_bounds)
    if calibration is None:
        # 标定失败 → 逐色法整条走不通。**不要在这里直接 return []**：那会让兜底永远
        # 走不到（子图1 之前就是这么被"意外"挡在外面的）。交给兜底，由它按开关+守卫决定。
        return _maybe_geometric_fallback(
            img, img_path, tick_color, plot_bounds, len(rgbs), "逐色法标定失败")
    labels = _vision_read_x_labels(img_path)
    series = []
    seen: set[tuple] = set()
    for hex_color in colors:
        rgb = _hex_to_rgb(hex_color)
        if rgb is None or _near_bg(rgb):  # 非法颜色或接近背景，跳过
            continue
        values, confidence = _extract_one_series(img, calibration, rgb, labels)
        # **同色 + 值/标签逐一相同** → 这不是两个系列，是同一个检测被跑了两遍
        # （vision 把两个色号报成了一样）。丢掉重复那份：漏一个系列下游看得出，
        # 拿重复充一个假系列下游**看不出来**——喂进数据集就是污染。
        # 判据必须带「值也相同」：颜色撞车但值不同时，那是两个真系列，都得留。
        key = (hex_color.lower(), tuple((lb, round(v, 6)) for lb, v in values))
        if key in seen:
            logger.warning(
                "图 %s：色号 %s 检出重复系列（值与标签完全相同），已丢弃一份；"
                "这通常意味着 vision 没数对系列，实际系列数可能偏少",
                img_path, hex_color,
            )
            continue
        seen.add(key)
        series.append({"color": hex_color, "values": values, "confidence": confidence})

    # **结果驱动的降级**：逐色法一根都没捞到 → 交给兜底（开关 + 守卫在里面判定）。
    if not any(s["confidence"] != "low" for s in series):
        return _maybe_geometric_fallback(
            img, img_path, tick_color, plot_bounds, len(rgbs), "逐色法未检出可信柱子")
    return series


def _detect_tick_color(img: Image.Image) -> tuple[int, int, int]:
    """检测刻度/轴线颜色，选**轴框最宽**的那个候选色。

    候选 = 固定灰度（业界绝大多数图的轴都在里面）+ 图里最常见的几个非背景墨色
    （救彩色/浅色轴）。

    判据是**轴框宽度**，不是刻度条数——实测（论文2 图4）：墨色 (48,48,48) 反而
    多中一条刻度（5 vs 4），但它的掩码盖不全 x 轴、线断成两截，轴框宽度腰斩
    （457 vs 849）。按条数选会选中它 → 柱子检测归零。轴框宽度直接反映「这个
    掩码把整条轴盖住了没有」，才是对的判据。
    """
    from .chart_extract import _estimate_bg, detect_axis_ticks, locate_plot_bounds

    gray_cands = [
        (0, 0, 0), (35, 35, 35), (60, 60, 60), (80, 80, 80),
        (100, 100, 100), (150, 150, 150),
    ]
    arr = np.array(img.convert("RGB")).astype(int)
    bg = np.array(_estimate_bg(arr))
    ink = arr[np.abs(arr - bg).max(axis=2) > 40]  # 明显不是背景的像素
    ink_cands: list[tuple[int, int, int]] = []
    if len(ink):
        q = (ink // 32) * 32
        key = q[:, 0] * 65536 + q[:, 1] * 256 + q[:, 2]
        vals, counts = np.unique(key, return_counts=True)
        for v in vals[np.argsort(-counts)[:6]]:
            v = int(v)
            ink_cands.append(((v >> 16 & 255) + 16, (v >> 8 & 255) + 16, (v & 255) + 16))

    best_color, best_w = None, 0
    for c in gray_cands + ink_cands:
        ticks = detect_axis_ticks(
            img, tick_color=c, tol=20, min_run_len=3, max_run_len=40
        )
        if not 2 <= len(ticks) <= 20:
            continue
        bounds = locate_plot_bounds(img, tick_color=c)
        if bounds is None or bounds[2] <= bounds[0]:
            continue
        w = bounds[2] - bounds[0]
        if w > best_w:
            best_color, best_w = c, w
    return best_color or (35, 35, 35)


def _hex_to_rgb(hex_color: str) -> tuple[int, int, int] | None:
    """#RRGGBB → (r, g, b)。非法输入返回 None（调用方跳过）。"""
    m = re.fullmatch(r"#?([0-9a-fA-F]{6})", (hex_color or "").strip())
    if not m:
        return None
    h = m.group(1)
    return tuple(int(h[i : i + 2], 16) for i in (0, 2, 4))


def _format_bar_text_multi(figure_label: str, y_label: str, series: list) -> str:
    """多系列柱状图格式化。series = [{"color", "values"}]（均为非 low 系列）。"""
    lines = []
    head = f"{figure_label} {y_label}".strip() if y_label else figure_label
    if head:
        lines.append(head)
    for si, s in enumerate(series):
        if si > 0:
            lines.append("")
        lines.append(f"系列{si + 1}（{s['color']}）:")
        for i, (label, v) in enumerate(s["values"]):
            tag = label if label else f"柱{i + 1}"
            lines.append(f"- {tag}: {round(v, 2)}")
    return "\n".join(lines) if lines else ""
