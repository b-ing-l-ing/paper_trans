"""影子判断：**不改正式输出**，只为「逐色法失败 → 降级几何法」这条路积累证据。

每个块仍按原管线走（本脚本 hook `_single_figure_to_text`：先记录、再原样调用），
同时记下：
  · 这个块 vision 判成什么、路由去哪、逐色法拿到几根
  · 颜色掩码的碎片特征（块数 / 最大块 fill / 中位 fill）——**原始值，不设阈值**
  · **几何法如果被调用会得到什么**：柱数、标定是否成功、置信度、前几个值
  · 「候选降级」只标无争议的那一半：逐色 **0 根 且 有有效色**
    （「掩码碎片化」那一半先只记原始值——样本不够，不急着定边界）

为什么要留「几何法实际结果」：否则一堆漂亮的碎片特征只能证明它识别了某种纹理，
不能证明它适合当降级路由。

运行：python 对比实验/shadow_fallback.py
产出：对比实验/shadow_fallback.csv（每次覆盖，便于累积对照）
"""
import csv
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

import numpy as np
from PIL import Image

from vision_cache import install

install()

from paper2data import chart_extract as ce
from paper2data import figure_extract as fe

FIGURES = [
    ("论文1_图2", BASE / "images/论文1_图2.png"),
    ("论文1_图4", BASE / "images/论文1_图4.png"),
    ("论文2_图1-2", BASE / "images/论文2_图1-2.jpeg"),
    ("论文2_图3-4", BASE / "images/论文2_图3-4.jpeg"),
    ("24_fig3", BASE / "blindset/erased/24_fig3.png"),
    ("yansuan_fig4", BASE / "blindset/erased/yansuan_fig4.png"),
]

FIELDS = ["图", "块", "vision_type", "has_labels", "有效色数", "路由", "逐色柱数",
          "掩码块数合计", "最大块fill", "中位fill", "几何柱数", "几何标定",
          "几何置信度", "几何值示例", "候选降级", "备注"]

rows: list[dict] = []
_current = {"fig": ""}

# detect_bars 记忆化：同一 (图对象, 色, 参数) 不重复算——它是纯 Python BFS，很贵
_orig_detect_bars = ce.detect_bars
_memo: dict = {}


def detect_bars_memo(img, color, tol=20, min_area=200, min_side=0, with_fill_ratio=False):
    key = (id(img), tuple(color), tol, min_area, min_side, with_fill_ratio)
    if key not in _memo:
        _memo[key] = _orig_detect_bars(img, color, tol=tol, min_area=min_area,
                                       min_side=min_side, with_fill_ratio=with_fill_ratio)
    return _memo[key]


ce.detect_bars = detect_bars_memo


def inspect_block(img_path: str, label: str, plot_bounds) -> dict:
    """照原样量一遍这个块——只读取，不干预任何决策。"""
    crop = Image.open(img_path)
    rec = dict.fromkeys(FIELDS, "")
    rec.update({"图": _current["fig"], "块": label, "有效色数": 0, "逐色柱数": 0,
                "掩码块数合计": 0})
    try:
        anal = fe.analyze_figure(img_path)
    except Exception as e:  # noqa: BLE001
        rec["备注"] = f"analyze 失败:{e}"
        return rec
    rec["vision_type"] = anal.get("type", "")
    rec["has_labels"] = anal.get("has_labels", "")

    arr = np.array(crop.convert("RGB")).astype(int)
    bg = ce._estimate_bg(arr)
    rgbs = [(c, fe._hex_to_rgb(c)) for c in (anal.get("colors") or [])]
    rgbs = [(c, r) for c, r in rgbs
            if r is not None and max(abs(r[i] - bg[i]) for i in range(3)) > 20]
    rec["有效色数"] = len(rgbs)
    gray = len(rgbs) >= 2 and all(
        abs(r[0] - r[1]) < 20 and abs(r[1] - r[2]) < 20 for _, r in rgbs)
    rec["路由"] = "几何" if gray else "逐色"

    # ---- 逐色法：与生产同参数真实调用，顺带留掩码碎片特征 ----
    if not gray:
        fills, nblocks, nbars = [], 0, 0
        for _hexc, rgb in rgbs:
            blocks = detect_bars_memo(crop, rgb, tol=50, min_area=200,
                                      min_side=15, with_fill_ratio=True)
            bars, _ = ce.detect_bars_baseline(crop, rgb, tol=50)
            nblocks += len(blocks)
            nbars += len(bars)
            fills += [b[4] for b in blocks]
        rec["逐色柱数"] = nbars
        rec["掩码块数合计"] = nblocks
        if fills:
            fills.sort()
            rec["最大块fill"] = round(fills[-1], 3)
            rec["中位fill"] = round(fills[len(fills) // 2], 3)
        rec["候选降级"] = "是" if (nbars == 0 and len(rgbs) >= 1) else ""

    # ---- 几何法：如果被调用会得到什么（复刻 _extract_gray_series 的后半段）----
    try:
        tc = fe._detect_tick_color(crop)
        bounds = plot_bounds or ce.locate_plot_bounds(crop, tick_color=tc)
        if bounds is None:
            rec["几何柱数"] = "轴框None"
            return rec
        geo = ce.detect_gray_bars(crop, bounds)
        rec["几何柱数"] = len(geo)
        cal, _ = fe._calibrate_axis(crop, img_path, tc, float(bounds[3]))
        if cal is None:
            rec["几何标定"] = "失败"
        else:
            rec["几何标定"] = f"成功({len(cal.pixels)}锚点)"
            results = [((b[0] + b[1]) / 2, cal.value_at_pixel(float(b[2]))) for b in geo]
            if results:
                labels = fe._vision_read_x_labels(img_path)
                rec["几何置信度"] = fe._assess_confidence(results, labels)
                rec["几何值示例"] = str([round(v, 2) for _, v in results[:5]])
    except Exception as e:  # noqa: BLE001
        rec["备注"] = f"几何法异常:{e}"
    return rec


_orig_single = fe._single_figure_to_text


def shadow(img_path, figure_label="", plot_bounds=None):
    rows.append(inspect_block(img_path, figure_label or "单图", plot_bounds))
    return _orig_single(img_path, figure_label, plot_bounds)  # 正式输出原样返回


fe._single_figure_to_text = shadow

for fig_name, path in FIGURES:
    _current["fig"] = fig_name
    print(f"影子扫描：{fig_name} ...", flush=True)
    try:
        fe.figure_to_text(str(path), "")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ 整图异常：{e}")

out = BASE / "shadow_fallback.csv"
with open(out, "w", newline="", encoding="utf-8-sig") as f:
    w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)

print(f"\n{'=' * 118}")
print(f"{'图':<12}{'块':<9}{'vision':<7}{'色':>3}{'路由':>5}{'逐色':>5}{'掩码块':>7}"
      f"{'maxFill':>8}{'几何':>5}{'标定':>10}{'置信':>7}{'候选降级':>8}")
print("=" * 118)
for r in rows:
    print(f"{r['图']:<12}{r['块']:<9}{str(r['vision_type']):<7}{r['有效色数']:>3}"
          f"{r['路由']:>5}{r['逐色柱数']:>5}{r['掩码块数合计']:>7}{str(r['最大块fill']):>8}"
          f"{str(r['几何柱数']):>5}{str(r['几何标定'])[:8]:>10}"
          f"{str(r['几何置信度']):>7}{str(r['候选降级']):>8}")
print("=" * 118)
print(f"共 {len(rows)} 个块 → {out.name}")
