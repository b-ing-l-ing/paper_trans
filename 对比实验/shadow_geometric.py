"""影子评估：24_fig3 若**切到几何法**，值会不会真的更准？（**不改正式输出**）

三步：
  ① 路由：逐块记 vision 报的 type/colors → 有效色 → 判据「≥2 色且全灰」→ 实际分支
  ② 影子：同一张裁剪图上，**逐色法**（现状走的路）与**几何法**（假想的路）各跑一遍，
          都走生产函数（`extract_unlabeled_bar_multi` / `_extract_gray_series`）
  ③ 评值：两者都拿 benchmark 做一对一匹配，报 配对 / ±5% / 精确 / 漏检 / 多检

⚠️ 纪律：**几何法能正确标定 ≠ 它提取的值更准**。标定只是其中一环，柱检测、柱顶、
   误差棒、基线的任何一环都能把值拉偏，所以必须逐块量到底，而不是看标定成功就下结论。

运行：python 对比实验/shadow_geometric.py
"""
import json
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
from score import align

IMAGE = BASE / "blindset/erased/24_fig3.png"
BENCH = BASE / "benchmark_blindset.json"
PANELS = ["a", "b", "c", "d"]


def gt_for(letter):
    t = json.loads(BENCH.read_text(encoding="utf-8"))["24_fig3"]
    return [(p["name"], lb, v) for p in t["panels"]
            if p["name"].startswith(letter + "-")
            for lb, v in zip(p["labels"], p["values"])]


def grade(values, letter, low_ok=True):
    """1-1 匹配；同时报 ±5% 和精确(≤0.5%)——只报配对数会被 DP 的 100% 阈值骗。"""
    gt = gt_for(letter)
    if not values:
        return {"读出": 0, "配对": 0, "真值": len(gt), "±5%": 0, "精确": 0,
                "漏检": len(gt), "多检": 0}
    pred = [{"label": "", "value": v} for v in values]
    pairs, extra = align(gt, pred)
    rels = [abs(pred[pi]["value"] - gt[gi][2]) / abs(gt[gi][2]) * 100 for gi, pi in pairs.items()]
    return {"读出": len(values), "配对": len(pairs), "真值": len(gt),
            "±5%": sum(1 for e in rels if e <= 5),
            "精确": sum(1 for e in rels if e <= 0.5),
            "漏检": len(gt) - len(pairs), "多检": len(extra)}


def main() -> None:
    img = Image.open(IMAGE)
    w, h = img.size
    tmpdir = Path(tempfile.mkdtemp())
    rows = []
    print("=" * 112)
    print(f"{'块':<6}{'路由':<6}{'有效色':>6}{'逐色法':>22}{'几何法':>22}   真值")
    print("=" * 112)
    for bi in range(4):
        fx0, fy0, fx1, fy1 = ce.detect_plot_frames(img)[bi]
        nx0, ny0, nx1, ny1 = fx0 / w, fy0 / h, fx1 / w, fy1 / h
        dx, dy = (nx1 - nx0) * 0.3, (ny1 - ny0) * 0.3
        cb = (int(max(0, nx0 - dx) * w), int(max(0, ny0 - dy) * h),
              int(min(1, nx1 + dx) * w), int(min(1, ny1 + dy) * h))
        crop = img.crop(cb)
        p = str(tmpdir / f"b{bi}.png")
        crop.save(p)
        bounds = fe._frame_to_plot_bounds((fx0, fy0, fx1, fy1), cb)
        tc = fe._detect_tick_color(crop)

        # ① 路由
        anal = fe.analyze_figure(p)
        colors = anal.get("colors") or []
        arr = np.array(crop.convert("RGB")).astype(int)
        bg = ce._estimate_bg(arr)
        rgbs = [r for r in (fe._hex_to_rgb(c) for c in colors)
                if r is not None and max(abs(r[i] - bg[i]) for i in range(3)) > 20]
        gray = len(rgbs) >= 2 and all(abs(r[0]-r[1]) < 20 and abs(r[1]-r[2]) < 20 for r in rgbs)

        # ② 影子：两条路都用生产函数真跑一遍
        pc_series = fe.extract_unlabeled_bar_multi(p, colors, plot_bounds=bounds)
        pc_vals = [v for s in pc_series if s["confidence"] != "low" for _, v in s["values"]]
        pc_conf = [s["confidence"] for s in pc_series]

        geo_series = fe._extract_gray_series(crop, p, tc, bounds)
        geo_vals = [v for s in geo_series if s["confidence"] != "low" for _, v in s["values"]]
        geo_conf = [s["confidence"] for s in geo_series]

        g_pc, g_geo = grade(pc_vals, PANELS[bi]), grade(geo_vals, PANELS[bi])
        rows.append({"块": f"子图{bi+1}", "轴框": list(bounds), "colors": colors,
                     "有效色": len(rgbs), "路由": "几何" if gray else "逐色",
                     "逐色法": {**g_pc, "置信度": pc_conf},
                     "几何法": {**g_geo, "置信度": geo_conf},
                     "geometry会走几何法吗": gray})

        def fmt(g):
            return f"配对{g['配对']:>2} ±5%{g['±5%']:>2} 精确{g['精确']:>2}"
        print(f"子图{bi+1:<3}{'几何' if gray else '逐色':<6}{len(rgbs):>6}"
              f"{fmt(g_pc):>22}{fmt(g_geo):>22}   {g_pc['真值']}")
        print(f"      逐色法置信度={pc_conf}  几何法置信度={geo_conf}"
              f"  几何法值={[round(v,2) for v in geo_vals]}")

    # 汇总
    tot = lambda key, which: sum(r[which][key] for r in rows)
    print("=" * 112)
    print(f"{'合计':<6}{'':<6}{'':>6}"
          f"{'配对%2d ±5%%%2d 精确%2d' % (tot('配对','逐色法'), tot('±5%','逐色法'), tot('精确','逐色法')):>22}"
          f"{'配对%2d ±5%%%2d 精确%2d' % (tot('配对','几何法'), tot('±5%','几何法'), tot('精确','几何法')):>22}"
          f"   {rows[0]['逐色法']['真值'] * 4}")
    print("=" * 112)
    print("⚠ 现状正式输出里 24_fig3 的**代码反算值是 0**（逐色法 0 根 → 全部落回 vision）。")
    print("⚠ 几何法这几列是**假想值**：正式输出没有变，路由也没改。")
    (BASE / "shadow_geometric.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"trace 已写 shadow_geometric.json")


if __name__ == "__main__":
    main()
