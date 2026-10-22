"""合成柱状图生成器 —— 把「难度」变成可调旋钮，真值精确（我们画的数就是真值）。

为什么需要它：真实图里「难度」是混在一起的（分辨率、系列数、填充、间隙、刻度密度
同时变），没法归因。合成图可以**一次只动一个旋钮**。

六张图，按已知的「代码死区」设计：

| 图 | 难度 | 特征 | 系列×组 | 压力点（我们已知的） |
|---|---|---|---|---|
| L1_single | 低   | 单系列、间隙大、刻度稀疏 | 1×5  | 基线（几何法该稳） |
| L2_two    | 低中 | 2 系列、有间隙、带误差棒 | 2×6  | 基线（逐色法该稳） |
| L3_tight  | 中   | 组内**完全无间隙** | 3×6  | 几何法死区（粘连） |
| L4_hatch  | 中高 | **斜纹填充** | 3×6  | 逐色法死区（纹理） |
| L5_hollow | 中高 | **空心柱**（只有描边） | 3×5  | 颜色法死区（柱身=背景色） |
| L6_dense  | 高   | 32 根、小图、密集刻度 | 4×8  | 综合（密集+小尺寸） |

图里**不印柱顶数值**（等价于 real 图集里的 `erased/`），逼两边都走标定/估读。

用法：python 对比实验/synthetic/make_charts.py
产出：对比实验/synthetic/<fig>.png
      对比实验/synthetic/benchmark.csv      （人看）
      对比实验/benchmark_synthetic.json     （喂 code_read/llm_read/score）
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parent
OUT_JSON = BASE.parent / "benchmark_synthetic.json"
RNG = np.random.default_rng(20261016)          # 固定种子 → 可复现


def vals(a, b, n, dec=2):
    """n 个落在 [a,b) 的「印刷感」数值。"""
    return [round(float(v), dec) for v in RNG.uniform(a, b, n)]


# ---------------------------------------------------------------------------
# 每张图的配方。values 是 [系列][组]，长度必须与 groups/series 对齐。
# ---------------------------------------------------------------------------
CHARTS = [
    dict(
        name="syn_L1_single", level="低",
        why="单系列、间隙大、刻度稀疏 —— 基线。柱子灰色 → 路由应走几何法",
        groups=["A", "B", "C", "D", "E"], series=["28d"],
        values=[vals(6, 24, 5)], colors=["#808080"],
        yticks=[0, 5, 10, 15, 20, 25], bar_w=0.55,
        figsize=(5.0, 3.6), dpi=150,
    ),
    dict(
        name="syn_L2_two", level="低中",
        why="2 系列、组间有间隙、带误差棒 —— 基线。两种彩色 → 路由该走逐色法",
        groups=["M0", "M15", "M30", "M45", "M60", "M75"], series=["7d", "28d"],
        values=[vals(20, 42, 6), vals(24, 46, 6)], colors=["#4C72B0", "#DD8452"],
        yticks=[0, 10, 20, 30, 40, 50], bar_w=0.38,
        err=True, figsize=(5.4, 3.6), dpi=150,
    ),
    dict(
        name="syn_L3_tight", level="中",
        why="组内**完全无间隙**（柱宽=1）—— 几何法死区：柱身连成一整块，只能靠柱顶跳变拆",
        groups=["C1", "C2", "C3", "C4", "C5", "C6"], series=["3d", "7d", "28d"],
        values=[vals(5, 13, 6), vals(8, 18, 6), vals(12, 26, 6)],
        colors=["#55A868", "#C44E52", "#8172B2"],
        yticks=[0, 5, 10, 15, 20, 25], bar_w=1 / 3,
        figsize=(5.4, 3.6), dpi=150,
    ),
    dict(
        name="syn_L4_hatch", level="中高",
        why="**斜纹填充** —— 逐色法死区：颜色掩码只剩线条、fill 过不了门槛",
        groups=["G1", "G2", "G3", "G4", "G5", "G6"], series=["3d", "7d", "28d"],
        values=[vals(4, 12, 6), vals(7, 17, 6), vals(11, 25, 6)],
        colors=["#F0A868", "#8CBF8C", "#9C8CC4"],
        hatch=["///", "...", "xxx"],
        yticks=[0, 5, 10, 15, 20, 25], bar_w=0.32,
        err=True, figsize=(5.4, 3.6), dpi=150,
    ),
    dict(
        name="syn_L5_hollow", level="中高",
        why="**空心柱**（只有描边）—— 颜色法死区：柱身=背景色，颜色掩码只有一圈边",
        groups=["H1", "H2", "H3", "H4", "H5"], series=["7d", "14d", "28d"],
        values=[vals(6, 15, 5), vals(9, 20, 5), vals(13, 28, 5)],
        colors=["#4C72B0", "#DD8452", "#55A868"], hollow=True,
        yticks=[0, 5, 10, 15, 20, 25, 30], bar_w=0.3,
        figsize=(5.4, 3.6), dpi=150,
    ),
    dict(
        name="syn_L6_dense", level="高",
        why="32 根 + 小尺寸 + 6 档刻度 —— 综合压力（对照 heldout 的 mat_fig2 那种塌方）",
        groups=["P0", "P10", "P20", "P30", "P40", "P50", "P60", "P70"],
        series=["7d", "14d", "28d", "56d"],
        values=[vals(15, 40, 8), vals(18, 44, 8), vals(21, 48, 8), vals(24, 52, 8)],
        colors=["#4C72B0", "#DD8452", "#55A868", "#C44E52"],
        yticks=[0, 10, 20, 30, 40, 50], bar_w=0.24,
        figsize=(4.6, 3.2), dpi=110,
    ),
]


def draw(spec: dict) -> None:
    fig, ax = plt.subplots(figsize=spec["figsize"], dpi=spec["dpi"])
    groups, series, values = spec["groups"], spec["series"], spec["values"]
    n_s, n_g = len(series), len(groups)
    x = np.arange(n_g)
    bw = spec["bar_w"]
    total = bw * n_s
    for si in range(n_s):
        off = -total / 2 + bw * si + bw / 2
        kw = dict(width=bw, label=series[si])
        if spec.get("hollow"):
            kw.update(facecolor="none", edgecolor=spec["colors"][si], linewidth=2.2)
        else:
            kw.update(color=spec["colors"][si])
        if spec.get("hatch"):
            kw.update(facecolor=spec["colors"][si], hatch=spec["hatch"][si],
                      edgecolor="black", linewidth=0.6)
        if spec.get("err"):
            kw.update(yerr=np.array(spec["values"][si]) * 0.06, capsize=2,
                      error_kw=dict(lw=1.0, ecolor="#333333"))
        ax.bar(x + off, values[si], **kw)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=8)
    ax.set_yticks(spec["yticks"])
    ax.set_yticklabels([str(t) for t in spec["yticks"]], fontsize=8)
    ax.set_ylabel("Value / MPa", fontsize=9)
    ax.legend(fontsize=7, frameon=False, ncol=len(series), loc="upper center",
              bbox_to_anchor=(0.5, 1.16))
    ax.tick_params(length=3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_linewidth(1.2)
    ax.spines["bottom"].set_linewidth(1.2)
    fig.tight_layout()
    fig.savefig(BASE / f"{spec['name']}.png")
    plt.close(fig)


def main() -> None:
    panels = {}
    rows = []
    for spec in CHARTS:
        draw(spec)
        labels, flat = [], []
        for gi, g in enumerate(spec["groups"]):
            for si, s in enumerate(spec["series"]):
                labels.append(f"{g}|{s}")
                flat.append(spec["values"][si][gi])
        key = f"{spec['name']}|p"
        panels[key] = {"name": key, "labels": labels, "values": flat}
        for lb, v in zip(labels, flat):
            g, s = lb.split("|")
            rows.append((spec["name"], "p", spec["name"], g, s, v, spec["why"]))
        print(f"  {spec['name']:<16}{spec['level']:<4}{len(flat):>3} 根柱  "
              f"{len(spec['series'])} 系列 × {len(spec['groups'])} 组")

    # 人看的 CSV
    with (BASE / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as f:
        import csv
        w = csv.writer(f)
        w.writerow(["figure", "panel", "panel_name", "group", "series", "value", "note"])
        w.writerows(rows)

    # 喂流水线的 JSON
    out = {"说明": "合成图真值。输入 = synthetic/<fig>.png（不印柱顶数值）。"
                   "难度梯度见 make_charts.py 的表。"}
    for spec in CHARTS:
        key = f"{spec['name']}|p"
        out[spec["name"]] = {
            "image": f"synthetic/{spec['name']}.png",
            "source": f"合成图 {spec['level']}难度 —— {spec['why']}",
            "panels": [panels[key]],
        }
    import json
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n写出 {OUT_JSON} 与 synthetic/benchmark.csv")


if __name__ == "__main__":
    main()
