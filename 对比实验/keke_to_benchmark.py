"""把 `keke/` 里那 4 张图合成评测集：`benchmark_keke.json`。

真值来源（只把 `image` 路径改指向 `keke/`）：
  · 论文1_图2 / 论文2_图1-2 / 论文2_图3-4   ← `benchmark.json`（开发集）
  · rm_fig3                                ← `benchmark_heldout.json`（留出集）

> 2026-10-16 用户要求：移除 `论文1_图4`（6 系列里有一对基色相同、靠斜纹区分 →
> 代码把两个系列并成一个，标签退化成「柱1…柱10」，不是读数问题但会拉低归属率）；
> 补入 `rm_fig3`。

**不用写新的 code_read / llm_read / score** —— 那三个脚本都是 `--set` 驱动的：
    python 对比实验/code_read.py --set keke
    python 对比实验/llm_read.py  --set keke
    python 对比实验/score.py     --set keke

⚠️ 口径：这个合集里 **3 张是开发集 + 1 张（rm_fig3）来自留出集** —— 混的，
**不是干净的留出集**，结论只能说「在这 4 张上」。另外 `rm_fig3` 自带一根**零高度柱**
（原图只印标签「0」、无柱体），score 会把它排除出分母、单列报数。

用法：python 对比实验/keke_to_benchmark.py
"""
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
OUT = BASE / "benchmark_keke.json"

ORDER = ["论文1_图2", "论文2_图1-2", "论文2_图3-4", "rm_fig3"]
HELDOUT_FROM = {"rm_fig3"}          # 这几张从 benchmark_heldout.json 取


def main() -> None:
    dev = json.loads((BASE / "benchmark.json").read_text(encoding="utf-8"))
    held = json.loads((BASE / "benchmark_heldout.json").read_text(encoding="utf-8"))

    out: dict = {
        "说明": "keke 合集（4 张 = 开发集 3 张 + 留出集 rm_fig3）。"
                "输入图 = keke/ 下的同名文件。⚠ 混了开发集，非干净留出集；"
                "rm_fig3 含 1 根零高度柱（已由 score 排除出分母）。",
    }
    missing = []
    for fig in ORDER:
        src = (held if fig in HELDOUT_FROM else dev).get(fig)
        if not src:
            print(f"  ⚠ {fig}: 对应真值里没有，跳过")
            continue
        entry = dict(src)
        entry["image"] = f"keke/{fig}.png" if fig == "rm_fig3" else f"keke/{Path(src['image']).name}"
        if not (BASE / entry["image"]).exists():
            missing.append(entry["image"])
        out[fig] = entry

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"写出 {OUT}")
    tot = 0
    for fig in ORDER:
        if fig not in out:
            continue
        n = sum(len(p["values"]) for p in out[fig]["panels"])
        zeros = sum(1 for p in out[fig]["panels"] for v in p["values"] if v == 0)
        tot += n
        print(f"  {fig:<14}{len(out[fig]['panels']):>2} 面板 / {n:>3} 个真值"
              f"{f'（零高度柱 {zeros}）' if zeros else '':<14}   ← {out[fig]['image']}")
    print(f"  合计 {tot} 个真值")
    if missing:
        print(f"  ⚠ 以下图片文件不存在：{missing}")


if __name__ == "__main__":
    main()
