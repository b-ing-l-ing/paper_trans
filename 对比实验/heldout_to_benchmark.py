"""把 heldout/benchmark.csv 转成和开发集同构的 benchmark_json（供 score.py 统一使用）。

映射与 `blindset_to_benchmark.py` 完全一致：CSV `group` → panels[].name、
`series` → panels[].labels、`value` → values；面板名带子图字母（`a-5`、`a-M0`）。

**这个脚本只搬数据，不做任何口径决策。** 但 CSV 里的 `note` 会被原样收进
`note` 字段——因为 heldout 有两条**必须先讲清**的：

- `rm_fig3` 的「零高度柱」（a/组5/3d = 0）：原图只印了标签「0」、**没有柱体**。
  这根柱**没有东西可测** → 计不计入分母、算不算漏读，是**评测口径**问题，不是代码问题。
- `mat_fig2` 的真值**低置信**（位图、柱顶印值小），CSV 自标「读数不清 / 待核」→
  它的数字只能当参考，别据此下强结论。

用法：python 对比实验/heldout_to_benchmark.py
产出：对比实验/benchmark_heldout.json
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "heldout" / "benchmark.csv"
OUT = BASE / "benchmark_heldout.json"

FROM = {
    "rm_fig3": "赤泥基全固废矿井充填材料 图3 —— 1 图 2 面板（AR/BR），6 组 × 3 龄期，"
               "柱顶印值（真值经**矢量几何双路确认**，高置信）",
    "mat_fig2": "materials-19-01019 Figure 2 上排 (a)(b) —— 10 组 × 3 龄期，柱顶印值"
                "（位图，真值**低置信**，CSV 自标「读数不清/待核」）",
}


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    figs: dict[str, dict] = {}
    notes: dict[str, list[str]] = {}

    for r in rows:
        fig = (r["figure"] or "").strip()
        val = (r["value"] or "").strip()
        if not fig or not val:
            continue                      # 空值 = 待填，不进真值
        key = f"{r['panel'].strip()}-{r['group'].strip()}"
        entry = figs.setdefault(fig, {"panels": {}})
        p = entry["panels"].setdefault(key, {"name": key, "labels": [], "values": []})
        p["labels"].append(r["series"].strip())
        p["values"].append(float(val))
        note = (r.get("note") or "").strip()
        if note:
            notes.setdefault(fig, []).append(f"{key}-{r['series'].strip()}: {note}")

    out: dict = {
        "说明": ("留出集真值。测试输入 = heldout/erased/（擦掉柱顶印值那版）。"
                 "⚠ rm_fig3 有一根「零高度柱」(a-5/3d=0，原图只印标签无柱体)；"
                 "⚠ mat_fig2 真值低置信，仅供参照。"),
    }
    for fig, e in figs.items():
        out[fig] = {
            "image": f"heldout/erased/{fig}.png",
            "image_with_labels": f"heldout/original/{fig}.png",
            "source": FROM.get(fig, fig),
            "panels": list(e["panels"].values()),
        }
        if fig in notes:
            out[fig]["note"] = "；".join(notes[fig])

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"写出 {OUT}")
    for fig, e in out.items():
        if isinstance(e, dict) and "panels" in e:
            n = sum(len(p["values"]) for p in e["panels"])
            zeros = sum(1 for p in e["panels"] for v in p["values"] if v == 0)
            print(f"  {fig}: {len(e['panels'])} 面板 / {n} 个真值"
                  f"{f'（其中零值 {zeros} 个）' if zeros else ''}")


if __name__ == "__main__":
    main()
