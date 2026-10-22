"""把 blindset/benchmark.csv 转成和开发集同构的 benchmark_json（供 score.py 统一使用）。

映射：CSV 的 `group` → panels[].name，`series` → panels[].labels，`value` → values。
面板名带上子图字母（如 "a-A1"），避免跨子图重名，也保住溯源信息。

测试输入默认用 `erased/`（无印值图）——那才是逼双方走「标定+反算」的那一版；
`original/`（带印值）另存在 `image_with_labels` 字段里，供「读印值」那个条件用。

用法：python 对比实验/blindset_to_benchmark.py
产出：对比实验/benchmark_blindset.json
"""
import csv
import json
from pathlib import Path

BASE = Path(__file__).resolve().parent
SRC = BASE / "blindset" / "benchmark.csv"
OUT = BASE / "benchmark_blindset.json"

# 子图文件名 → 人类可读的说明（写进 source，供读结果的人对照）
FROM = {
    "24_fig3": "24.pdf 图3 —— 4 面板分组柱（抗压强度），柱顶印值",
    "yansuan_fig4": "碱与硫酸盐篇 图4 —— 3 面板分组柱（抗压强度），柱顶印值",
}


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8-sig")))
    figs: dict[str, dict] = {}
    notes: dict[str, list[str]] = {}

    for r in rows:
        fig = (r["figure"] or "").strip()
        val = (r["value"] or "").strip()
        if not fig or not val:
            continue  # 空值 = 待填，不进真值
        v = float(val)
        key = f"{r['panel'].strip()}-{r['group'].strip()}"  # 如 "a-A1"
        entry = figs.setdefault(fig, {"source": FROM.get(fig, fig), "panels": {}})
        p = entry["panels"].setdefault(key, {"name": key, "labels": [], "values": []})
        p["labels"].append(r["series"].strip())
        p["values"].append(v)
        note = (r.get("note") or "").strip()
        if note and "已验证" in note:
            notes.setdefault(fig, []).append(f"{key}-{r['series'].strip()}")

    # 按 CSV 出现顺序固定 panel 顺序（dict 保序），并挂上图片路径
    out: dict = {"说明": "盲测集真值。测试输入 = erased/（无印值图）；original/ 是带印值原图。"}
    for fig, e in figs.items():
        out[fig] = {
            "image": f"blindset/erased/{fig}.png",
            "image_with_labels": f"blindset/original/{fig}.png",
            "source": e["source"],
            "panels": list(e["panels"].values()),
        }
        if fig in notes:
            out[fig]["note"] = "正文已验证：" + "、".join(notes[fig])

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    n = sum(len(p["values"]) for f in out.values() if isinstance(f, dict) for p in f.get("panels", []))
    print(f"写出 {OUT}")
    for fig, e in out.items():
        if isinstance(e, dict):
            print(f"  {fig}: {len(e['panels'])} 面板 / {sum(len(p['values']) for p in e['panels'])} 个真值")


if __name__ == "__main__":
    main()
