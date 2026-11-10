"""检查 benchmark 里的真值**能否在 PDF 中找到**。

为什么需要：有些真值来自补充材料（Supplementary）或学生估读，正文 PDF 里根本不存在。
这种情况下工具永远抽不到，却会被记成「漏抽」——用这种真值算准确率是在冤枉工具。

做法：把 benchmark 的每行当成 records，喂给现有的零 LLM 溯源模块，
在「正文 + 图文本」里找每个值。找不到的列出来。

用法：
    .venv/Scripts/python.exe scripts/check_benchmark.py data/benchmarks/dev02.csv \
        "data/开发集/dev02_xxx.pdf" [--figures output/dev02_debug.figures.txt]

输出：
    · 按字段的可获取率
    · 不可获取值的清单（行号 / 字段 / 值）
    · 可选：写 <benchmark>.unavailable.csv，供评测跳过
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from paper2data.parser import parse_pdf          # noqa: E402
from paper2data.schema import Schema             # noqa: E402
from paper2data.verify import build_provenance   # noqa: E402


def _load_figures(path: str | None) -> list[dict]:
    """读 --debug 落盘的图文本（带页码）。没有就返回空。"""
    if not path or not Path(path).exists():
        return []
    text = Path(path).read_text(encoding="utf-8")
    out, cur_page, cur = [], None, []
    for line in text.splitlines():
        if line.startswith("### 第") and "页" in line:
            if cur_page is not None:
                out.append({"page": cur_page, "text": "\n".join(cur)})
            try:
                cur_page = int(line.split("第")[1].split("页")[0])
            except ValueError:
                cur_page = None
            cur = []
        else:
            cur.append(line)
    if cur_page is not None:
        out.append({"page": cur_page, "text": "\n".join(cur)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="检查 benchmark 真值能否在 PDF 中找到")
    ap.add_argument("benchmark", help="benchmark CSV")
    ap.add_argument("pdf", help="对应论文 PDF")
    ap.add_argument("--figures", default=None,
                    help="图文本（--debug 生成的 .figures.txt）")
    ap.add_argument("--out", default=None, help="写出不可获取清单 CSV")
    args = ap.parse_args()

    schema = Schema.load("paper2data/schema.json")
    bench = pd.read_csv(args.benchmark)
    fields = [f for f in schema.field_names() if f in bench.columns]

    # 只检查非空、非 0 的单元格（0 是「不掺」，不要求出现在原文）
    records = []
    index_map = []
    for i, row in bench.iterrows():
        rec = {}
        for f in fields:
            v = row[f]
            if pd.notna(v) and float(v) != 0:
                rec[f] = str(v)
        records.append(rec)
        index_map.append(i + 1)

    _, pages_text, _, _ = parse_pdf(args.pdf, image_dir="./output/images")
    figures = _load_figures(args.figures)
    print(f"正文 {len(pages_text)} 页 / 图文本 {len(figures)} 块")

    report = build_provenance(records, schema, pages_text, figure_texts=figures)

    # 按字段统计
    stats: dict[str, dict[str, int]] = {}
    missing = []
    for e in report.entries:
        st = stats.setdefault(e.field, {"total": 0, "found": 0})
        st["total"] += 1
        if e.found:
            st["found"] += 1
        else:
            missing.append({
                "row": index_map[e.record_index],
                "field": e.field,
                "value": e.value,
                "status": e.status,
            })

    print("\n按字段的可获取率（真值非空且非 0 的单元格）")
    print(f"  {'字段':<12}{'总数':>6}{'可获取':>8}{'比例':>8}")
    for f, st in sorted(stats.items(), key=lambda x: x[1]["found"] / x[1]["total"]):
        rate = st["found"] / st["total"] if st["total"] else 0
        print(f"  {f:<14}{st['total']:>6}{st['found']:>8}{rate:>8.0%}")

    total = sum(s["total"] for s in stats.values())
    found = sum(s["found"] for s in stats.values())
    print(f"\n总计：可获取 {found}/{total}（{found/total:.1%}）")

    if missing:
        print(f"\n不可获取的值（前 30 条，共 {len(missing)}）")
        for m in missing[:30]:
            print(f"  行{m['row']:>3}  {m['field']:<10} {m['value']:<12} [{m['status']}]")

    if args.out:
        pd.DataFrame(missing).to_csv(args.out, index=False, encoding="utf-8-sig")
        print(f"\n已写 {args.out}")


if __name__ == "__main__":
    main()
