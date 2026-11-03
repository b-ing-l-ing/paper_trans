"""从 xlsx 真值表拆出每篇论文的 benchmark CSV。

真值来源：`data/多元固废数据搜集.xlsx` Sheet1，按最后一列 `参考文献` 分组。
集划分：`data/dev_papers.json` / `data/test_papers.json`。
输出：`data/benchmarks/<id>.csv`，列 = schema 59 字段 + `_xlsx_row`。

清洗规则（**只做无损转换**）：
  · NaN / "—" / "–" / "-" / 空白  → 空
  · Excel 把胶砂比 "3:1" 存成 datetime.time(3, 1) → 3.0
  · 字符串 "3:1" → 3.0
  · 数字字符串 → float
  · **0 原样保留** —— 不猜「0 是不是没填」，那是评测口径的事，不是真值的事

跑：.venv/Scripts/python.exe scripts/build_benchmarks.py
"""

from __future__ import annotations

import datetime
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from paper2data.schema import Schema

XLSX = Path("data/多元固废数据搜集.xlsx")
OUT_DIR = Path("data/benchmarks")
SETS = [("data/dev_papers.json", "开发集"), ("data/test_papers.json", "测试集")]

_EMPTY = {"", "-", "—", "–", "—", "–"}
_RATIO_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)\s*$")

# 已核实的真值修正 —— **必须在论文原文找到证据**，并同步写进
# data/benchmarks/CORRECTIONS.md。格式：{论文 id: {错列: 对应列}}
#   dev02：xlsx 把原文 Table 2 的 PG（磷石膏）录成了 DG（脱硫石膏）；
#          原文表头 `PG (wt%) GGBS (wt%) OPC (wt%) CS (wt%) w/b`，正文
#          "the PG content was fixed at 40 wt%"。错录让行匹配从 17/19 掉到 3/19。
CORRECTIONS: dict[str, dict[str, str]] = {
    "dev02": {"DG": "PG"},
}


def clean(v, field: str, stats: Counter):
    """单个单元格 → 数值 / 空 / 原字符串（原字符串会记进 stats）。"""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, datetime.time):
        # "3:1" 被 Excel 存成 3 时 1 分
        stats[f"{field}:time→比值"] += 1
        return v.hour / v.minute if v.minute else None
    if isinstance(v, str):
        s = v.strip()
        if s in _EMPTY:
            return None
        m = _RATIO_RE.match(s)
        if m:
            num, den = float(m.group(1)), float(m.group(2))
            stats[f"{field}:字符串比值"] += 1
            return num / den if den else None
        try:
            return float(s)
        except ValueError:
            stats[f"{field}:残留非数值"] += 1
            return s
    if isinstance(v, (int, float)):
        return float(v)
    stats[f"{field}:未知类型"] += 1
    return str(v)


def main() -> None:
    schema = Schema.load("paper2data/schema.json")
    cols = schema.field_names()
    df = pd.read_excel(XLSX, sheet_name=0)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for set_path, set_name in SETS:
        spec = json.loads(Path(set_path).read_text(encoding="utf-8"))
        print(f"=== {set_name}（{len(spec['papers'])} 篇）")
        for p in spec["papers"]:
            mask = df["参考文献"].astype(str) == p["ref"]
            sub = df[mask]
            assert len(sub) == p["rows"], f"{p['id']} 行数不符：{len(sub)} vs {p['rows']}"

            # 应用已核实的真值修正（见 CORRECTIONS）
            corr = CORRECTIONS.get(p["id"])
            if corr:
                sub = sub.copy()
                for wrong, right in corr.items():
                    sub[right] = sub[wrong]
                    sub[wrong] = 0.0

            stats: Counter = Counter()
            rows = []
            for idx, (_, r) in zip(sub.index, sub.iterrows()):
                row = {c: clean(r[c], c, stats) for c in cols}
                row["_xlsx_row"] = int(idx) + 2   # Excel 行号（表头占第 1 行）
                rows.append(row)

            out = OUT_DIR / f"{p['id']}.csv"
            pd.DataFrame(rows, columns=cols + ["_xlsx_row"]).to_csv(
                out, index=False, encoding="utf-8-sig"
            )
            note = " ".join(f"{k}×{v}" for k, v in stats.items()) or "干净"
            print(f"  {p['id']}  {len(rows):2d} 行  → {out.name}   {note}")

    print(f"\n输出目录：{OUT_DIR}")


if __name__ == "__main__":
    main()
