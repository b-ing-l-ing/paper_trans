"""LLM 读图方：纯 LLM 直接看图估读数值（公平 prompt，不给刻度/标签/子图提示）。

运行：python 对比实验/llm_read.py                  # 开发集（默认）
      python 对比实验/llm_read.py --set blindset   # 盲测集（输入用 erased/ 无印值图）
输出：<set>/llm_result.json   读数列表（喂 score.py）
      <set>/llm_result.csv    人可读明细：图 / 序号 / 值
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE.parent))

import exp_sets
from vision_cache import cached_vision_call
from paper2data.figure_extract import _mime

_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def parse_values(raw: str) -> list[float]:
    """容错解析 LLM 输出的数值列表。

    LLM 估读时常写「约148」「148（估）」，直接 float() 会整份失败 → 逐段抽数字，
    非数值 token 跳过（不因一个「约」丢掉整张图）。每个 token 取**最后一个**数
    （标签/说明在前、数值在后，如「组1: 148」「共12根柱子: 约4.2」）。"""
    raw = raw.replace("，", ",").replace("、", ",")
    vals = []
    for token in raw.split(","):
        nums = _NUM_RE.findall(token)
        if nums:
            vals.append(float(nums[-1]))
    return vals


def llm_read_chart(img_path: str) -> list[float]:
    """公平估读：不给纵轴范围、不给标签、不提示子图分割。"""
    prompt = (
        "这是柱状图（可能含多个子图）。请从左到右、从上到下读出所有柱子的数值，"
        "只输出数字列表（逗号分隔），不要任何解释。"
    )
    raw = cached_vision_call(Path(img_path).read_bytes(), _mime(img_path), prompt)
    return parse_values(raw)


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM 读图方")
    exp_sets.add_set_arg(ap)
    args = ap.parse_args()
    set_name = args.set_name

    bench = json.loads(exp_sets.bench_path(set_name).read_text(encoding="utf-8"))
    result: dict[str, list[float]] = {}
    for fig_name, fig in bench.items():
        if fig_name == "说明":
            continue
        img_path = str(BASE / fig["image"])
        print(f"LLM 读图：{fig_name} ...")
        values = llm_read_chart(img_path)
        result[fig_name] = values
        print(f"  读出 {len(values)} 个值：{values}")

    out_json = exp_sets.out_path("llm_result", set_name)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人可读 CSV：一行一个值（LLM 不报标签，只给从左到右的序号）
    rows = [
        {"图": fig_name, "序号": i + 1, "值": v}
        for fig_name, vals in result.items()
        for i, v in enumerate(vals)
    ]
    out_csv = exp_sets.out_path("llm_result", set_name, "csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["图", "序号", "值"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n完成：{out_json.name}（读数）+ {out_csv.name}（{len(rows)} 行明细）")


if __name__ == "__main__":
    main()
