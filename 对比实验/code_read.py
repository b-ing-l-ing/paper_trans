"""代码读图方：用 paper2data 的 figure_to_text（代码切框 + 标定反算）读图。

运行：python 对比实验/code_read.py                  # 开发集（默认）
      python 对比实验/code_read.py --set blindset   # 盲测集（输入用 erased/ 无印值图）
输出：<set>/code_result.json   原文（喂 score.py）
      <set>/code_result.csv    人可读明细：图 / 面板 / 标签 / 值 / 来源
"""
import argparse
import csv
import json
import re
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))          # 对比实验/ 自身（vision_cache / exp_sets / score）
sys.path.insert(0, str(BASE.parent))   # 项目根目录，用于 import paper2data

import exp_sets
from vision_cache import install
from paper2data.figure_extract import figure_to_text
from score import extract_code_items

# CSV 的「面板」列只取标题里的**短标记**（子图N / 图N）——标题常是一整句 vision 描述，
# 整句塞进列里撑爆表格，而且代码切的块标题是「子图1 SEM显微照片…」，取整句反而没区分度
_MARK_RE = re.compile(r"子图\s*\d+|图\s*\d+\s*[（(]?\s*[a-zA-Z]?\s*[)）]?")


def _panel_short(heading: str) -> str:
    m = _MARK_RE.search(heading or "")
    return m.group(0).strip() if m else (heading or "")[:16]


def main() -> None:
    ap = argparse.ArgumentParser(description="代码读图方")
    exp_sets.add_set_arg(ap)
    args = ap.parse_args()
    set_name = args.set_name

    n = install()  # 把 vision 调用换成缓存版：同图复跑结果一致（改算法逻辑仍会变）
    print(f"[{set_name}] vision 缓存：命中 {n} 条（删 对比实验/.vision_cache/ 可重新问 vision）")

    bench = json.loads(exp_sets.bench_path(set_name).read_text(encoding="utf-8"))
    result: dict[str, str] = {}
    for fig_name, fig in bench.items():
        if fig_name == "说明":
            continue
        img_path = str(BASE / fig["image"])
        print(f"代码读图：{fig_name} ...")
        try:
            text = figure_to_text(img_path, "")
        except Exception as e:
            text = f"[异常] {e}"
        result[fig_name] = text

    out_json = exp_sets.out_path("code_result", set_name)
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    # 人可读 CSV：把 figure_to_text 的文本解析成一行一个值
    rows = []
    for fig_name, text in result.items():
        for it in extract_code_items(text):
            rows.append({
                "图": fig_name,
                "面板": _panel_short(it.get("heading") or it["panel"]),
                "标签": it["label"],
                "值": it["value"],
                "来源": "约(落回)" if it["approx"] else "反算",
            })
    out_csv = exp_sets.out_path("code_result", set_name, "csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=["图", "面板", "标签", "值", "来源"])
        w.writeheader()
        w.writerows(rows)

    print(f"\n完成：{out_json.name}（原文）+ {out_csv.name}（{len(rows)} 行明细）")


if __name__ == "__main__":
    main()
