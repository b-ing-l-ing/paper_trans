"""命令行入口 —— 一键：论文 PDF → 数据集 CSV。

用法：
    python -m paper2data 论文.pdf
    python -m paper2data 论文.pdf -s schema.json -o out.csv
"""

import argparse
import json
import logging
from pathlib import Path

from .parser import parse_pdf
from .table_detector import detect_tables
from .figure_extract import figure_to_text
from .schema import Schema
from .extract import extract_dataset
from .compute import normalize_record, compute_computed_fields
from .export import to_dataframe, save_csv
from .config import settings

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def extract_pdf(
    pdf_path: str,
    schema_path: str,
    output_path: str | None = None,
    composition_path: str | None = None,
) -> str:
    """单篇论文 → CSV。返回 CSV 路径。"""
    pdf = Path(pdf_path)
    schema = Schema.load(schema_path)
    composition = None
    if composition_path:
        composition = json.loads(Path(composition_path).read_text(encoding="utf-8"))

    print(f"解析 PDF: {pdf.name}")
    text, pages_text, pages_images = parse_pdf(pdf, image_dir=settings.image_dir)

    # 接入坐标聚类表格（无边框表），与 PyMuPDF 的表格互补
    detected = detect_tables(str(pdf))
    if detected:
        text += "\n\n" + _format_detected_tables(detected)
        print(f"  坐标聚类检测到 {len(detected)} 个无边框表格")

    # 图数据转文本（多模态），附加到全文，让 LLM 统一抽取
    figure_texts = []
    for pn in sorted(pages_images):
        for fname in pages_images[pn]:
            img_path = Path(settings.image_dir) / fname
            try:
                ft = figure_to_text(str(img_path), figure_label=f"图(第{pn}页)")
                if ft:
                    figure_texts.append(ft)
            except Exception as e:
                logger.warning("图 %s 提取失败: %s", fname, e)
    if figure_texts:
        text += "\n\n## 图表数据(从图提取)\n" + "\n\n".join(figure_texts)
        print(f"  从 {len(figure_texts)} 张图提取到图表数据")

    print("LLM 抽取中...")
    result = extract_dataset(text, schema)
    print(f"  抽到 {len(result.records)} 条试验, {len(result.new_fields)} 个新字段")
    if result.new_fields:
        for nf in result.new_fields:
            print(f"    · 新字段 {nf.name} ({nf.label})")

    normalized = [normalize_record(r, schema, result.new_fields) for r in result.records]
    if composition:
        normalized = [compute_computed_fields(r, schema, composition) for r in normalized]
        print("  已计算 computed 字段（成分库）")

    # 规则校验（硬约束：比例和、值域、摩尔比）
    from .verify import check_records

    issues = check_records(normalized, schema)
    if issues:
        print(f"  ⚠ 规则校验发现 {len(issues)} 个问题：")
        for issue in issues:
            print(f"    行{issue['record_index']} [{issue['rule']}] {issue['message']}")

    df = to_dataframe(normalized, schema, result.new_fields)

    output = Path(output_path) if output_path else pdf.with_suffix(".csv")
    csv_path = save_csv(df, output)
    print(f"  已导出 CSV: {csv_path} ({df.shape[0]} 行 x {df.shape[1]} 列)")
    return csv_path


def _format_detected_tables(tables: list[dict]) -> str:
    """把坐标聚类检测到的表格转成 markdown 文本，喂给 LLM。"""
    parts = ["## 坐标聚类表格（无边框表，精确结构）"]
    for i, t in enumerate(tables, 1):
        header = t.get("header", [])
        if not header:
            continue
        parts.append(f"\n### 表格 {i}（第 {t.get('page', '?')} 页）")
        parts.append("| " + " | ".join(str(h) for h in header) + " |")
        parts.append("|" + "|".join("---" for _ in header) + "|")
        for row in t.get("rows", []):
            cells = ["" if v is None else str(v) for v in row]
            while len(cells) < len(header):
                cells.append("")
            parts.append("| " + " | ".join(cells[: len(header)]) + " |")
    return "\n".join(parts)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="paper2data", description="论文 PDF → 数据集 CSV"
    )
    parser.add_argument("pdf", help="论文 PDF 路径")
    parser.add_argument(
        "-s", "--schema", default=str(Path(__file__).parent / "schema.json"),
        help="字段库 JSON 路径",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="输出 CSV 路径（默认与 PDF 同名）"
    )
    parser.add_argument(
        "-c", "--composition", default=None,
        help="原料成分库 JSON 路径（可选，用于计算 computed 字段）",
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        parser.error(f"文件不存在: {args.pdf}")

    try:
        extract_pdf(args.pdf, args.schema, args.output, args.composition)
    except Exception as e:
        logger.error("处理失败：%s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
