"""命令行入口 —— 一键：论文 PDF → 数据集 CSV。

用法：
    python -m paper2data 论文.pdf
    python -m paper2data 论文.pdf -s schema.json -o out.csv
"""

import argparse
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
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
    use_figures: bool = True,
    debug: bool = False,
) -> str:
    """单篇论文 → CSV。返回 CSV 路径。

    `use_figures=False` 时**跳过图提取**（不调用 vision）——快很多，适合：
      · 只关心正文/表格里印出来的数；
      · 调试 / 验证流程本身（比如溯源），不想每次都为图付调用。
    溯源仍然对正文和表格生效；图里的值会显示成「查无出处」（它们确实不在正文里）。

    `debug=True` 时把中间产物落盘，方便定位问题：
      · `<输出名>.figures.txt` —— 每张图提取出的文本（有序 + 页码）
      · `<输出名>.tables.json`  —— 坐标聚类检测到的表格结构
    """
    pdf = Path(pdf_path)
    schema = Schema.load(schema_path)
    composition = None
    if composition_path:
        composition = json.loads(Path(composition_path).read_text(encoding="utf-8"))

    # 输出路径先算出来：主 CSV、溯源 CSV、debug 产物都放一起
    output = Path(output_path) if output_path else pdf.with_suffix(".csv")

    import time
    _t0 = time.time()
    print(f"解析 PDF: {pdf.name}")
    text, pages_text, pages_images = parse_pdf(pdf, image_dir=settings.image_dir)
    _t_parse = time.time() - _t0

    # 接入坐标聚类表格（无边框表），与 PyMuPDF 的表格互补
    _t1 = time.time()
    detected = detect_tables(str(pdf))
    _t_table = time.time() - _t1
    if detected:
        text += "\n\n" + _format_detected_tables(detected)
        print(f"  坐标聚类检测到 {len(detected)} 个无边框表格")
    if debug:
        dbg_tables = output.with_name(output.stem + ".tables.json")
        dbg_tables.write_text(
            json.dumps(detected, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"  [debug] 表格结构已写：{dbg_tables.name}（{len(detected)} 个）")

    # 图数据转文本（多模态），附加到全文，让 LLM 统一抽取
    # **保留页码映射**：溯源要靠它把「图里读出来的值」归到正确的页。
    # 喂给 LLM 的文本内容一字不变（下面 join 的是 f["text"]）。
    figure_texts: list[dict] = []
    _t2 = time.time()
    if not use_figures:
        print("  跳过图提取（--no-figures）：不调用 vision")
    else:
        def _one(pn: int, fname: str) -> dict | None:
            img_path = Path(settings.image_dir) / fname
            try:
                ft = figure_to_text(str(img_path), figure_label=f"图(第{pn}页)")
                return {"page": pn, "filename": fname, "text": ft} if ft else None
            except Exception as e:
                logger.warning("图 %s 提取失败: %s", fname, e)
                return None

        tasks = [(pn, fname) for pn in sorted(pages_images) for fname in pages_images[pn]]
        # 图之间并发（实测 4 并发无 429，耗时约 ÷2）。线程池保序，结果按页/文件名顺序回来。
        workers = max(1, int(os.getenv("PAPER2DATA_FIGURE_WORKERS", "4")))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for out in ex.map(lambda t: _one(*t), tasks):
                if out:
                    figure_texts.append(out)
        if figure_texts:
            # ⚠️ 图文本必须放在**正文前面**：dev02 正文 62k 字符，图文本 6.7k，
            # 拼在末尾会被 text_limit 截断（之前 40k）——所有读数白做。
            fig_block = "## 图表数据(从图提取)\n" + "\n\n".join(f["text"] for f in figure_texts)
            text = fig_block + "\n\n## 论文正文\n" + text
            print(f"  从 {len(figure_texts)} 张图提取到图表数据（已前置，避免被截断）")
            # 贯穿性检查：图文本到底有没有完整进入 prompt、有没有被截断
            limit = 120000
            print(f"  [检查] 图文本 {len(fig_block)} 字符 / 总文本 {len(text)} 字符 / "
                  f"上限 {limit} → {'⚠ 会截断' if len(text) > limit else '完整送入'}")
    _t_fig = time.time() - _t2
    if debug:
        dbg_fig = output.with_name(output.stem + ".figures.txt")
        blocks = [
            f"### 第{f['page']}页  {f['filename']}\n{f['text']}"
            for f in figure_texts
        ]
        dbg_fig.write_text("\n\n".join(blocks) or "（没有提取到图文本）", encoding="utf-8")
        print(f"  [debug] 图文本已写：{dbg_fig.name}（{len(figure_texts)} 张）")

    print("LLM 抽取中...")
    _t3 = time.time()
    result = extract_dataset(text, schema)
    _t_extract = time.time() - _t3
    print(f"  抽到 {len(result.records)} 条试验, {len(result.new_fields)} 个新字段")
    print(f"  [计时] 解析 {_t_parse:.1f}s / 表格 {_t_table:.1f}s / "
          f"图 {_t_fig:.1f}s / 抽取 {_t_extract:.1f}s")
    if result.new_fields:
        for nf in result.new_fields:
            print(f"    · 新字段 {nf.name} ({nf.label})")

    # 溯源（零 LLM）：把每个抽出来的值回溯到原文「页码 + 证据句」。
    # **必须在 normalize 之前**——那之后就只剩 4000.0，"4000 kPa" 的原始串没了。
    try:
        from .export import save_provenance_csv
        from .verify import build_provenance, to_csv_rows

        prov = build_provenance(
            result.records, schema, pages_text,
            tables=detected, figure_texts=figure_texts, new_fields=result.new_fields,
        )
        prov_path = output.with_name(output.stem + ".provenance.csv")
        save_provenance_csv(to_csv_rows(prov), prov_path)

        s = prov.summary
        rate = 100 - s["查无出处比例"] if s["分母"] else 0.0
        print(f"  溯源：可溯源 {s['可溯源']}/{s['分母']}（{rate:.1f}%），"
              f"查无出处 {s['查无出处']}（{s['查无出处比例']}%）")
        src = s["按来源"]
        print(f"    来源：正文 {src.get('text', 0)} / 表格 {src.get('table', 0)} / "
              f"图 {src.get('figure', 0)} / 无法定位 {s['查无出处']}")
        if s["歧义"] or s["完全未命中"]:
            print(f"    查无出处拆分：歧义 {s['歧义']} / 完全未命中 {s['完全未命中']}")
        if s["查无出处最多的字段"]:
            worst = "、".join(
                f"{w['field']} {w['count']}（{w['ratio']}%）" for w in s["查无出处最多的字段"]
            )
            print(f"    查无出处最多的字段：{worst}")
        print(f"    已导出：{prov_path.name}")
    except Exception as e:  # 溯源是审计附加物，不能拖垮主 CSV
        logger.error("溯源失败（不影响主 CSV）：%s", e)

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
    parser.add_argument(
        "--no-figures", action="store_true",
        help="跳过图提取（不调用 vision，快很多）。溯源仍对正文和表格生效",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="落盘中间产物：<输出名>.figures.txt（每张图的提取文本）"
             " + <输出名>.tables.json（表格检测结构）",
    )
    args = parser.parse_args()

    if not Path(args.pdf).exists():
        parser.error(f"文件不存在: {args.pdf}")

    try:
        extract_pdf(args.pdf, args.schema, args.output, args.composition,
                    use_figures=not args.no_figures, debug=args.debug)
    except Exception as e:
        logger.error("处理失败：%s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
