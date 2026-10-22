"""Verification CLI —— 独立命令：抽取 + 验证 + 输出报告。

用法：
    uv run python -m paper_trans.verify_cli verify paper.pdf
"""

import logging
from pathlib import Path

import typer

from .config import settings
from .parser import parse_pdf
from .extractor import extract_with_review
from .pdf_parser import detect_tables
from .verify import verify_paper, save_report
from .renderer import (
    render_markdown,
    insert_images_by_page,
    save_markdown,
    save_json,
)

app = typer.Typer(
    name="paper-verify",
    help="对论文抽取结果运行可靠性验证（规则校验 + 证据溯源 + 交叉验证）",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trans")


@app.callback()
def _callback() -> None:
    """paper-verify 命令组入口。"""
    pass


@app.command()
def verify(
    pdf_path: str = typer.Argument(..., help="PDF 论文路径"),
    output_dir: str = typer.Option(
        settings.output_dir, "--output", "-o", help="输出目录"
    ),
    image_dir: str = typer.Option(
        settings.image_dir, "--images", help="图片输出目录"
    ),
    model: str = typer.Option(
        settings.model_name, "--model", "-m", help="LLM 模型名称"
    ),
    no_crosscheck: bool = typer.Option(
        False, "--no-crosscheck", help="跳过 LLM 交叉验证（只做零成本校验）"
    ),
) -> None:
    """抽取一篇 PDF，然后运行验证管线并输出提取结果 + 验证报告。"""
    pdf = Path(pdf_path)
    if not pdf.exists():
        typer.echo(f"错误: 文件不存在 — {pdf_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f" 解析 PDF: {pdf.name}")
    text, pages_text, pages_images = parse_pdf(pdf, image_dir=image_dir)

    typer.echo(f" 提取数据 (模型: {model})...")
    tables = detect_tables(str(pdf))
    data = extract_with_review(text, model=model, tables=tables)
    typer.echo(
        f"   提取到: {len(data.tables)} 个表格, {len(data.figures)} 个图表, "
        f"{len(data.numerical_data)} 个数值数据点"
    )

    typer.echo(" 运行验证管线...")
    report = verify_paper(
        text,
        data,
        pages_text=pages_text,
        tables=tables,
        model=model,
        do_crosscheck=not no_crosscheck,
    )
    typer.echo(f"   {report.summary}")

    typer.echo(" 保存提取结果与报告...")
    md = render_markdown(data, str(pdf))
    if pages_images:
        md = insert_images_by_page(md, pages_images)
    data_md = save_markdown(md, data, output_dir)
    data_json = save_json(data, output_dir)
    verify_md, verify_json = save_report(report, output_dir)

    typer.echo(f"\n 完成!")
    typer.echo(f"   提取结果: {data_md} / {data_json}")
    typer.echo(f"   验证报告: {verify_md} / {verify_json}")


def main() -> None:
    app()


if __name__ == "__main__":
    main()
