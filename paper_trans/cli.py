"""CLI 入口 —— Typer 命令行界面。"""

import sys
import logging
from pathlib import Path

import typer

from . import __version__
from .config import settings
from .parser import parse_pdf
from .extractor import extract_with_review
from .renderer import (
    render_markdown,
    insert_images_by_page,
    save_markdown,
    save_json,
)

app = typer.Typer(
    name="paper-trans",
    help="从科学论文中提取结构化数据（支持 PDF + 网页）",
    add_completion=False,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("paper_trans")


# ═══════════════════════════════════════════════════════════════════════════════
# PDF 提取
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def extract(
    pdf_path: str = typer.Argument(..., help="PDF 论文路径"),
    output_dir: str = typer.Option(
        settings.output_dir, "--output", "-o", help="输出目录"
    ),
    image_dir: str = typer.Option(
        settings.image_dir, "--images", help="图片输出目录"
    ),
    no_review: bool = typer.Option(
        False, "--no-review", help="跳过自查审核（省 API 费用）"
    ),
    multimodal: bool = typer.Option(
        False, "--multimodal", help="启用多模态提取（Gemini，文本+图片）"
    ),
    model: str = typer.Option(
        settings.model_name, "--model", "-m", help="LLM 模型名称"
    ),
    version: bool = typer.Option(
        False, "--version", "-V", help="显示版本"
    ),
) -> None:
    """从单篇 PDF 论文中提取结构化数据。

    输出 Markdown（人读）和 JSON（机读）到指定目录。
    """
    if version:
        typer.echo(f"paper-trans v{__version__}")
        raise typer.Exit()

    pdf = Path(pdf_path)
    if not pdf.exists():
        typer.echo(f"错误: 文件不存在 — {pdf_path}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f" 解析 PDF: {pdf.name}")
    text, pages_text, pages_images = parse_pdf(pdf, image_dir=image_dir)
    typer.echo(f"   提取文本 {len(text):,} 字符，{len(pages_images)} 页含图片")

    typer.echo(f" 提取数据 (模型: {model})...")
    from .pdf_parser import detect_tables
    tables = detect_tables(str(pdf))
    if tables:
        typer.echo(f"   检测到 {len(tables)} 个结构化表格")
    if multimodal:
        from .extractor import extract_multimodal
        img_paths = [
            str(Path(image_dir) / f)
            for files in pages_images.values()
            for f in files
        ]
        typer.echo(f"   多模态模式：{len(img_paths)} 张图片")
        data = extract_multimodal(text, img_paths, model=model, tables=tables)
    elif no_review:
        from .extractor import extract_data
        data = extract_data(text, model=model, tables=tables)
    else:
        data = extract_with_review(text, model=model, tables=tables)

    typer.echo(f"   提取到: {len(data.tables)} 个表格, "
               f"{len(data.figures)} 个图表, "
               f"{len(data.numerical_data)} 个数值数据点")

    typer.echo(" 渲染 Markdown...")
    md = render_markdown(data, str(pdf))
    md = insert_images_by_page(md, pages_images)

    md_path = save_markdown(md, data, output_dir)
    json_path = save_json(data, output_dir)

    typer.echo(f"\n 完成!")
    typer.echo(f"   Markdown: {md_path}")
    typer.echo(f"   JSON:     {json_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 网页爬取
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def login(
    url: str = typer.Argument(..., help="登录页面 URL"),
    session_file: str = typer.Option(
        "./session.json", "--session", "-s", help="Session 保存路径"
    ),
) -> None:
    """打开有头浏览器，手动登录后保存 cookie / session。

    示例：
        paper-trans login https://example.com/login
        paper-trans login https://example.com/login -s my_session.json
    """
    from .scraper import login_and_save

    typer.echo(f" 打开浏览器: {url}")
    login_and_save(url, session_file)
    typer.echo(f" session 已保存: {session_file}")


@app.command()
def scrape(
    url: str = typer.Argument(..., help="论文网页 URL"),
    output_dir: str = typer.Option(
        settings.output_dir, "--output", "-o", help="输出目录"
    ),
    session_file: str = typer.Option(
        "./session.json", "--session", "-s", help="Session 文件路径"
    ),
    headed: bool = typer.Option(
        False, "--headed", help="显示浏览器窗口（调试用）"
    ),
    no_review: bool = typer.Option(
        False, "--no-review", help="跳过自查审核"
    ),
    model: str = typer.Option(
        settings.model_name, "--model", "-m", help="LLM 模型名称"
    ),
) -> None:
    """从论文网页中提取结构化数据。

    自动使用已保存的 session（先 login 一次即可）。
    输出 Markdown + JSON。

    示例：
        paper-trans login https://publisher.com/login     # 只需一次
        paper-trans scrape https://publisher.com/paper/xxx
    """
    from .scraper import scrape_paper_from_url

    typer.echo(f" 加载页面: {url}")
    data, pages_text, pages_images = scrape_paper_from_url(
        url,
        session_file=session_file,
        headless=not headed,
        with_review=not no_review,
        model=model,
    )

    typer.echo(f"   提取到: {len(data.tables)} 个表格, "
               f"{len(data.figures)} 个图表, "
               f"{len(data.numerical_data)} 个数值数据点")

    typer.echo(" 渲染 Markdown...")
    md = render_markdown(data, url)
    if pages_images:
        md = insert_images_by_page(md, pages_images)

    md_path = save_markdown(md, data, output_dir)
    json_path = save_json(data, output_dir)

    typer.echo(f"\n 完成!")
    typer.echo(f"   Markdown: {md_path}")
    typer.echo(f"   JSON:     {json_path}")


def main() -> None:
    """入口（python -m paper_trans）。"""
    app()


# ═══════════════════════════════════════════════════════════════════════════════
# 批量处理
# ═══════════════════════════════════════════════════════════════════════════════

@app.command()
def batch(
    directory: str = typer.Argument(..., help="论文 PDF 目录"),
    output_dir: str = typer.Option(
        settings.output_dir, "--output", "-o", help="输出目录"
    ),
    image_dir: str = typer.Option(
        settings.image_dir, "--images", help="图片输出目录"
    ),
    no_review: bool = typer.Option(
        False, "--no-review", help="跳过自查审核"
    ),
    multimodal: bool = typer.Option(
        False, "--multimodal", help="启用多模态提取（Gemini）"
    ),
    model: str = typer.Option(
        settings.model_name, "--model", "-m", help="LLM 模型名称"
    ),
) -> None:
    """批量处理目录下所有 PDF 论文。"""
    from .parser import parse_pdf
    from .renderer import render_markdown, insert_images_by_page, save_markdown, save_json
    from .pdf_parser import detect_tables

    pdf_dir = Path(directory)
    if not pdf_dir.is_dir():
        typer.echo(f"错误: 目录不存在 — {directory}", err=True)
        raise typer.Exit(code=1)

    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        typer.echo(f"目录内无 PDF: {directory}", err=True)
        raise typer.Exit(code=1)

    typer.echo(f"批量处理 {len(pdfs)} 篇论文\n")
    ok = fail = 0

    for i, pdf in enumerate(pdfs, 1):
        typer.echo(f"[{i}/{len(pdfs)}] {pdf.name}")
        try:
            text, pages_text, pages_images = parse_pdf(pdf, image_dir=image_dir)
            tables = detect_tables(str(pdf))

            if multimodal:
                from .extractor import extract_multimodal
                img_list = [str(Path(image_dir) / f)
                            for files in pages_images.values() for f in files]
                data = extract_multimodal(text, img_list, model=model, tables=tables)
            elif no_review:
                from .extractor import extract_data
                data = extract_data(text, model=model, tables=tables)
            else:
                from .extractor import extract_with_review
                data = extract_with_review(text, model=model, tables=tables)

            md = render_markdown(data, str(pdf))
            if pages_images:
                md = insert_images_by_page(md, pages_images)
            save_markdown(md, data, output_dir)
            save_json(data, output_dir)
            typer.echo(f"   -> OK")
            ok += 1
        except Exception as e:
            typer.echo(f"   -> FAIL: {e}")
            fail += 1

    typer.echo(f"\n完成: {ok} 成功, {fail} 失败")


if __name__ == "__main__":
    main()
