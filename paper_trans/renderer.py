"""输出渲染 —— Markdown + JSON。"""

import json
import re
import logging
from datetime import datetime
from pathlib import Path

from .schemas import PaperData

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Markdown 渲染
# ---------------------------------------------------------------------------

def render_markdown(data: PaperData, source: str) -> str:
    """将 PaperData 渲染为 Markdown 文档。

    章节顺序：标题 → 摘要 → 目标 → 表格 → 图表 → 数值数据 → 方法 → 发现 → 结论
    数据章节（表格/图表/数值）排在前面，体现"数据优先"原则。
    """
    d = data.model_dump()
    md: list[str] = []

    # ── 标题 + 元信息 ──
    md.append(f"# {d.get('paper_title') or 'Untitled'}\n")
    if d.get("authors"):
        md.append(f"**作者:** {', '.join(d['authors'])}")
    if d.get("journal"):
        md.append(f"**期刊:** {d['journal']}")
    if d.get("year"):
        md.append(f"**年份:** {d['year']}")
    if d.get("doi"):
        md.append(f"**DOI:** [{d['doi']}](https://doi.org/{d['doi']})")
    if d.get("field_tags"):
        md.append(f"**标签:** {', '.join(d['field_tags'])}")
    md.append(f"**来源:** {source}")
    md.append(f"**提取时间:** {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ── 摘要 ──
    if d.get("abstract"):
        md.append("---\n## 摘要\n")
        md.append(d["abstract"] + "\n")

    # ── 研究目标 ──
    if d.get("research_objective"):
        md.append("---\n## 研究目标\n")
        md.append(d["research_objective"] + "\n")

    # ── 表格（数据优先） ──
    _render_tables(d.get("tables", []), md)

    # ── 图表 ──
    _render_figures(d.get("figures", []), md)

    # ── 数值数据汇总 ──
    _render_numerical_data(d.get("numerical_data", []), md)

    # ── 方法（概括） ──
    _render_methods(d.get("methods", []), md)

    # ── 关键发现 ──
    if d.get("key_findings"):
        md.append("---\n## 关键发现\n")
        for f in d["key_findings"]:
            md.append(f"- {f}")
        md.append("")

    # ── 结论 ──
    if d.get("conclusions"):
        md.append("---\n## 结论\n")
        md.append(d["conclusions"] + "\n")

    # ── 脚注 ──
    md.append(f"\n---\n*数据由 {_model_name()} 提取，请核对原文。*\n")

    return "\n".join(md)


def _render_tables(tables: list[dict], md: list[str]) -> None:
    """渲染表格章节。"""
    if not tables:
        return
    md.append("---\n## 表格\n")
    for t in tables:
        label = t.get("table_id") or "表格"
        md.append(f"### {label}\n")
        if t.get("caption"):
            md.append(f"*{t['caption']}*\n")

        headers = t.get("headers", [])
        rows = t.get("rows", [])
        if headers and rows:
            md.append("| " + " | ".join(str(h) for h in headers) + " |")
            md.append("|" + "|".join("------" for _ in headers) + "|")
            for row in rows:
                padded = list(row) + [""] * (len(headers) - len(row))
                cells = [
                    str(c).replace("|", "\\|").replace("\n", " ")
                    for c in padded[: len(headers)]
                ]
                md.append("| " + " | ".join(cells) + " |")
            md.append("")

        if t.get("notes"):
            md.append(f"*注: {t['notes']}*\n")


def _render_figures(figures: list[dict], md: list[str]) -> None:
    """渲染图表章节。"""
    if not figures:
        return
    md.append("---\n## 图表\n")
    for f in figures:
        label = f.get("figure_id") or "图表"
        md.append(f"### {label}\n")
        if f.get("caption"):
            md.append(f"*{f['caption']}*\n")
        md.append(f"{f.get('description', '')}\n")
        if f.get("chart_type"):
            md.append(f"- **类型:** {f['chart_type']}")
        if f.get("key_data_points"):
            md.append("- **关键数据点:**")
            for k, v in f["key_data_points"].items():
                md.append(f"  - {k}: {v}")
        md.append("")


def _render_numerical_data(ndata: list[dict], md: list[str]) -> None:
    """渲染数值数据汇总表。"""
    if not ndata:
        return
    md.append("---\n## 数值数据\n")
    md.append("| 名称 | 数值 | 单位 | 上下文 | 类别 |")
    md.append("|------|------|------|--------|------|")
    for nd in ndata:
        name = nd.get("name", "-")
        value = nd.get("value", "-")
        unit = nd.get("unit") or "-"
        context = nd.get("context") or "-"
        category = nd.get("category") or "-"
        md.append(
            f"| {_esc(name)} | {_esc(value)} | {_esc(unit)} "
            f"| {_esc(context)} | {_esc(category)} |"
        )
    md.append("")


def _render_methods(methods: list[dict], md: list[str]) -> None:
    """渲染方法章节。"""
    if not methods:
        return
    md.append("---\n## 方法\n")
    for i, m in enumerate(methods, 1):
        name = m.get("name", f"方法 {i}")
        md.append(f"### {name}\n")
        md.append(f"{m.get('description', '')}\n")
        techs = m.get("techniques", [])
        if techs:
            md.append("**技术:** " + ", ".join(techs) + "\n")


def _esc(s: str) -> str:
    """转义 Markdown 表格中的特殊字符。"""
    return s.replace("|", "\\|").replace("\n", " ")


def _model_name() -> str:
    """获取当前使用的模型名称（用于脚注）。"""
    from .config import settings

    return settings.model_name


# ---------------------------------------------------------------------------
# 图片插入
# ---------------------------------------------------------------------------

def insert_images_by_page(
    markdown: str,
    pages_images: dict[int, list[str]],
) -> str:
    """将 PDF 提取的图片按页码分组，插入文末「附图」附录。

    Args:
        markdown: 渲染后的 Markdown 文本。
        pages_images: {页码: [图片文件名列表]}。

    Returns:
        插入图片附录后的 Markdown。
    """
    if not pages_images:
        return markdown

    refs: list[str] = []
    for pn in sorted(pages_images):
        label = "网页图片" if pn == 0 else f"第 {pn} 页"
        refs.append(f"\n**{label}**\n")
        for img in pages_images[pn]:
            refs.append(f"![图](images/{img})\n")

    appendix = "\n---\n## 附图\n\n" + "\n".join(refs) + "\n"

    footer = "\n---\n*数据由"
    if footer in markdown:
        markdown = markdown.replace(footer, f"{appendix}{footer}")
    else:
        markdown += f"\n{appendix}\n"

    return markdown


# ---------------------------------------------------------------------------
# 文件保存
# ---------------------------------------------------------------------------

def save_markdown(
    md_text: str,
    data: PaperData,
    output_dir: str | Path = "./output",
) -> str:
    """保存 Markdown 文件，文件名从标题生成。

    Returns:
        保存的文件路径。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(data.paper_title or "paper")
    filepath = out / f"{filename}.md"
    filepath.write_text(md_text, encoding="utf-8")
    logger.info("Markdown 已保存: %s", filepath)
    return str(filepath)


def save_json(
    data: PaperData,
    output_dir: str | Path = "./output",
) -> str:
    """保存 JSON 文件。

    Returns:
        保存的文件路径。
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    filename = _sanitize_filename(data.paper_title or "paper")
    filepath = out / f"{filename}.json"
    filepath.write_text(
        json.dumps(data.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("JSON 已保存: %s", filepath)
    return str(filepath)


def _sanitize_filename(title: str, max_len: int = 80) -> str:
    """从论文标题生成安全的文件名。"""
    name = re.sub(r"[^\w\s-]", "", title)[:max_len]
    name = re.sub(r"[-\s]+", "_", name)
    return name.strip("_") or "paper"
