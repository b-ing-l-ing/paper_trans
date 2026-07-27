"""网页论文提取 —— HTML → 结构化数据。"""

from __future__ import annotations

import hashlib
import logging
import re
import tempfile
import urllib.request
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

from ..schemas import PaperData
from ..extractor import extract_with_review, extract_data

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# HTML → 纯文本
# ---------------------------------------------------------------------------

class _TextExtractor(HTMLParser):
    """轻量 HTML→文本，保留段落结构。"""

    def __init__(self):
        super().__init__()
        self.text: list[str] = []
        self._skip = False
        self._table_mode = False
        self._row: list[str] = []
        self._cell = ""

        self._skip_tags = {"script", "style", "nav", "footer", "header", "noscript"}
        self._block_tags = {
            "div", "p", "h1", "h2", "h3", "h4", "h5", "h6",
            "li", "tr", "section", "article", "br", "hr",
        }

    def handle_starttag(self, tag, attrs):
        if tag in self._skip_tags:
            self._skip = True
        elif tag == "table":
            self._table_mode = True
            self.text.append("")  # table 前后空行
        elif tag == "tr" and self._table_mode:
            self._row = []
        elif tag in ("td", "th") and self._table_mode:
            self._cell = ""

    def handle_endtag(self, tag):
        if tag in self._skip_tags:
            self._skip = False
        elif tag == "table":
            self._table_mode = False
            self.text.append("")
        elif tag == "tr" and self._table_mode:
            if self._row:
                self.text.append("| " + " | ".join(self._row) + " |")
        elif tag in ("td", "th") and self._table_mode:
            self._row.append(self._cell.strip())
        elif tag in self._block_tags:
            if self.text and self.text[-1] != "":
                self.text.append("")

    def handle_data(self, data):
        if self._skip:
            return
        stripped = data.strip()
        if not stripped:
            return
        if self._table_mode and self._cell is not None:
            self._cell += " " + stripped
        else:
            if self.text and self.text[-1] and not self.text[-1].endswith(" "):
                self.text[-1] += " "
            if self.text:
                self.text[-1] += stripped
            else:
                self.text.append(stripped)

    def get_text(self) -> str:
        return "\n".join(line.strip() for line in self.text if line.strip())


def html_to_text(html: str) -> str:
    """HTML → 纯文本（保留基本结构 + 表格）。"""
    parser = _TextExtractor()
    parser.feed(html)
    return parser.get_text()


# ---------------------------------------------------------------------------
# 网页 → PaperData
# ---------------------------------------------------------------------------

def scrape_paper(
    html: str,
    url: str = "",
    with_review: bool = True,
    model: str | None = None,
) -> PaperData:
    """从网页 HTML 中提取论文结构化数据。

    Args:
        html: 网页 HTML 内容。
        url: 来源 URL（用于记录）。
        with_review: 是否启用自审。

    Returns:
        PaperData 实例。
    """
    # 1. HTML → 纯文本（降噪）
    text = html_to_text(html)

    if len(text) < 200:
        raise ValueError(
            f"提取的文本过短（{len(text)} 字符），可能页面需要登录或 JS 未执行。"
        )

    logger.info("HTML → 文本: %s 字符", f"{len(text):,}")

    # 2. 走现有提取管线
    if with_review:
        data = extract_with_review(text, model=model)
    else:
        data = extract_data(text, model=model)

    return data


def _has_data_tables(html: str) -> bool:
    """HTML 中是否包含真正的数据表格（排除公式排版用 table）。"""
    # 找所有 <table，排除 class 含 equation 的
    for m in re.finditer(r'<table[^>]*>', html, re.I):
        tag = m.group()
        if 'equation' not in tag.lower():
            return True
    return False


def _find_pdf_url(page, url: str) -> str | None:
    """尝试从页面中找出 PDF 下载链接。"""
    # arXiv: abs 或 html → pdf（去掉版本号后缀 v1/v2）
    m = re.match(r"(https?://arxiv\.org)/(?:abs|html)/([^/]+)", url)
    if m:
        paper_id = re.sub(r"v\d+$", "", m.group(2))
        return f"{m.group(1)}/pdf/{paper_id}"

    # 通用策略：meta 标签
    try:
        el = page.query_selector('meta[name="citation_pdf_url"]')
        if el:
            href = el.get_attribute("content")
            if href:
                return href
    except Exception as e:
        logger.debug("PDF URL 查找失败 (meta): %s", e)

    # 通用策略：页面里的 PDF 链接
    try:
        links = page.query_selector_all('a[href$=".pdf"]')
        for link in links:
            href = link.get_attribute("href")
            if href and "pdf" in href.lower():
                return href
    except Exception as e:
        logger.debug("PDF URL 查找失败 (links): %s", e)

    return None


def _extract_web_images(page, url: str, image_dir: str = "./output/images") -> dict[int, list[str]]:
    """下载网页中所有图片，返回 {页码: [文件名]}。"""
    out = Path(image_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 文件名前缀
    prefix = re.sub(r"[^\w-]", "_", url.split("//")[-1].split("/")[-1] or "page")[:40]

    # 复用浏览器 cookie
    ctx_cookies = page.context.cookies()
    cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in ctx_cookies)

    imgs = page.query_selector_all("img")
    result: dict[int, list[str]] = {}
    seen_hashes: set[str] = set()
    img_idx = 0

    for img in imgs:
        src = img.get_attribute("src") or img.get_attribute("data-src")
        if not src:
            continue

        src = urljoin(url, src)
        img_idx += 1

        try:
            req = urllib.request.Request(str(src))
            req.add_header("User-Agent", "paper-trans/0.2")
            req.add_header("Referer", url)
            if cookie_str:
                req.add_header("Cookie", cookie_str)

            data = urllib.request.urlopen(req, timeout=10).read()

            # 跳过极小图（图标、装饰）
            if len(data) < 1024:
                continue

            h = hashlib.md5(data).hexdigest()
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            ext = src.rsplit(".", 1)[-1].split("?")[0]
            if ext.lower() not in ("png", "jpg", "jpeg", "gif", "webp", "svg"):
                ext = "png"

            filename = f"{prefix}_img{img_idx}.{ext}"
            (out / filename).write_bytes(data)
            result.setdefault(0, []).append(filename)

        except Exception as e:
            logger.debug("图片下载失败 %s: %s", src, e)
            continue

    if result:
        logger.info("下载网页图片: %d 张", sum(len(v) for v in result.values()))
    return result


def _pdf_name_from_url(pdf_url: str) -> str:
    """从 PDF URL 提取有意义的文件名前缀。"""
    parts = pdf_url.rstrip("/").rsplit("/", 1)
    name = parts[-1] if len(parts) > 1 else "paper"
    name = re.sub(r"\.pdf$", "", name, flags=re.I)
    return re.sub(r"[^\w-]", "_", name)[:60]


def _download_pdf(page, pdf_url: str) -> bytes:
    """通过 urllib 下载 PDF，复用浏览器 cookie。"""
    cookies = page.context.cookies()
    cookie_str = "; ".join(f"{c['name']}={c['value']}" for c in cookies)

    req = urllib.request.Request(pdf_url)
    if cookie_str:
        req.add_header("Cookie", cookie_str)
    req.add_header("User-Agent", "paper-trans/0.2")

    with urllib.request.urlopen(req, timeout=30) as resp:
        return resp.read()


def scrape_paper_from_url(
    url: str,
    session_file: str = "./session.json",
    headless: bool = True,
    with_review: bool = True,
    auto_pdf: bool = True,
    model: str | None = None,
) -> tuple[PaperData, dict, dict]:
    """完整流程：打开浏览器 → 加载页面 → 提取数据。

    HTML 内容过短时自动尝试下载 PDF 回退到 extract 管线。

    Args:
        url: 论文页面 URL。
        session_file: 登录态文件路径。
        headless: 是否无头模式。
        with_review: 是否启用自审。
        auto_pdf: 内容不足时是否自动下载 PDF。
        model: LLM 模型名称，None 则使用默认配置。

    Returns:
        (PaperData, pages_text, pages_images) — 后两者在 HTML 提取时为空字典。
    """
    from .browser import ScraperBrowser
    from ..parser import parse_pdf
    from ..extractor import extract_with_review as pdf_extract

    with ScraperBrowser(session_file, headless=headless) as page:
        logger.info("加载页面: %s", url)
        page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        page.wait_for_timeout(2_000)

        html = page.content()
        logger.info("HTML 长度: %s 字符", f"{len(html):,}")

        text = html_to_text(html)
        has_tables = _has_data_tables(html)

        # 内容不足 或 缺表格 → 试着下 PDF
        need_pdf = len(text) < 5000 or not has_tables
        if auto_pdf and need_pdf:
            reason = "内容不足" if len(text) < 5000 else "HTML 无表格"
            logger.info("%s（%s 字符），尝试下载 PDF...", reason, f"{len(text):,}")
            pdf_url = _find_pdf_url(page, url)
            if pdf_url:
                logger.info("找到 PDF 链接: %s", pdf_url)
                try:
                    pdf_bytes = _download_pdf(page, pdf_url)
                    pdf_name = _pdf_name_from_url(pdf_url)
                    tmp_path = str(Path(tempfile.gettempdir()) / f"{pdf_name}.pdf")
                    Path(tmp_path).write_bytes(pdf_bytes)
                    try:
                        pdf_text, pages_text, pages_images = parse_pdf(tmp_path)
                        logger.info("PDF 提取文本: %s 字符", f"{len(pdf_text):,}")
                        data = pdf_extract(pdf_text, model=model)
                        return data, pages_text, pages_images
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                except Exception as e:
                    logger.warning("PDF 下载失败: %s，回退 HTML 提取", e)
            else:
                logger.warning("未找到 PDF 链接，用 HTML 提取")

        # HTML 路径：也抓网页图片
        pages_images = _extract_web_images(page, url)
        data = scrape_paper(html, url, with_review=with_review, model=model)

    return data, {}, pages_images
