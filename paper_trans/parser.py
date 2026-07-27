"""PDF 解析 —— 提取文本和嵌入图片。"""

import hashlib
import logging
import re
from pathlib import Path
from collections import defaultdict
from io import BytesIO

import fitz
import pdfplumber
from PIL import Image

logger = logging.getLogger(__name__)


def _mark_tables(text: str) -> str:
    """在文本中标记表格区域，帮助 LLM 聚焦提取。

    策略：找到 'Table N' / '表 N' 标注的行，
    把该行到下一个空行块之间的内容标记为表格区域。
    """
    lines = text.split("\n")
    result: list[str] = []
    in_table = False
    table_start_pattern = re.compile(
        r"(?:^|\n)(?:Table|表)\s*\d+", re.IGNORECASE
    )

    for line in lines:
        stripped = line.strip()
        # 表头标记
        if table_start_pattern.match(stripped):
            if in_table:
                result.append("[TABLE_END]")
            result.append("[TABLE_BEGIN]")
            result.append(line)
            in_table = True
        # 空行结束表格区域
        elif in_table and stripped == "":
            result.append("[TABLE_END]")
            result.append(line)
            in_table = False
        else:
            result.append(line)

    if in_table:
        result.append("[TABLE_END]")

    return "\n".join(result)


def _clean_text(text: str) -> str:
    """修复 pdfplumber 的中文字符间多余空格，保留数字/英文之间的空格（表格列分隔）。"""
    # 中文字符之间的空格 → 删除
    text = re.sub(r"(?<=[一-鿿])\s+(?=[一-鿿])", "", text)
    # 中文后紧跟英文/数字之间的空格 → 删除（如 "�� 1" → "��1"）
    text = re.sub(r"(?<=[一-鿿])\s+(?=[\w])", "", text)
    # 英文/数字后紧跟中文之间的空格 → 删除（如 "1 ��" → "1��"）
    text = re.sub(r"(?<=[\w])\s+(?=[一-鿿])", "", text)
    return text


def _extract_tables(doc) -> str:
    """用 PyMuPDF 检测 PDF 中的表格结构，返回 markdown 表格文本。

    PyMuPDF 读 PDF 内部线条和文本坐标来定位表格，不受空单元格干扰。
    """
    import re

    all_tables: list[str] = []
    table_count = 0

    for page_num, page in enumerate(doc, 1):
        try:
            tables = page.find_tables(strategy="lines")
        except Exception:
            continue

        for tab in tables.tables:
            table_count += 1
            rows: list[list[str]] = []
            for row in tab.extract():
                rows.append([str(c or "") for c in row])

            if not rows:
                continue

            header = rows[0]
            md = [f"\n### Table {table_count}\n"]
            md.append("| " + " | ".join(header) + " |")
            md.append("|" + "|".join("---" for _ in header) + "|")
            for row in rows[1:]:
                cells = [c.replace("|", "\\|").replace("\n", " ") for c in row]
                while len(cells) < len(header):
                    cells.append("")
                md.append("| " + " | ".join(cells[:len(header)]) + " |")
            md.append("")
            all_tables.append("\n".join(md))

    if table_count:
        logger.info("检测到 %d 个表格结构", table_count)
    result = "\n".join(all_tables)
    if result:
        result = "## 内置表格数据（精确结构，优先使用）\n\n" + result
    return result


def _dedup_images(
    pages_images: dict[int, list[str]],
    image_dir: Path,
) -> dict[int, list[str]]:
    """过滤掉在多页重复出现的图片（页眉/页脚装饰元素）。

    规则：同一 hash 出现在超过半数含图页面上 → 视为装饰图，删除。
    """
    if len(pages_images) <= 1:
        return pages_images

    # 统计每张图的 hash（同时记录 file→hash 映射，避免二读）
    hash_pages: dict[str, list[int]] = defaultdict(list)
    file_hash: dict[tuple[int, str], str] = {}  # (page, filename) → hash
    for pn, files in pages_images.items():
        for f in files:
            data = (image_dir / f).read_bytes()
            h = hashlib.md5(data).hexdigest()
            hash_pages[h].append(pn)
            file_hash[(pn, f)] = h

    # 找出重复 hash（出现在超过半数含图页面）
    total_img_pages = len(pages_images)
    repeat_hashes = {
        h for h, pns in hash_pages.items()
        if len(pns) > total_img_pages / 2
    }

    if not repeat_hashes:
        return pages_images

    # 过滤并删除重复图片（复用第一轮 hash）
    filtered: dict[int, list[str]] = {}
    removed = 0
    for pn, files in pages_images.items():
        filtered[pn] = []
        for f in files:
            h = file_hash[(pn, f)]
            if h in repeat_hashes:
                (image_dir / f).unlink(missing_ok=True)
                removed += 1
            else:
                filtered[pn].append(f)

    if removed:
        logger.info("已过滤 %d 张重复图片（页眉/页脚）", removed)

    # 清除空页
    return {pn: files for pn, files in filtered.items() if files}


def parse_pdf(
    pdf_path: str | Path,
    image_dir: str | Path = "./output/images",
) -> tuple[str, dict[int, str], dict[int, list[str]]]:
    """解析 PDF：提取文本和嵌入图片。

    Args:
        pdf_path: PDF 文件路径。
        image_dir: 图片输出目录。

    Returns:
        (full_text, pages_text, pages_images)
        - full_text: 全文纯文本，页间用双换行分隔。
        - pages_text: {页码: 文本}。
        - pages_images: {页码: [图片文件名列表]}。
    """
    out = Path(image_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 用 PDF 文件名做前缀，避免多篇论文图片重名
    pdf_stem = Path(pdf_path).stem
    # 取前 40 个字符，去掉特殊字符
    prefix = re.sub(r'[^\w\s-]', '', pdf_stem)[:40].strip().replace(' ', '_')
    if not prefix:
        prefix = "paper"

    doc = fitz.open(str(pdf_path))
    pages_text: dict[int, str] = {}
    pages_images: dict[int, list[str]] = defaultdict(list)

    # pdfplumber 提取文本（布局感知，保留表格列对齐）
    try:
        plumber = pdfplumber.open(str(pdf_path))
        for page_num, ppage in enumerate(plumber.pages, 1):
            raw = ppage.extract_text() or ""
            pages_text[page_num] = _clean_text(raw)
        plumber.close()
    except Exception:
        logger.debug("pdfplumber 失败，回退 PyMuPDF 文本", exc_info=True)
        pages_text.clear()

    # PyMuPDF 提取文本（pdfplumber 失败时回退）
    if not pages_text:
        for page_num, page in enumerate(doc, 1):
            pages_text[page_num] = page.get_text("text")

    # 提取图片（不论 pdfplumber 是否成功都执行）
    for page_num, page in enumerate(doc, 1):
        for img_idx, img in enumerate(page.get_images()):
            xref = img[0]
            base_image = doc.extract_image(xref)
            ext = base_image["ext"]
            img_bytes = base_image["image"]

            # JPEG2000 → PNG 转换（科学论文常见格式）
            if ext in ("jpx", "jp2"):
                pil_img = Image.open(BytesIO(img_bytes))
                ext = "png"
                filename = f"{prefix}_p{page_num}_img{img_idx + 1}.{ext}"
                if pil_img.mode == "CMYK":
                    pil_img = pil_img.convert("RGB")
                pil_img.save(out / filename, "PNG")
            else:
                filename = f"{prefix}_p{page_num}_img{img_idx + 1}.{ext}"
                (out / filename).write_bytes(img_bytes)

            pages_images[page_num].append(filename)

    # 提取表格结构数据（PyMuPDF lines 策略，有线框表）
    table_md = _extract_tables(doc)

    doc.close()

    # 过滤重复图片（页眉/页脚等装饰元素）
    pages_images = _dedup_images(pages_images, out)

    full_text = "\n\n".join(pages_text.values())
    if table_md:
        full_text = table_md + "\n\n" + full_text

    # 标记潜在的表格区域（帮助 LLM 聚焦）
    full_text = _mark_tables(full_text)

    return full_text, pages_text, pages_images
