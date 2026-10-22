"""表格结构检测 —— 用 pdfplumber 坐标聚类恢复二维表格。

核心思路：不依赖 LLM，纯粹从 PDF 文字坐标重建单元格网格。
空单元格根据 x/y 坐标位置自然产生，不需要模型猜测。
"""

from __future__ import annotations

import logging
from pathlib import Path

import pdfplumber

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 公开 API
# ---------------------------------------------------------------------------

def detect_tables(
    pdf_path: str | Path,
    *,
    min_rows: int = 3,
    min_cols: int = 3,
) -> list[dict]:
    """检测 PDF 中的所有表格，返回结构化 JSON 列表。

    Args:
        pdf_path: PDF 文件路径。
        min_rows: 最少数据行数。
        min_cols: 最少列数。

    Returns:
        [{"page": int, "header": [str], "rows": [[str|null]]}, ...]
    """
    pdf_path = Path(pdf_path)
    all_tables: list[dict] = []

    with pdfplumber.open(str(pdf_path)) as plumber:
        for page_num, page in enumerate(plumber.pages, 1):
            words = page.extract_words(keep_blank_chars=True, x_tolerance=2)
            if not words:
                continue

            # 1. 按 y 聚类成行
            rows = _lines_from_words(words)

            # 2. 每行按 x 聚类成单元格
            cell_rows = [_cells_from_line(r) for r in rows]

            # 3. 找表格区域（连续多行、多列）
            regions = _table_regions(cell_rows, rows, min_rows, min_cols)

            # 4. 对齐成网格
            for start, end in regions:
                table = _build_grid(cell_rows[start:end], rows[start:end])
                if table and len(table["rows"]) >= min_rows:
                    all_tables.append({"page": page_num, **table})

    logger.info("检测到 %d 个表格结构", len(all_tables))
    return all_tables


# ---------------------------------------------------------------------------
# 第 1 步：词 → 行
# ---------------------------------------------------------------------------

def _lines_from_words(words: list[dict], y_gap_ratio: float = 2.5) -> list[list[dict]]:
    """按 y 坐标把词聚合成行。阈值自适应（中位行间距 × ratio）。"""
    if not words:
        return []

    sorted_w = sorted(words, key=lambda w: (w["top"], w["x0"]))

    gaps = []
    for i in range(1, len(sorted_w)):
        g = abs(sorted_w[i]["top"] - sorted_w[i - 1]["top"])
        if g > 0.5:
            gaps.append(g)
    gaps.sort()
    line_gap = max(gaps[len(gaps) // 2] * y_gap_ratio, 3.0) if gaps else 5.0

    lines: list[list[dict]] = []
    cur: list[dict] = []
    cur_y = None
    for w in sorted_w:
        if cur_y is None or abs(w["top"] - cur_y) <= line_gap:
            cur.append(w)
            if cur_y is None:
                cur_y = w["top"]
        else:
            if cur:
                cur.sort(key=lambda w: w["x0"])
                lines.append(cur)
            cur = [w]
            cur_y = w["top"]
    if cur:
        cur.sort(key=lambda w: w["x0"])
        lines.append(cur)
    return lines


# ---------------------------------------------------------------------------
# 第 2 步：行 → 单元格
# ---------------------------------------------------------------------------

def _cells_from_line(line: list[dict], x_gap: float = 8.0) -> list[list[dict]]:
    """把一行词按 x 间距聚类成单元格。"""
    if not line:
        return []
    cells: list[list[dict]] = []
    cur = [line[0]]
    for w in line[1:]:
        if w["x0"] - cur[-1]["x1"] > x_gap:
            cells.append(cur)
            cur = [w]
        else:
            cur.append(w)
    cells.append(cur)
    return cells


def _cell_text(words: list[dict]) -> str:
    """合并单元格内词为纯文本。"""
    return " ".join(w["text"].strip() for w in words if w["text"].strip())


# ---------------------------------------------------------------------------
# 第 3 步：找表格区域
# ---------------------------------------------------------------------------

def _table_regions(
    cell_rows: list[list[list[dict]]],
    raw_lines: list[list[dict]],
    min_rows: int,
    min_cols: int,
) -> list[tuple[int, int]]:
    """扫描行，找多列、多数字的连续区域。"""
    regions: list[tuple[int, int]] = []
    start = None

    for i, (cells, raw) in enumerate(zip(cell_rows, raw_lines)):
        n = len(cells)
        # 这一行的文本
        texts = [_cell_text(c) for c in cells]
        nums = sum(1 for t in texts if _is_number(t))
        # 表头行特征：多短词
        short = sum(1 for t in texts if 0 < len(t) < 20)
        # 数据行特征：多数字
        is_table_like = n >= min_cols and (nums >= 2 or short >= min_cols)

        if is_table_like:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_rows + 1:
                # 验证：至少一半行有数字
                region = cell_rows[start:i]
                data_rows = sum(
                    1 for c in region
                    if sum(1 for t in [_cell_text(x) for x in c] if _is_number(t)) >= 2
                )
                if data_rows / len(region) >= 0.4:
                    regions.append((start, i))
            start = None

    if start is not None and len(cell_rows) - start >= min_rows + 1:
        region = cell_rows[start:]
        data_rows = sum(
            1 for c in region
            if sum(1 for t in [_cell_text(x) for x in c] if _is_number(t)) >= 2
        )
        if data_rows / len(region) >= 0.4:
            regions.append((start, len(cell_rows)))

    return regions


def _is_number(s: str) -> bool:
    s = s.strip()
    if not s:
        return False
    return s.replace(".", "").replace("-", "").replace("+", "").isdigit()


# ---------------------------------------------------------------------------
# 第 4 步：对齐成网格
# ---------------------------------------------------------------------------

def _build_grid(
    cell_rows: list[list[list[dict]]],
    raw_lines: list[list[dict]],
) -> dict | None:
    """把表区域对齐成矩形网格。空单元格 = None。

    用表头行（列数最多的行）确定列边界，
    每个数据单元格匹配最近的列边界。
    """
    if len(cell_rows) < 2:
        return None

    # 找表头行：列数最多的行
    best_idx = max(range(len(cell_rows)), key=lambda i: len(cell_rows[i]))
    header_cells = cell_rows[best_idx]
    n_cols = len(header_cells)
    if n_cols < 2:
        return None

    # 每列的中心 x 坐标
    col_centers = [
        sum((w["x0"] + w["x1"]) / 2 for w in cell) / len(cell)
        for cell in header_cells
    ]

    # 表头文本
    header = [_cell_text(c) for c in header_cells]

    # 数据行
    rows: list[list[str | None]] = []
    seen_header = False
    for i, (cells, raw) in enumerate(zip(cell_rows, raw_lines)):
        # 跳过表头行本身
        if i == best_idx:
            seen_header = True
            continue

        row: list[str | None] = []
        used = set()
        for center in col_centers:
            best = None
            best_dist = 30.0
            for wi, w in enumerate(raw):
                if wi in used:
                    continue
                wx = (w["x0"] + w["x1"]) / 2
                dist = abs(wx - center)
                if dist < best_dist:
                    best = w
                    best_dist = dist
            if best is not None:
                used.add(raw.index(best))
                row.append(best["text"].strip())
            else:
                row.append(None)

        if any(v is not None for v in row):
            rows.append(row)

    # 如果 header 是数据行（数字为主），把它也当数据行
    header_is_data = sum(1 for h in header if _is_number(h)) >= len(header) * 0.5
    if header_is_data:
        rows.insert(0, header)
        # 用下一行当表头，或留空
        for idx in range(len(cell_rows)):
            if idx != best_idx:
                h_text = [_cell_text(c) for c in cell_rows[idx]]
                if sum(1 for h in h_text if _is_number(h)) < len(h_text) * 0.5:
                    header = h_text
                    break

    return {"header": header, "rows": rows, "columns": n_cols}
