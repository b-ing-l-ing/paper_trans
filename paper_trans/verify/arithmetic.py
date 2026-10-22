"""硬算术重算 —— 从表格里用代码算增量/比例，不靠 LLM 心算。

针对「组标签 + 数值列」结构的表格（如 C0/C1/C2… 组），计算各组
相对基准组的增量与比例，生成事实摘要，喂给一致性审计 agent 做语义对齐。
"""

import logging
import re

from ..schemas import PaperData, TableData

logger = logging.getLogger(__name__)

_GROUP_RE = re.compile(r"^[A-Z]{1,2}\d+$")  # A1 / C0 / C2 / B3 ...（排除 SiO2 等化学式）


def build_table_facts(data: PaperData) -> str:
    """从提取的表格硬算增量/比例，返回事实摘要文本（可能为空）。"""
    lines: list[str] = []
    for t in data.tables:
        try:
            _process_table(t, lines)
        except Exception as e:
            logger.debug("表格 %s 硬算失败: %s", t.table_id, e)
    if not lines:
        return ""
    return "## 表格硬算事实（程序已从表格计算，勿重新心算）\n" + "\n".join(lines)


def _process_table(t: TableData, lines: list[str]) -> None:
    headers = t.headers or []
    rows = t.rows or []
    if not headers or len(rows) < 2:
        return
    if _looks_row_wise(rows):
        _row_wise(t, headers, rows, lines)
    elif _looks_col_wise(headers):
        _col_wise(t, headers, rows, lines)


def _looks_row_wise(rows: list[list[str]]) -> bool:
    """第一列是否都是组标签（A1/C0/C2…）。"""
    return sum(1 for r in rows if r and _GROUP_RE.match(str(r[0]).strip())) >= 2


def _looks_col_wise(headers: list[str]) -> bool:
    """列标题是否形如组标签（C0/C2/A1…）。"""
    return sum(1 for h in headers[1:] if _GROUP_RE.match(str(h).strip())) >= 2


def _to_float(s: str) -> float | None:
    try:
        return float(str(s).strip().replace(",", ""))
    except (ValueError, TypeError):
        return None


def _fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def _find_base_row(rows: list[list[str]]) -> int:
    """基准行：优先标签以 0 结尾（如 C0 对照组），其次以 1 结尾（如 A1），否则第 0 行。"""
    for i, r in enumerate(rows):
        label = str(r[0]).strip() if r else ""
        if label.endswith("0"):
            return i
    for i, r in enumerate(rows):
        label = str(r[0]).strip() if r else ""
        if label.endswith("1"):
            return i
    return 0


def _row_wise(t: TableData, headers: list[str], rows: list[list[str]], lines: list[str]) -> None:
    base_idx = _find_base_row(rows)
    base_vals = rows[base_idx]
    base_label = str(base_vals[0]).strip()
    tid = t.table_id or "Table"

    for col in range(1, len(headers)):
        col_name = str(headers[col]).strip()
        base_f = _to_float(base_vals[col]) if col < len(base_vals) else None
        if base_f is None:
            continue
        parts = [f"- {tid}「{col_name}」基准 {base_label}={_fmt(base_f)}"]
        for r in rows:
            label = str(r[0]).strip()
            if label == base_label:
                continue
            v = _to_float(r[col]) if col < len(r) else None
            if v is None:
                continue
            seg = f"{label}={_fmt(v)}"
            if v != base_f:
                seg += f" (Δ{_fmt(v - base_f)})"
            if base_f:
                seg += f" (×{_fmt(v / base_f)})"
            parts.append(seg)
        lines.append("; ".join(parts))


def _col_wise(t: TableData, headers: list[str], rows: list[list[str]], lines: list[str]) -> None:
    # 基准列：标题以 0 结尾的列，否则第 1 列
    base_col = 1
    for c in range(1, len(headers)):
        if str(headers[c]).strip().endswith("0"):
            base_col = c
            break
    base_label = str(headers[base_col]).strip()
    tid = t.table_id or "Table"

    for r in rows:
        if not r:
            continue
        row_label = str(r[0]).strip()
        base_f = _to_float(r[base_col]) if base_col < len(r) else None
        if base_f is None:
            continue
        parts = [f"- {tid}「{row_label}」基准 {base_label}={_fmt(base_f)}"]
        for c in range(1, len(headers)):
            if c == base_col:
                continue
            v = _to_float(r[c]) if c < len(r) else None
            if v is None:
                continue
            parts.append(f"{str(headers[c]).strip()}={_fmt(v)} (Δ{_fmt(v - base_f)})")
        lines.append("; ".join(parts))
