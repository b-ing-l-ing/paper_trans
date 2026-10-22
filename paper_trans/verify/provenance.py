"""证据溯源 —— 把提取目标定位回原文页码与证据句（零 LLM）。

策略：优先用 caption / 表格标题等独特性强的文本定位，数值则用其数字部分匹配。
命中结果标注为「首个匹配」，不保证唯一，仅供人工快速跳转原文。
"""

import logging
import re

from ..schemas import PaperData
from .models import ProvenanceEntry

logger = logging.getLogger(__name__)

_SNIPPET_RADIUS = 80  # 证据句上下文半径（字符）


def build_provenance(
    data: PaperData,
    pages_text: dict[int, str],
) -> list[ProvenanceEntry]:
    """为每个表格 / 图表 / 数值定位原文位置。"""
    entries: list[ProvenanceEntry] = []

    for t in data.tables:
        target = t.table_id or "unnamed table"
        page, snippet = _locate_by_candidates(
            _id_candidates(t.table_id) + [t.caption], pages_text
        )
        entries.append(
            ProvenanceEntry(
                target=f"table:{target}",
                kind="table",
                page=page,
                found=page is not None,
                evidence_text=snippet,
            )
        )

    for f in data.figures:
        target = f.figure_id or "unnamed figure"
        page, snippet = _locate_by_candidates(
            _id_candidates(f.figure_id) + [f.caption], pages_text
        )
        entries.append(
            ProvenanceEntry(
                target=f"figure:{target}",
                kind="figure",
                page=page,
                found=page is not None,
                evidence_text=snippet,
            )
        )

    for i, nd in enumerate(data.numerical_data):
        query = _first_number(nd.value)
        page, snippet = _locate(query, pages_text) if query else (None, None)
        entries.append(
            ProvenanceEntry(
                target=f"numerical:{nd.name}",
                kind="numerical",
                page=page,
                found=page is not None,
                evidence_text=snippet,
            )
        )

    return entries


def _locate_by_candidates(
    candidates: list[str | None],
    pages_text: dict[int, str],
) -> tuple[int | None, str | None]:
    """依次尝试多个候选文本，返回首个命中的 (页码, 证据片段)。"""
    for cand in candidates:
        if not cand:
            continue
        # caption 取前 50 字符作为查询，避免整段太长匹配失败
        query = cand.strip().replace("\n", " ")[:50]
        page, snippet = _locate(query, pages_text)
        if page is not None:
            return page, snippet
    return None, None


def _first_number(value: str) -> str | None:
    """提取数值字符串中的第一个数字 token，如 '3.2 ± 0.1' -> '3.2'。"""
    m = re.search(r"\d+(?:\.\d+)?", value)
    return m.group(0) if m else None


def _id_candidates(id_text: str | None) -> list[str]:
    """把英文表格/图 id 展开成中英候选，提高中文论文命中率。"""
    if not id_text:
        return []
    m = re.match(
        r"^(Table|Figure|Fig\.?)\s*(\d+)$", id_text.strip(), re.IGNORECASE
    )
    if not m:
        return [id_text]
    kind, num = m.group(1), m.group(2)
    if kind.lower().startswith("table"):
        return [f"Table {num}", f"Table{num}", f"表{num}", f"表 {num}"]
    return [
        f"Figure {num}", f"Fig. {num}", f"Fig.{num}",
        f"Fig {num}", f"Fig{num}", f"图{num}", f"图 {num}",
    ]


_NOISE_RES = [
    re.compile(r"文章编号"),
    re.compile(r"年第?期?卷?\s*\d+"),
]


def _is_noise(snippet: str) -> bool:
    """判断证据片段是否为期刊页眉/卷期噪声。"""
    if not snippet:
        return False
    return any(pat.search(snippet) for pat in _NOISE_RES)


def _locate(
    query: str, pages_text: dict[int, str]
) -> tuple[int | None, str | None]:
    """在分页文本中查找 query，返回首个非噪声命中的 (页码, 证据片段)。"""
    q = query.strip()
    if not q:
        return None, None
    for page in sorted(pages_text):
        text = pages_text[page]
        idx = 0
        while True:
            idx = text.find(q, idx)
            if idx < 0:
                break
            snippet = _snippet(text, idx, len(q))
            if not _is_noise(snippet):
                return page, snippet
            idx += len(q)
    return None, None


def _snippet(text: str, idx: int, qlen: int) -> str:
    """截取匹配位置附近的上下文片段。"""
    start = max(0, idx - _SNIPPET_RADIUS)
    end = min(len(text), idx + qlen + _SNIPPET_RADIUS)
    return text[start:end].strip()
