"""数据溯源 —— 把每个抽出来的值回溯到原文的页码与证据句（**零 LLM**）。

要回答的唯一问题：**这个值来自原文哪一页、哪一句**。
`found=False` 本身就是**幻觉信号** —— 一个值在全文一处都找不到，它要么是算出来的、
要么是编的。这份数据要喂 ML 训练，静默的错标签比漏数据危害大得多。

三个证据源，都带页码：
  · **正文**（`parse_pdf` 返回的 `pages_text`）
  · **结构化表格**（`detect_tables` 的输出，每张有 page）
  · **图文本**（`figure_to_text` 的结果，按图所在页归位）

匹配纯确定性：字符串查找 + 正则 + 数字边界，**不调任何模型、可复现、可单测**。

⚠️ **必须在 `normalize_record` 之前调用**：那一层会把 `"4000 kPa"` 变成 `4000.0`，
原始串和单位都没了，就再也回不去原文查。

⚠️ 本版**不做**单位等价换算（`4 MPa` ↔ `4000 kPa`）：`prompts.py` 已明确要求 LLM
「不要自己换算」，真出现换写的值会落进 `absent` —— **那正是后续单位校验要抓的问题**，
在这里顺手掩盖掉反而不好。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Iterator

from ..compute import split_number_unit
from ..schema import Schema

logger = logging.getLogger(__name__)

_SNIPPET_RADIUS = 80        # 证据句上下文半径（字符）
_MAX_CELL_LEN = 40          # 表格证据里单元格的最大长度
_MAX_HITS = 5               # 同一个值最多记几处出现

# 这些「值」等于没填，不生成候选、不进分母
_EMPTY_VALUES = {"", "-", "—", "–", "n/a", "na", "none", "null", "未提及", "未报道", "无"}

# 期刊页眉 / 元数据噪声：命中这些片段的位置不算证据
_NOISE_RES = [
    re.compile(r"文章编号"),
    re.compile(r"DOI\s*[:：]?\s*10\.", re.IGNORECASE),
    re.compile(r"收稿日期"),
    re.compile(r"基金项目"),
    re.compile(r"作者简介"),
    re.compile(r"通信作者"),
    re.compile(r"引用格式"),
    re.compile(r"中图分类号"),
    re.compile(r"文献标志码"),
    re.compile(r"第\s*\d+\s*卷\s*第\s*\d+\s*期"),
    re.compile(r"Vol\.?\s*\d+.*No\.?\s*\d+", re.IGNORECASE),
    re.compile(r"ISSN\s*\d{4}", re.IGNORECASE),
]


# ---------------------------------------------------------------------------
# 结果结构
# ---------------------------------------------------------------------------


@dataclass
class ProvenanceEntry:
    """一条溯源记录：某个 record 的某个字段值，来自哪里。"""

    record_index: int
    field: str
    field_label: str
    value: str                        # 原文串（normalize 前），如 "4000 kPa"
    status: str = "absent"            # located_text / located_table / located_figure / ambiguous / absent / empty / computed
    source: str = "none"              # text / table / figure / none
    page: int | None = None
    evidence: str | None = None
    match_kind: str = "none"          # exact / compact / number / none
    found: bool = False
    note: str | None = None

    def to_row(self) -> dict[str, Any]:
        return {
            "record_index": self.record_index,
            "field": self.field,
            "field_label": self.field_label,
            "value": self.value,
            "page": self.page if self.page is not None else "",
            "evidence": self.evidence or "",
            "status": self.status,
            "source": self.source,
            "match_kind": self.match_kind,
            "found": 1 if self.found else 0,
            "note": self.note or "",
        }


@dataclass
class ProvenanceReport:
    entries: list[ProvenanceEntry] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


@dataclass
class _FieldCtx:
    """一个字段的上下文（用于生成候选、判断「是否在说同一件事」）。"""

    name: str
    label: str
    computed: bool = False
    aliases: list[str] = field(default_factory=list)

    @property
    def anchors(self) -> list[str]:
        """字段的「锚点词」——弱候选（裸数字）命中时，附近必须出现其中之一。"""
        out = [self.label, self.name, *self.aliases]
        return [a for a in out if a and len(a) >= 2]


# ---------------------------------------------------------------------------
# 文本工具
# ---------------------------------------------------------------------------


def _compact_with_map(text: str) -> tuple[str, list[int]]:
    """去掉所有空白，并给出「压缩后下标 → 原文下标」的映射。"""
    chars: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if not ch.isspace():
            chars.append(ch)
            idx.append(i)
    return "".join(chars), idx


def _compact(s: str) -> str:
    return "".join(ch for ch in str(s) if not ch.isspace())


def _norm_ws(s: str) -> str:
    return re.sub(r"\s+", " ", str(s)).strip()


def _is_noise(snippet: str) -> bool:
    """证据片段是不是期刊页眉 / 卷期 / 元数据这类噪声。"""
    if not snippet:
        return False
    return any(pat.search(snippet) for pat in _NOISE_RES)


def _snippet(text: str, idx: int, qlen: int, radius: int = _SNIPPET_RADIUS) -> str:
    """截取匹配位置附近的上下文，并尽量向外扩到句子边界。"""
    start = max(0, idx - radius)
    end = min(len(text), idx + qlen + radius)
    # 扩到最近的句末标点（最多再扩 radius 个字符）
    for brk in ("。", "！", "？", "；", "\n"):
        p = text.rfind(brk, max(0, start - radius), start)
        if p >= 0:
            start = p + 1
            break
    for brk in ("。", "！", "？", "；", "\n"):
        p = text.find(brk, end, min(len(text), end + radius))
        if p >= 0:
            end = p + 1
            break
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _has_anchor(snippet: str, anchors: Iterable[str]) -> bool:
    """片段里有没有字段锚点词（用来判断裸数字是不是在说同一个字段）。"""
    s = _compact(snippet)
    return any(_compact(a) in s for a in anchors)


# ---------------------------------------------------------------------------
# 候选生成
# ---------------------------------------------------------------------------


def _value_candidates(raw: str) -> list[tuple[str, str]]:
    """把原文值拆成 (候选串, 类型) 列表，按强度降序。

    类型：`exact`（原串）/ `number`（裸数字，弱候选）。
    去空白匹配是统一走的（所有候选都在压缩文本里找），所以不单列一种。
    """
    text = _norm_ws(raw)
    if not text:
        return []
    cands: list[tuple[str, str]] = [(text, "exact")]
    # 长串里带单位的（"4000 kPa"）——再补一个裸数字做兜底
    num, _unit = split_number_unit(text)
    if num is not None:
        n = _format_number(num)
        if n and n != text:
            cands.append((n, "number"))
    return cands


def _format_number(v: float) -> str:
    """把一个数格式化成正文里最可能的样子（去掉无意义的 .0）。"""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return repr(v)


# ---------------------------------------------------------------------------
# 搜索
# ---------------------------------------------------------------------------


@dataclass
class _Hit:
    page: int
    source: str
    evidence: str
    kind: str


def _boundary_ok(s: str, start: int, end: int) -> bool:
    """裸数字命中的前后必须不是数字/小数点，避免 `45` 命中 `450`、`3.2` 命中 `3.25`。

    ⚠️ 必须对着**原文**（带空格）判，不能对着压缩文本判：
    表格逐格拼起来是 `16.911.675`（空格没了），`16.91` 后面紧跟数字，
    在压缩文本上判会把**真命中**误杀（实测 98 个值就是这么被判成"查无出处"的）。
    """
    before = s[start - 1] if start > 0 else ""
    after = s[end] if end < len(s) else ""
    return not (before.isdigit() or before == "." or after.isdigit() or after == ".")


def _iter_raw_spans(s: str, query: str) -> Iterator[tuple[int, int]]:
    """在 s 里找 query（空白不敏感），产出原文里的 (start, end) 区间。"""
    qc = _compact(query)
    if not qc:
        return
    ctext, cmap = _compact_with_map(s)
    i = ctext.find(qc)
    while i >= 0:
        start = cmap[i]
        last = min(i + len(qc) - 1, len(cmap) - 1)
        yield start, cmap[last] + 1
        i = ctext.find(qc, i + 1)


def _find_in_text(
    pages_text: dict[int, str], query: str, kind: str, radius: int,
    anchors: list[str] | None = None, require_anchor: bool = False,
) -> _Hit | None:
    """在正文里找 query，跳过噪声位置。require_anchor=True 时片段里必须有锚点词。"""
    for page in sorted(pages_text):
        text = pages_text[page]
        for start, end in _iter_raw_spans(text, query):
            if kind == "number" and not _boundary_ok(text, start, end):
                continue
            snip = _snippet(text, start, end - start, radius)
            if not _is_noise(snip) and (
                not require_anchor or _has_anchor(snip, anchors or [])
            ):
                return _Hit(page=page, source="text", evidence=snip, kind=kind)
    return None


def _find_in_tables(
    tables: list[dict], query: str, kind: str,
) -> _Hit | None:
    """在结构化表格的单元格里找 query。表格自带页码，命中就是强证据。

    **逐格匹配**，不是把整行拼起来匹配——拼起来会把相邻格粘成一个长数字串。
    """
    for ti, tab in enumerate(tables):
        page = tab.get("page")
        header = [str(h) for h in (tab.get("header") or [])]
        for ri, row in enumerate(tab.get("rows") or []):
            cells = ["" if c is None else str(c) for c in row]
            for cell in cells:
                if not cell:
                    continue
                for start, end in _iter_raw_spans(cell, query):
                    if kind == "number" and not _boundary_ok(cell, start, end):
                        continue
                    ev = _table_evidence(ti, page, ri, header, cells)
                    return _Hit(page=page, source="table", evidence=ev, kind=kind)
    return None


def _table_evidence(ti: int, page, ri: int, header: list[str], cells: list[str]) -> str:
    def clip(s: str) -> str:
        s = re.sub(r"\s+", " ", s).strip()
        return s if len(s) <= _MAX_CELL_LEN else s[: _MAX_CELL_LEN - 1] + "…"

    head = "/".join(clip(h) for h in header) or "—"
    line = "/".join(clip(c) for c in cells) or "—"
    return f"表{ti + 1}（第{page}页）第{ri + 1}行 | 表头: {head} | 本行: {line}"


def _find_in_figures(figure_texts: list[dict], query: str, kind: str) -> _Hit | None:
    """在图文本里找。图里反算出来的值不会出现在正文，这里是它们唯一的出处。"""
    q = _compact(query)
    if not q:
        return None
    for fig in figure_texts:
        text = fig.get("text") or ""
        for line in text.splitlines():
            for start, end in _iter_raw_spans(line, query):
                if kind == "number" and not _boundary_ok(line, start, end):
                    continue
                page = fig.get("page")
                ev = re.sub(r"\s+", " ", line).strip()
                if fig.get("filename"):
                    ev = f"{ev} [图: {fig['filename']}]"
                return _Hit(page=page, source="figure", evidence=ev, kind=kind)
    return None


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------


def _field_contexts(schema: Schema, new_fields: list | None) -> dict[str, _FieldCtx]:
    ctxs: dict[str, _FieldCtx] = {}
    for f in schema.fields:
        ctxs[f.name] = _FieldCtx(
            name=f.name,
            label=getattr(f, "label", "") or f.name,
            computed=bool(getattr(f, "computed", False)),
            aliases=list(getattr(f, "aliases", None) or []),
        )
    for nf in new_fields or []:
        name = getattr(nf, "name", None)
        if not name or name in ctxs:
            continue
        ctxs[name] = _FieldCtx(
            name=name,
            label=getattr(nf, "label", "") or name,
            computed=False,
        )
    return ctxs


def _is_empty(raw: Any) -> bool:
    if raw is None:
        return True
    s = str(raw).strip().lower()
    return s in _EMPTY_VALUES


def _locate_one(
    raw: str,
    ctx: _FieldCtx,
    pages_text: dict[int, str],
    tables: list[dict],
    figure_texts: list[dict],
    radius: int,
) -> ProvenanceEntry | None:
    """给一个值找出处。找不到返回 None（调用方记 absent）。"""
    for cand, kind in _value_candidates(raw):
        if kind == "exact":
            # 强候选：原串（含单位）——正文 → 表格 → 图
            hit = _find_in_text(pages_text, cand, kind, radius)
            if hit:
                return _entry_from_hit(hit, "located_text")
            hit = _find_in_tables(tables, cand, kind)
            if hit:
                return _entry_from_hit(hit, "located_table")
            hit = _find_in_figures(figure_texts, cand, kind)
            if hit:
                return _entry_from_hit(hit, "located_figure")
        else:
            # 弱候选：裸数字——表格优先（有行上下文），正文必须带字段锚点
            hit = _find_in_tables(tables, cand, kind)
            if hit:
                return _entry_from_hit(hit, "located_table")
            hit = _find_in_text(
                pages_text, cand, kind, radius,
                anchors=ctx.anchors, require_anchor=True,
            )
            if hit:
                return _entry_from_hit(hit, "located_text")
            hit = _find_in_figures(figure_texts, cand, kind)
            if hit:
                return _entry_from_hit(hit, "located_figure")
            # 数字确实出现在正文里、但附近没有字段锚点 → 有歧义（可能是别的字段的数字）
            loose = _find_in_text(pages_text, cand, kind, radius)
            if loose:
                e = _entry_from_hit(loose, "ambiguous")
                e.note = "数字在原文出现，但附近找不到字段名，无法确认是同一个量"
                return e
    return None


def _entry_from_hit(hit: _Hit, status: str) -> ProvenanceEntry:
    return ProvenanceEntry(
        record_index=-1,          # 由调用方填
        field="",
        field_label="",
        value="",
        status=status,
        source=hit.source,
        page=hit.page,
        evidence=hit.evidence,
        match_kind=hit.kind,
        found=status.startswith("located"),
    )


def build_provenance(
    records: list[dict[str, Any]],
    schema: Schema,
    pages_text: dict[int, str],
    *,
    tables: list[dict] | None = None,
    figure_texts: list[dict] | None = None,
    new_fields: list | None = None,
    snippet_radius: int = _SNIPPET_RADIUS,
) -> ProvenanceReport:
    """给每条记录、每个非空非计算字段，回溯它在原文里的出处。

    `records` 必须是 **normalize 之前**的原始 records（值形如 "4000 kPa"）。
    `figure_texts` 形如 `[{"page": 5, "filename": "...", "text": "图(第5页) ..."}]`。

    返回值只用于报告，**不修改 records**。
    """
    tables = tables or []
    figure_texts = figure_texts or []
    ctxs = _field_contexts(schema, new_fields)

    entries: list[ProvenanceEntry] = []
    for ri, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        for name, ctx in ctxs.items():
            if ctx.computed:
                continue
            if name not in record:
                continue
            raw = record.get(name)
            if _is_empty(raw):
                entries.append(ProvenanceEntry(
                    record_index=ri, field=name, field_label=ctx.label,
                    value="" if raw is None else str(raw),
                    status="empty", found=False,
                ))
                continue
            text = str(raw).strip()
            e = _locate_one(text, ctx, pages_text, tables, figure_texts, snippet_radius)
            if e is None:
                e = ProvenanceEntry(
                    record_index=ri, field=name, field_label=ctx.label, value=text,
                    status="absent", found=False, match_kind="none",
                )
            e.record_index = ri
            e.field = name
            e.field_label = ctx.label
            e.value = text
            entries.append(e)

    report = ProvenanceReport(entries=entries)
    report.summary = summarize_provenance(report, schema, new_fields)
    return report


def summarize_provenance(
    report: ProvenanceReport,
    schema: Schema | None = None,
    new_fields: list | None = None,
) -> dict[str, Any]:
    """统计「查无出处」的比例。

    **分母** = 非空、非 computed 的单元格数（empty / computed 都不进分母）。
    **分子** = ambiguous + absent —— 两者分开列，不混着报。
    """
    by_status: dict[str, int] = {}
    by_source: dict[str, int] = {}
    per_field: dict[str, list[int]] = {}   # field -> [总数, 查无出处数]

    for e in report.entries:
        by_status[e.status] = by_status.get(e.status, 0) + 1
        if e.found:
            by_source[e.source] = by_source.get(e.source, 0) + 1
        if e.status in ("empty", "computed"):
            continue
        st = per_field.setdefault(e.field, [0, 0])
        st[0] += 1
        if e.status in ("ambiguous", "absent"):
            st[1] += 1

    denominator = sum(1 for e in report.entries if e.status not in ("empty", "computed"))
    not_found = by_status.get("ambiguous", 0) + by_status.get("absent", 0)
    located = denominator - not_found

    worst = sorted(
        ((f, c, n) for f, (c, n) in per_field.items() if n > 0),
        key=lambda t: (-t[2], t[0]),
    )[:5]

    return {
        "分母": denominator,
        "可溯源": located,
        "查无出处": not_found,
        "查无出处比例": round(not_found / denominator * 100, 1) if denominator else 0.0,
        "按状态": by_status,
        "按来源": by_source,
        "歧义": by_status.get("ambiguous", 0),
        "完全未命中": by_status.get("absent", 0),
        "查无出处最多的字段": [
            {"field": f, "count": n, "total": c, "ratio": round(n / c * 100, 1) if c else 0.0}
            for f, c, n in worst
        ],
        "已跳过_空值": by_status.get("empty", 0),
    }


def to_csv_rows(report: ProvenanceReport) -> list[dict[str, Any]]:
    """转成 CSV 行（只保留非空、非 computed 的单元格，避免表膨胀）。"""
    return [e.to_row() for e in report.entries if e.status not in ("empty", "computed")]
