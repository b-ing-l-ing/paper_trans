"""Verification 管线 —— 把规则校验、证据溯源、交叉验证串成一次报告。"""

import json
import logging
import re
from pathlib import Path

from ..schemas import PaperData
from .models import VerificationReport, Issue
from .rules import run_rule_checks
from .provenance import build_provenance
from .crosscheck import run_crosscheck

logger = logging.getLogger(__name__)


def verify_paper(
    text: str,
    data: PaperData,
    pages_text: dict[int, str] | None = None,
    tables: list[dict] | None = None,
    model: str | None = None,
    do_crosscheck: bool = True,
) -> VerificationReport:
    """对一次提取结果运行完整验证管线。

    Args:
        text: 论文全文。
        data: 已提取的结构化数据。
        pages_text: {页码: 文本}，证据溯源需要。
        tables: 结构化表格（detect_tables 输出），可选。
        model: LLM 模型名称。
        do_crosscheck: 是否运行多 agent 交叉验证（成本较高）。

    Returns:
        VerificationReport 实例。
    """
    pages_text = pages_text or {}

    rules = run_rule_checks(data)
    provenance = build_provenance(data, pages_text)
    crosscheck = (
        run_crosscheck(text, data, tables=tables, model=model)
        if do_crosscheck
        else []
    )

    summary = _summarize(rules, crosscheck, provenance)
    return VerificationReport(
        paper_title=data.paper_title,
        rules=rules,
        crosscheck=crosscheck,
        provenance=provenance,
        summary=summary,
    )


def _summarize(
    rules: list[Issue],
    crosscheck: list[Issue],
    provenance: list,
) -> str:
    all_issues = rules + crosscheck
    n_err = sum(1 for i in all_issues if i.severity == "error")
    n_warn = sum(1 for i in all_issues if i.severity == "warning")
    n_info = sum(1 for i in all_issues if i.severity == "info")
    n_found = sum(1 for p in provenance if p.found)
    return (
        f"规则校验 {len(rules)} 条，交叉验证 {len(crosscheck)} 条"
        f"（error {n_err}、warning {n_warn}、info {n_info}）。"
        f"证据溯源命中 {n_found}/{len(provenance)}。"
    )


# ---------------------------------------------------------------------------
# 报告渲染与保存
# ---------------------------------------------------------------------------

_SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


def render_report(report: VerificationReport) -> str:
    """把验证报告渲染为 Markdown。"""
    md: list[str] = []
    md.append(f"# 验证报告：{report.paper_title or 'Untitled'}\n")
    md.append(f"> {report.summary}\n")

    md.append("## 发现问题\n")
    issues = sorted(
        report.rules + report.crosscheck,
        key=lambda i: _SEVERITY_ORDER.get(i.severity, 3),
    )
    if not issues:
        md.append("未发现问题。\n")
    else:
        for issue in issues:
            md.append(f"### [{issue.severity.upper()}] {issue.description}")
            if issue.location:
                md.append(f"- **位置：** {issue.location}")
            if issue.evidence:
                md.append(f"- **证据：** {issue.evidence}")
            if issue.suggestion:
                md.append(f"- **建议：** {issue.suggestion}")
            if issue.source:
                md.append(f"- **来源：** {issue.source}")
            md.append("")

    md.append("## 证据溯源\n")
    md.append("| 目标 | 类型 | 页码 | 命中 | 证据 |")
    md.append("|------|------|------|------|------|")
    for p in report.provenance:
        md.append(
            f"| {_esc(p.target)} | {p.kind} | {p.page or '-'} "
            f"| {'✓' if p.found else '✗'} | {_esc(p.evidence_text or '-')} |"
        )
    md.append("")

    return "\n".join(md)


def save_report(
    report: VerificationReport,
    output_dir: str | Path = "./output",
    base_name: str | None = None,
) -> tuple[str, str]:
    """保存验证报告（Markdown + JSON），返回文件路径。"""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    name = base_name or _sanitize(report.paper_title or "verification")
    md_path = out / f"{name}.verify.md"
    json_path = out / f"{name}.verify.json"

    md_path.write_text(render_report(report), encoding="utf-8")
    json_path.write_text(
        json.dumps(report.model_dump(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    logger.info("验证报告已保存：%s / %s", md_path, json_path)
    return str(md_path), str(json_path)


def _esc(s: str) -> str:
    return s.replace("|", "\\|").replace("\n", " ")


def _sanitize(name: str, max_len: int = 80) -> str:
    clean = re.sub(r"[^\w\s-]", "", name)[:max_len]
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean.strip("_") or "verification"
