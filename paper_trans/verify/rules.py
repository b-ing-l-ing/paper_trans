"""规则化一致性校验 —— 零 LLM 成本的自洽检查。

只依据提取结果自身的结构做检查，不调用模型，速度极快。
"""

import logging

from ..schemas import PaperData
from .models import Issue

logger = logging.getLogger(__name__)


def run_rule_checks(data: PaperData) -> list[Issue]:
    """对提取结果做规则化自洽检查，返回问题列表。"""
    issues: list[Issue] = []
    _check_table_alignment(data, issues)
    _check_duplicate_ids(data, issues)
    _check_numerical_conflicts(data, issues)
    return issues


def _check_table_alignment(data: PaperData, issues: list[Issue]) -> None:
    """检查表格每一行的列数是否与表头一致。"""
    for i, t in enumerate(data.tables):
        loc = t.table_id or f"tables[{i}]"
        ncols = len(t.headers)
        if ncols == 0:
            issues.append(
                Issue(
                    id=f"rule-table-empty-header-{i}",
                    severity="warning",
                    category="schema",
                    location=loc,
                    description="表格表头为空，无法校验列对齐。",
                    suggestion="核对原文，补全表头。",
                )
            )
            continue
        for r, row in enumerate(t.rows):
            if len(row) != ncols:
                issues.append(
                    Issue(
                        id=f"rule-table-align-{i}-{r}",
                        severity="warning",
                        category="schema",
                        location=loc,
                        description=(
                            f"第 {r + 1} 行有 {len(row)} 个单元格，"
                            f"但表头有 {ncols} 列；可能列错位或漏掉空单元格。"
                        ),
                        suggestion="核对原文，空单元格用空字符串占位补齐列。",
                    )
                )


def _check_duplicate_ids(data: PaperData, issues: list[Issue]) -> None:
    """检查表格 / 图表的 id 是否重复。"""
    for kind, items, label in (
        ("table", data.tables, "表格"),
        ("figure", data.figures, "图表"),
    ):
        seen: dict[str, int] = {}
        for i, item in enumerate(items):
            item_id = item.table_id if kind == "table" else item.figure_id
            if not item_id:
                continue
            if item_id in seen:
                issues.append(
                    Issue(
                        id=f"rule-dup-{kind}-{item_id}",
                        severity="warning",
                        category="schema",
                        location=item_id,
                        description=f"{label} id 重复：{item_id} 出现多次。",
                        suggestion="核对原文，可能为子图或重复提取。",
                    )
                )
            else:
                seen[item_id] = i


def _check_numerical_conflicts(data: PaperData, issues: list[Issue]) -> None:
    """检查同名数值指标是否出现多个不同取值。"""
    by_name: dict[str, list[tuple[int, str]]] = {}
    for i, nd in enumerate(data.numerical_data):
        by_name.setdefault(nd.name, []).append((i, nd.value))

    for name, entries in by_name.items():
        if len(entries) <= 1:
            continue
        values = {v for _, v in entries}
        if len(values) > 1:
            issues.append(
                Issue(
                    id=f"rule-num-conflict-{name[:20]}",
                    severity="info",
                    category="inconsistent",
                    location=f"numerical_data[{entries[0][0]}]",
                    description=(
                        f"同名指标 '{name}' 出现 {len(values)} 个不同取值，"
                        "可能对应不同实验条件，建议人工核对。"
                    ),
                )
            )
