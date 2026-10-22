"""规则校验 —— 零 LLM 的物理检查（取 JMI 综述「physical checks」精华）。

对数据集行做硬约束检查，不调模型、不走网络。
"""

from __future__ import annotations

from typing import Any

from ..schema import Schema


def check_records(
    records: list[dict[str, Any]],
    schema: Schema,
    ratio_tolerance: float = 5.0,
) -> list[dict]:
    """对一批数据集行做规则校验，返回问题列表。

    每条问题：{record_index, field, rule, message, value}。
    """
    issues: list[dict] = []
    for i, record in enumerate(records):
        _check_ratio_sum(i, record, schema, ratio_tolerance, issues)
        _check_value_range(i, record, schema, issues)
        _check_molar_ratio(i, record, schema, issues)
    return issues


def _check_ratio_sum(i, record, schema, tolerance, issues):
    """填了的原料字段（category=原料，非 null）的比例和若明显超 100，报抄错。

    只抓「超 100」：比例和不可能超过 100%，超了必是抄错；
    不抓「不足 100」：可能是配方含字段库外的原料（如 WS），属正常。
    """
    ratios = [
        float(record[f.name])
        for f in schema.fields
        if f.category == "原料" and not f.computed
        and isinstance(record.get(f.name), (int, float))
    ]
    if not ratios:
        return
    total = sum(ratios)
    if total > 100 + tolerance:
        issues.append({
            "record_index": i,
            "field": None,
            "rule": "ratio_sum",
            "message": f"原料比例和 = {total:.1f}，超过 100，可能抄错",
            "value": total,
        })


def _check_value_range(i, record, schema, issues):
    """值域检查：wt% 字段应 0~100，性能字段应 ≥ 0。"""
    for f in schema.fields:
        v = record.get(f.name)
        if not isinstance(v, (int, float)):
            continue
        if f.unit == "wt%":
            if v < 0 or v > 100:
                issues.append({
                    "record_index": i, "field": f.name, "rule": "value_range",
                    "message": f"{f.name} = {v} 超出 0~100", "value": v,
                })
        elif f.category == "性能" and v < 0:
            issues.append({
                "record_index": i, "field": f.name, "rule": "value_range",
                "message": f"{f.name} = {v} 为负值", "value": v,
            })


def _check_molar_ratio(i, record, schema, issues):
    """摩尔比（Ca/Si、Si/Al、Na/Al）应在 0~10 之间。"""
    for name in ("Ca/Si", "Si/Al", "Na/Al"):
        if name not in record:
            continue
        v = record[name]
        if not isinstance(v, (int, float)):
            continue
        if v < 0 or v > 10:
            issues.append({
                "record_index": i, "field": name, "rule": "molar_ratio",
                "message": f"{name} = {v} 超出 0~10", "value": v,
            })
