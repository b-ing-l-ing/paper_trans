"""执行层 —— 数值归一化 + 计算字段硬算。

数值归一化：把 LLM 返回的「数值 + 单位」字符串（如 "4000 kPa"）
按字段库的目标单位换算成纯数值（如 4.0）。

计算字段硬算（computed）：按 schema 的 compute 配置，用成分库加权
计算 Ca/Si、MgO 等字段。
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .schema import Schema

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 单位换算表：同一量纲下，各单位 → 基准单位的因子
# ---------------------------------------------------------------------------

_LENGTH = {"nm": 1e-9, "um": 1e-6, "μm": 1e-6, "mm": 1e-3, "cm": 1e-2, "m": 1.0}
_PRESSURE = {"pa": 1.0, "kpa": 1e3, "mpa": 1e6, "gpa": 1e9}
_MASS = {"mg": 1e-6, "g": 1e-3, "kg": 1.0, "t": 1e3}
_TIME = {"s": 1.0, "min": 60.0, "h": 3600.0, "d": 86400.0}
_PERCENT = {"%": 0.01, "wt%": 0.01, "vol%": 0.01, "ppm": 1e-6}

UNIT_TABLE: dict[str, float] = {}
for _d in (_LENGTH, _PRESSURE, _MASS, _TIME, _PERCENT):
    UNIT_TABLE.update(_d)


# 匹配一个数字（含正负号、小数、科学计数法）
_NUM_RE = re.compile(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?")

# 单位里要清理掉的杂字符（± 不确定度符号、比较符、空白、逗号）
_UNIT_NOISE_RE = re.compile(r"[±~<>≤≥\s,，]+")


def _split_number_unit(raw: str) -> tuple[float | None, str | None]:
    """拆分「数值 + 单位」。

    例：'4000 kPa' -> (4000.0, 'kPa')；'3.2 ± 0.1 MPa' -> (3.2, 'MPa')；
    '1.72E-2' -> (0.0172, None)；'标准养护' -> (None, None)。

    值取第一个数字；单位 = 删掉所有数字后剩余的非数字字符（顺带清掉 ±）。
    """
    raw = str(raw).strip()
    m = _NUM_RE.search(raw)
    if not m:
        return None, None
    value = float(m.group())
    unit = _NUM_RE.sub("", raw)
    unit = _UNIT_NOISE_RE.sub("", unit)
    return value, unit or None


def convert(value: float, from_unit: str | None, to_unit: str | None) -> float:
    """把 value 从 from_unit 换算到 to_unit。

    同单位 / 无单位 / 未知单位（换算表查不到）都直接返回原值，保守不瞎换算。
    """
    if from_unit is None or to_unit is None:
        return value
    if from_unit.lower() == to_unit.lower():
        return value
    f = UNIT_TABLE.get(from_unit.lower())
    t = UNIT_TABLE.get(to_unit.lower())
    if f is None or t is None:
        return value
    return value * f / t


def normalize_record(
    record: dict[str, Any],
    schema: Schema,
    new_fields: list | None = None,
) -> dict[str, Any]:
    """把一行 records 归一化。

    规则：
    - 数值字段：拆出数值 + 单位，换算到字段库 / 新字段声明的目标单位。
    - 分类字段 / 解析不出数值的：原样保留。
    - 传入 new_fields 时，新字段也按其声明的 unit/type 归一化。
    """
    field_map = {f.name: f for f in schema.fields}
    if new_fields:
        for nf in new_fields:
            field_map.setdefault(nf.name, nf)
    out: dict[str, Any] = {}
    for key, raw in record.items():
        f = field_map.get(key)
        if f is None or f.type in ("categorical", "discrete"):
            out[key] = raw
            continue
        value, unit = _split_number_unit(str(raw))
        if value is None:
            out[key] = raw
            continue
        out[key] = convert(value, unit, f.unit)
    return out


# ---------------------------------------------------------------------------
# 计算字段硬算（computed）—— 通用计算类型 + 按 schema 配置调度
# ---------------------------------------------------------------------------

# 氧化物摩尔质量（g/mol）
_MOLAR_MASS = {
    "SiO2": 60.08, "CaO": 56.08, "Al2O3": 101.96, "Na2O": 61.98,
    "MgO": 40.30, "Fe2O3": 159.69, "K2O": 94.20, "SO3": 80.06,
}


def _compute_oxide_totals(
    ratios: dict[str, float],
    composition: dict[str, dict[str, float | None]],
) -> dict[str, float]:
    """配方加权求和：混合氧化物总量（百分比）。

    totals[oxide] = Σ(配比i × 成分i[oxide])，结果单位为百分比。
    """
    totals: dict[str, float] = {}
    for mat, ratio in ratios.items():
        comp = composition.get(mat)
        if not comp:
            continue
        for oxide, pct in comp.items():
            if pct is None:
                continue
            totals[oxide] = totals.get(oxide, 0.0) + ratio * pct / 100.0
    return totals


def _molar_ratio(
    totals: dict[str, float],
    num_oxide: str,
    den_oxide: str,
    num_factor: int = 1,
    den_factor: int = 1,
) -> float | None:
    """氧化物摩尔比。num_factor/den_factor 处理多原子（Al2O3 含 2 个 Al 等）。"""
    m_num = totals.get(num_oxide)
    m_den = totals.get(den_oxide)
    if m_num is None or m_den is None or m_den == 0:
        return None
    n_num = m_num / _MOLAR_MASS[num_oxide] * num_factor
    n_den = m_den / _MOLAR_MASS[den_oxide] * den_factor
    return n_num / n_den if n_den else None


def _to_percent(v: float) -> float:
    """把原料配比统一到百分数（0~100）。若 0 < v <= 1 视为小数比例，×100。"""
    return v * 100.0 if 0 < v <= 1 else v


def compute_computed_fields(
    record: dict[str, Any],
    schema: Schema,
    composition: dict[str, dict[str, float | None]],
) -> dict[str, Any]:
    """对一条 record 计算 computed 字段。

    算法由字段的 compute 配置决定（通用），不硬编码字段名。
    支持的 kind：
    - weighted_oxide：混合氧化物含量，参数 oxide
    - molar_ratio：摩尔比，参数 num / den（可选 num_factor / den_factor）

    Args:
        record: 归一化后的 record（原料字段已是纯数值，单位 wt%）。
        schema: 字段库（computed 字段带 compute 配置）。
        composition: 原料成分库，{原料名: {氧化物: 百分比}}。

    Returns:
        填好 computed 字段后的 record（原地修改并返回）。
    """
    # 1. 取出配方（原料字段的配比，强制归到百分数 wt%）
    ratios: dict[str, float] = {
        f.name: _to_percent(float(record[f.name]))
        for f in schema.fields
        if f.category == "原料" and isinstance(record.get(f.name), (int, float))
    }

    # 1.5 覆盖率检查：成分库没覆盖的原料会被跳过，告警别静默
    total_ratio = sum(ratios.values())
    covered_ratio = sum(v for k, v in ratios.items() if k in composition)
    if total_ratio > 0 and covered_ratio / total_ratio < 0.95:
        uncovered = [k for k in ratios if k not in composition]
        logger.warning(
            "成分库覆盖率 %.1f%%（未覆盖：%s），computed 字段会系统性偏低",
            covered_ratio / total_ratio * 100, "、".join(uncovered),
        )

    # 2. 加权求和得混合氧化物总量（所有 computed 字段共用）
    totals = _compute_oxide_totals(ratios, composition)

    # 3. 按每个 computed 字段的 compute 配置填值
    for f in schema.fields:
        if not f.computed or not f.compute:
            continue
        kind = f.compute.get("kind")
        if kind == "weighted_oxide":
            record[f.name] = totals.get(f.compute.get("oxide"))
        elif kind == "molar_ratio":
            record[f.name] = _molar_ratio(
                totals,
                f.compute.get("num"),
                f.compute.get("den"),
                f.compute.get("num_factor", 1),
                f.compute.get("den_factor", 1),
            )
        else:
            record[f.name] = None  # 未知计算类型，留空
    return record
