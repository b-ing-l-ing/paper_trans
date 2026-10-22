"""抽取输出模型 —— 数据集行 + 新字段候选。

一次抽取的产出分两块：
- records：若干条试验（每条一行，字段名 → 值的 dict）
- new_fields：这次发现的模板外字段（带完整元信息，供人工审后入库）

本文件定义的是「JSON 证据层」的结构 —— LLM 抽完的原始结果。
数值保留「原文数值 + 单位」的字符串形式（如 "4000 kPa"），
归一化成纯数值（4.0）是后面 compute 层的事，不在这里做。
"""

from __future__ import annotations

from typing import Any, Optional

from pydantic import BaseModel, Field


class NewField(BaseModel):
    """LLM 发现的模板外字段（候选新字段，待人工确认入库）。

    注意 type / category 用宽松的 str 而非 Literal：这是 LLM 吐出来的，
    可能不按预设分类写、类型名拼错，宽松一点避免直接报错崩掉。
    """

    name: str = Field(description="字段名/列名，如 'curing_agent_dosage'")
    label: str = Field(description="中文名，如 '固化剂掺量'")
    category: str = Field(description="字段类别：原料/成分/工艺/性能")
    unit: Optional[str] = Field(default=None, description="单位")
    type: str = Field(description="字段类型：continuous/discrete/categorical")
    desc: str = Field(description="字段含义 / 取值条件")


class ExtractResult(BaseModel):
    """一次论文抽取的完整返回（JSON 证据层）。"""

    records: list[dict[str, Any]] = Field(
        default_factory=list, description="数据集行，每行一条试验，字段名 → 值"
    )
    new_fields: list[NewField] = Field(
        default_factory=list, description="本次发现的模板外字段候选"
    )
    notes: Optional[str] = Field(
        default=None, description="抽取备注（无法确定的字段、缺失原因等）"
    )
