"""Verification 数据模型 —— 交叉验证、规则校验、证据溯源的结果结构。"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Issue(BaseModel):
    """一条验证发现的问题或提示。"""

    id: str = Field(description="问题唯一标识")
    severity: Literal["info", "warning", "error"] = Field(
        description="严重程度：error=确定有错, warning=疑似, info=提示"
    )
    category: Literal["missing", "incorrect", "inconsistent", "schema", "provenance"] = Field(
        description="问题类别：缺失/错误/不一致/结构/溯源"
    )
    location: Optional[str] = Field(
        default=None,
        description="定位，如 'Table 2' / 'Figure 3' / 'numerical_data[12]'",
    )
    description: str = Field(description="问题描述")
    evidence: Optional[str] = Field(
        default=None, description="原文证据片段（尽量逐字引用）"
    )
    suggestion: Optional[str] = Field(default=None, description="修正建议")
    source: Optional[str] = Field(
        default=None, description="来源（交叉验证时为审查 agent 名）"
    )


class CrossCheckVerdict(BaseModel):
    """单个审查 agent 的输出。"""

    issues: list[Issue] = Field(
        default_factory=list, description="该视角发现的问题列表"
    )


class ProvenanceEntry(BaseModel):
    """一条可溯源记录：某个提取目标在原文中的位置。"""

    target: str = Field(
        description="目标标识，如 'numerical:band gap' / 'table:Table 2'"
    )
    kind: Literal["table", "figure", "numerical"] = Field(description="目标类型")
    page: Optional[int] = Field(default=None, description="定位到的页码")
    found: bool = Field(default=False, description="是否在原文定位到")
    evidence_text: Optional[str] = Field(
        default=None, description="原文证据片段（首个匹配位置上下文）"
    )


class VerificationReport(BaseModel):
    """一次验证的完整报告。"""

    paper_title: Optional[str] = Field(default=None, description="论文标题")
    rules: list[Issue] = Field(
        default_factory=list, description="规则校验发现的问题"
    )
    crosscheck: list[Issue] = Field(
        default_factory=list, description="多 agent 交叉验证发现的问题"
    )
    provenance: list[ProvenanceEntry] = Field(
        default_factory=list, description="证据溯源记录"
    )
    summary: str = Field(default="", description="一段话总结")
