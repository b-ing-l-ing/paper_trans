"""Pydantic 数据模型 —— 领域无关的论文数据结构。

设计原则：
- 表格/图表/数值数据是第一优先级（全面、准确）
- 实验方法只需概括性总结
- 不绑定任何特定学科
"""

from pydantic import BaseModel, Field
from typing import Optional


class TableData(BaseModel):
    """论文中的完整表格。"""
    table_id: Optional[str] = Field(
        default=None,
        description="表格标识，如 'Table 2', 'Tab. S1'",
    )
    caption: Optional[str] = Field(
        default=None,
        description="表格标题原文",
    )
    headers: list[str] = Field(
        default_factory=list,
        description="列标题",
    )
    rows: list[list[str]] = Field(
        default_factory=list,
        description="数据行；每行是一个字符串列表，对应各列",
    )
    notes: Optional[str] = Field(
        default=None,
        description="表格脚注或补充说明",
    )


class FigureData(BaseModel):
    """论文中的图表 / 图片描述及数据提取。"""
    figure_id: Optional[str] = Field(
        default=None,
        description="图表标识，如 'Figure 3', 'Fig. 2b'",
    )
    caption: Optional[str] = Field(
        default=None,
        description="图表标题原文",
    )
    description: str = Field(
        description="图表显示的内容及可从中得出的结论",
    )
    chart_type: Optional[str] = Field(
        default=None,
        description="图表类型，如 'bar chart', 'scatter plot', 'SEM image', 'flowchart'",
    )
    key_data_points: Optional[dict[str, str]] = Field(
        default=None,
        description="图表中标注的关键数值，name-value 对",
    )


class NumericalDatum(BaseModel):
    """从正文、表格或图表中提取的定量数据点。"""
    name: str = Field(
        description="简短描述，如 'accuracy', 'band gap', 'p-value', 'sample size'",
    )
    value: str = Field(
        description="数值及不确定度，如 '92.3%', '3.2 ± 0.1', '< 0.001'",
    )
    unit: Optional[str] = Field(
        default=None,
        description="单位，如 'eV', 'nm', 'K', '%'",
    )
    context: Optional[str] = Field(
        default=None,
        description="该值所属的实验/样本/条件",
    )
    category: Optional[str] = Field(
        default=None,
        description="大致分类，如 'performance_metric', 'physical_property', 'statistical_test'",
    )


class MethodSummary(BaseModel):
    """实验或分析方法的概括性总结。"""
    name: str = Field(
        description="方法/步骤的简短名称，如 'Sample Preparation', 'Model Training'",
    )
    description: str = Field(
        description="1-3 句话概括做了什么、为什么做",
    )
    techniques: list[str] = Field(
        default_factory=list,
        description="使用的技术/仪器，如 'XRD', 'PCR', 'random forest'",
    )


class PaperData(BaseModel):
    """一篇论文的完整结构化数据。"""

    # --- 元数据 ---
    paper_title: Optional[str] = Field(default=None, description="论文标题")
    authors: list[str] = Field(default_factory=list, description="作者列表")
    doi: Optional[str] = Field(default=None, description="DOI")
    journal: Optional[str] = Field(default=None, description="期刊/会议名称")
    year: Optional[str] = Field(default=None, description="发表年份")
    abstract: Optional[str] = Field(default=None, description="摘要")
    field_tags: list[str] = Field(
        default_factory=list,
        description="学科标签，如 'materials', 'biology', 'CS', 'physics'",
    )

    # --- 核心内容 ---
    research_objective: Optional[str] = Field(
        default=None,
        description="研究目标/核心问题",
    )
    methods: list[MethodSummary] = Field(
        default_factory=list,
        description="实验/分析方法的高层次概括（不要罗列每个合成步骤）",
    )

    # --- 数据（最高优先级） ---
    tables: list[TableData] = Field(
        default_factory=list,
        description="论文中所有表格的完整内容",
    )
    figures: list[FigureData] = Field(
        default_factory=list,
        description="所有图表及其关键数据",
    )
    numerical_data: list[NumericalDatum] = Field(
        default_factory=list,
        description="论文中出现的所有定量数据",
    )

    # --- 总结 ---
    key_findings: list[str] = Field(
        default_factory=list,
        description="主要发现，每项一条",
    )
    conclusions: Optional[str] = Field(default=None, description="结论")
