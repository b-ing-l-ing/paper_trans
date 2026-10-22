"""Schema 层 —— 字段库与字段定义。

字段库是抽取的「模板」：每个字段描述数据集的一列。
它是持久化的 JSON，可以人工审阅、随抽取不断扩充。

本文件只定义数据结构，不调用 LLM、不干活。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field

# 字段类型：连续数值 / 离散整数 / 类别。
# 直接对应 CTGAN 的 continuous / discrete / categorical，方便下游无缝衔接。
FieldType = Literal["continuous", "discrete", "categorical"]

# 字段类别，取自「数据标签解释」docx 的分类。
FieldCategory = Literal["原料", "成分", "工艺", "性能"]


class FieldSpec(BaseModel):
    """一个字段（数据集的一列）的完整定义。

    注意：类名不叫 Field，因为 pydantic 已有一个 Field（用于字段描述），
    重名会与 LLM 结构化输出冲突。
    """

    name: str = Field(description="字段名/列名，唯一标识，如 'C-28d'")
    label: str = Field(description="中文名，如 '28天抗压强度'")
    category: FieldCategory = Field(description="字段类别：原料/成分/工艺/性能")
    unit: Optional[str] = Field(
        default=None, description="单位，如 'MPa' / 'wt%'；无量纲则为 null"
    )
    type: FieldType = Field(
        description="字段类型：continuous / discrete / categorical"
    )
    computed: bool = Field(
        default=False, description="是否计算字段（代码硬算，LLM 不填）"
    )
    desc: str = Field(
        default="", description="字段描述，给 LLM 的抽取提示，含取值条件"
    )
    aliases: list[str] = Field(
        default_factory=list,
        description="历史别名列表，如 'CA'、'Stabilizer' 都归一到标准名",
    )
    compute: Optional[dict] = Field(
        default=None,
        description=(
            "computed 字段的计算配置，如 {'kind': 'weighted_oxide', 'oxide': 'MgO'} "
            "或 {'kind': 'molar_ratio', 'num': 'CaO', 'den': 'SiO2'}"
        ),
    )


class Schema(BaseModel):
    """一份字段库。domain 标识领域，fields 是字段列表。"""

    domain: str = Field(description="领域名，如 '煤基多源固废低碳胶凝材料'")
    version: str = Field(default="1.0", description="字段库版本")
    fields: list[FieldSpec] = Field(default_factory=list, description="字段列表")

    # ------------------------------------------------------------------
    # 便捷方法
    # ------------------------------------------------------------------

    def field_names(self) -> list[str]:
        """所有字段名。"""
        return [f.name for f in self.fields]

    def computed_names(self) -> list[str]:
        """计算字段名（LLM 跳过，代码硬算）。"""
        return [f.name for f in self.fields if f.computed]

    def direct_fields(self) -> list[FieldSpec]:
        """LLM 需要读取（非计算）的字段。"""
        return [f for f in self.fields if not f.computed]

    def find_by_name_or_alias(self, name: str) -> FieldSpec | None:
        """按标准名或别名查字段，用于别名归一。"""
        for f in self.fields:
            if f.name == name or name in f.aliases:
                return f
        return None

    # ------------------------------------------------------------------
    # 持久化
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, path: str | Path) -> "Schema":
        """从 JSON 文件加载字段库。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    def save(self, path: str | Path) -> None:
        """保存字段库到 JSON。"""
        Path(path).write_text(
            json.dumps(self.model_dump(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
