"""verify —— 数据集抽取的验证管线。

三条腿（取 JMI 综述「物理检查 / 溯源 / 交叉验证」精华）：
- rules.py：零 LLM 的物理检查（比例和 / 值域 / 摩尔比）
- provenance.py：**数据溯源**（已实现）—— 每个值回原文定位页码 + 证据句，零 LLM
- crosscheck.py：多 agent 交叉验证（**尚未实现**）
"""

from .provenance import (
    ProvenanceEntry,
    ProvenanceReport,
    build_provenance,
    summarize_provenance,
    to_csv_rows,
)
from .rules import check_records

__all__ = [
    "check_records",
    "build_provenance",
    "summarize_provenance",
    "to_csv_rows",
    "ProvenanceEntry",
    "ProvenanceReport",
]
