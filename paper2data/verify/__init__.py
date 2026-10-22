"""verify —— 数据集抽取的验证管线。

三条腿（取 JMI 综述「物理检查 / 溯源 / 交叉验证」精华）：
- rules.py：零 LLM 的物理检查（比例和 / 值域 / 摩尔比）
- provenance.py：数据溯源（后续）
- crosscheck.py：多 agent 交叉验证（后续）
"""

from .rules import check_records

__all__ = ["check_records"]
