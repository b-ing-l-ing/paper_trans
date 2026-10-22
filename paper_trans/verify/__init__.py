"""paper_trans.verify —— 抽取结果的可靠性验证管线。

三条腿：
- rules：零 LLM 的规则化自洽校验（列对齐 / 重复 id / 数值冲突）
- provenance：零 LLM 的证据溯源（定位页码与证据句）
- crosscheck：多 agent 独立交叉验证（LLM）

入口为 verify_paper()，输出 VerificationReport。
"""

from .models import (
    Issue,
    CrossCheckVerdict,
    ProvenanceEntry,
    VerificationReport,
)
from .pipeline import verify_paper, render_report, save_report

__all__ = [
    "Issue",
    "CrossCheckVerdict",
    "ProvenanceEntry",
    "VerificationReport",
    "verify_paper",
    "render_report",
    "save_report",
]
