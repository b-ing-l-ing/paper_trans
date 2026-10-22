"""字段归并 —— 新字段与字段库的别名归一。

第一层（本文件）：字面精确匹配，纯代码，零 LLM。
  新字段名 / 别名命中某字段的 name 或 aliases → 归一到标准名。
第二层：LLM 语义同义判断（语义同但字面不同，如 CA vs Stabilizer），后续补。
"""

from __future__ import annotations

from .schema import Schema
from .records import NewField


def resolve_new_fields(
    new_fields: list[NewField],
    schema: Schema,
) -> list[dict]:
    """对新字段做别名归一（字面匹配）。

    返回列表，每项：
    - name: 新字段名
    - resolved_to: 归一到哪个标准字段名；None 表示真正的新字段
    - is_new: 是否真正的新字段
    """
    result: list[dict] = []
    for nf in new_fields:
        matched = schema.find_by_name_or_alias(nf.name)
        result.append(
            {
                "name": nf.name,
                "resolved_to": matched.name if matched else None,
                "is_new": matched is None,
            }
        )
    return result
