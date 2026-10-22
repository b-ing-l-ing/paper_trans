"""数据集抽取提示词 —— 按字段库把一篇论文抽成一张表。

与 paper_trans 主包的 prompts.py（文档式总结抽取）不同，
这里的目标是「数据集」：每条试验一行，列由字段库决定。

本文件只负责「生成 prompt 文本」，不调用 LLM。
"""

from __future__ import annotations

from .schema import Schema


def build_extract_prompt(
    schema: Schema,
    text: str,
    text_limit: int = 40000,
) -> str:
    """构建数据集抽取 prompt。

    Args:
        schema: 字段库（要抽取的列）。
        text: 论文全文。
        text_limit: 传入的文本上限（字符数）。

    Returns:
        抽取 prompt 字符串。
    """
    direct = schema.direct_fields()
    computed = schema.computed_names()

    field_lines = [
        f"- {f.name}（{f.label}，单位 {f.unit or '无'}）：{f.desc or f.label}"
        for f in direct
    ]
    field_desc = "\n".join(field_lines) if field_lines else "（空字段库）"
    computed_desc = "、".join(computed) if computed else "无"

    return f"""\
你是材料学论文数据抽取专家。把一篇论文抽成结构化数据集，供机器学习训练使用。

## 论文原文（可能被截断）
{text[:text_limit]}

## 字段库（你要抽取的列）
{field_desc}

## 任务

1. **每条试验 = 一行**：识别论文中每一组独立的试验条件（不同配比 / 不同掺量 / 不同温度 / 不同龄期等），每组条件抽成一行。
   - **跨表合并**：同一配方（原料和掺量完全相同）的性能常分散在论文的多个表格里（如强度表、劈裂强度表、干湿循环表）。这些表里同名的组是同一个配方，它们的性能必须合并进同一行，不要拆成多行。

2. **红线：什么算「一条试验」**：
   - 一条试验 = 一组可区分的配方/工艺条件 + **测得的性能结果**。
   - 若一张表只有原料信息、没有性能（如第一列是矿渣/钢渣等原料名，列是 SiO2/CaO 等化学成分），是原料成分说明，**不是试验，不要抽成行**。
   - 若一张表只有中间参数、没有配方和性能（如最大干密度、最佳含水率、击实试验结果等），是制备过程的中间量，**也不是试验，不要抽成行**。

3. **按字段填值**：对每一行，字段库里的列都要出现。有值填「数值 单位」；论文里没提到的字段显式填 null，不要填 0、不要省略、不要瞎猜。

4. **忠实原文 + 单位**：数值字段抽「原文数值 + 原文单位」，例如强度写 4000 kPa 就吐 "4000 kPa"，不要自己换算。

5. **计算字段填 null**：以下字段由程序计算，你显式填 null（后面程序会硬算补上）：{computed_desc}。

6. **发现模板外字段**：若论文里有重要自变量 / 变量不在字段库里（如研究「固化剂掺量」但字段库没有），加进 new_fields：
   - name：用 snake_case 英文命名，语义清晰（如 curing_agent_dosage），不要用 CA 这种缩写黑话；
   - desc：详细写清「是什么 + 单位 + 取值条件 + 与相近字段的区别」；
   - 这些字段的值要同时填进对应行的 records 里。

## 输出格式（JSON）

{{"records": [{{"字段名": "数值 单位", ...}}, ...], "new_fields": [{{"name": "...", "label": "...", "category": "...", "unit": "...", "type": "...", "desc": "..."}}], "notes": "..."}}

只输出 JSON，不要输出其他文字。"""
