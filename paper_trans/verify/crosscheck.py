"""多 Agent 交叉验证 —— 用多个独立审查视角核对同一份提取结果（LLM）。

与 extractor 的 self-review 不同：这里不是「让同一个模型再看一遍自己的答案」，
而是启动多个持不同视角的独立 agent 各自核对，再合并去重。
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from ..client import get_client
from ..config import settings
from ..schemas import PaperData
from ..extractor import _format_tables_for_llm
from .models import CrossCheckVerdict, Issue
from .arithmetic import build_table_facts

logger = logging.getLogger(__name__)


# 每个审查 agent 持一个独立视角，避免单视角盲区。
AGENT_LENSES = [
    {
        "name": "完整性审计",
        "category": "missing",
        "brief": "找出被遗漏的内容：正文里还有哪些表格、图表、数值、方法没有被抽取出来。",
    },
    {
        "name": "准确性审计",
        "category": "incorrect",
        "brief": "找出抽取错误：数值抄错、列错位、图表描述与原文不符、caption 张冠李戴。",
    },
    {
        "name": "一致性审计",
        "category": "inconsistent",
        "brief": "找出自相矛盾：正文报告值与程序硬算的表格事实不符、同一指标取值冲突、正文结论与表格/图对不上。对每个冲突写出算式和结果，不要心算。",
    },
]


def _build_prompt(
    lens: dict, content: str, data_json: str, text_limit: int
) -> str:
    return f"""\
你是论文数据抽取的独立审查员。你负责的视角是「{lens['name']}」。

你的职责：{lens['brief']}

## 论文原文（可能被截断）
{content[:text_limit]}

## 待审查的提取结果（JSON）
{data_json}

请严格围绕「{lens['name']}」这一视角逐条核对，只报告与你的视角相关的问题。
每条问题必须给出：
- location：定位，如 "Table 2" / "Figure 3" / "numerical_data[12]"
- severity：error（确定有错）/ warning（疑似）/ info（提示）
- description：具体描述
- evidence：原文证据，尽量逐字引用
- suggestion：修正建议

如果没有发现与你视角相关的问题，返回空列表，不要为了凑数而编造。"""


def _run_one(
    client,
    model_name: str,
    lens: dict,
    content: str,
    data_json: str,
    text_limit: int,
) -> CrossCheckVerdict:
    prompt = _build_prompt(lens, content, data_json, text_limit)
    return client.chat.completions.create(
        model=model_name,
        response_model=CrossCheckVerdict,
        max_retries=settings.max_retries,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )


def _dedupe(issues: list[Issue]) -> list[Issue]:
    """按 (类别, 位置, 描述) 粗略去重，合并不同 agent 报告的同一问题。"""
    seen: set[tuple] = set()
    result: list[Issue] = []
    for issue in issues:
        desc = issue.description.lower().strip()
        key = (issue.category, issue.location, desc[:40])
        if key in seen:
            continue
        seen.add(key)
        result.append(issue)
    return result


def run_crosscheck(
    text: str,
    data: PaperData,
    tables: list[dict] | None = None,
    model: str | None = None,
    n_agents: int = 3,
) -> list[Issue]:
    """并行运行多个独立审查 agent，合并去重后返回问题列表。

    Args:
        text: 论文全文。
        data: 已提取的结构化数据。
        tables: 结构化表格（detect_tables 输出），可选。
        model: LLM 模型名称，None 使用默认。
        n_agents: 审查 agent 数量（1-3，对应三个视角）。

    Returns:
        去重后的问题列表。
    """
    client = get_client()
    model_name = model or settings.model_name
    data_json = data.model_dump_json(indent=2)

    content = text
    if tables:
        content += "\n\n" + _format_tables_for_llm(tables)
    facts = build_table_facts(data)
    if facts:
        content += "\n\n" + facts

    lenses = AGENT_LENSES[: max(1, min(n_agents, len(AGENT_LENSES)))]
    text_limit = settings.review_text_limit
    logger.info(
        "交叉验证：启动 %d 个独立审查 agent（模型 %s）", len(lenses), model_name
    )

    issues: list[Issue] = []
    with ThreadPoolExecutor(max_workers=len(lenses)) as ex:
        futures = {
            ex.submit(
                _run_one, client, model_name, lens, content, data_json, text_limit
            ): lens
            for lens in lenses
        }
        for fut in as_completed(futures):
            lens = futures[fut]
            try:
                verdict = fut.result()
            except Exception as e:
                logger.warning("审查 agent「%s」失败：%s", lens["name"], e)
                continue
            for issue in verdict.issues:
                issue.source = lens["name"]
                issues.append(issue)

    deduped = _dedupe(issues)
    logger.info("交叉验证完成：%d 条问题（去重后）", len(deduped))
    return deduped
