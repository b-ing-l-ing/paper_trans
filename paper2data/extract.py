"""执行层 —— 调用 LLM，把论文抽成数据集。

这是「agent」真正跑起来的地方：把 prompt 发给 LLM，
用 instructor 的 response_model 强制它返回 ExtractResult 结构。

顶层结构（records / new_fields / notes）靠 response_model 锁死；
行内的列名是动态的，靠 prompts.py 里的格式约束。
"""

from __future__ import annotations

import logging

from .schema import Schema
from .records import ExtractResult
from .prompts import build_extract_prompt

# 自包含：使用本包内的 LLM 客户端与配置。
from .client import get_client
from .config import settings

logger = logging.getLogger(__name__)


def extract_dataset(
    text: str,
    schema: Schema,
    model: str | None = None,
    text_limit: int = 40000,
) -> ExtractResult:
    """从论文文本抽取数据集。

    Args:
        text: 论文全文。
        schema: 字段库。
        model: 模型名，None 用默认配置。
        text_limit: 传入的文本上限（字符数）。

    Returns:
        ExtractResult（records + new_fields + notes）。
    """
    client = get_client()
    model_name = model or settings.model_name
    prompt = build_extract_prompt(schema, text, text_limit)

    logger.info("正在抽取数据集（模型 %s）...", model_name)

    result = client.chat.completions.create(
        model=model_name,
        response_model=ExtractResult,
        max_retries=settings.max_retries,
        messages=[{"role": "user", "content": prompt}],
        temperature=settings.temperature,
    )
    return result
