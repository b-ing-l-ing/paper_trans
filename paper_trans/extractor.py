"""LLM 提取 + 自审循环。"""

import json
import logging
import re
from copy import deepcopy

from .client import get_client
from .schemas import PaperData
from .prompts import EXTRACTION_SYSTEM_PROMPT, build_review_prompt
from .config import settings

logger = logging.getLogger(__name__)


def extract_data(
    text: str,
    model: str | None = None,
    tables: list[dict] | None = None,
) -> PaperData:
    """从论文文本中提取结构化数据。

    Args:
        text: 论文全文（纯文本）。
        model: 模型名称，None 则使用默认配置。
        tables: 结构化表格数据（detect_tables 输出），可选。

    Returns:
        PaperData 实例。
    """
    client = get_client()
    model_name = model or settings.model_name
    logger.info("正在提取数据（模型: %s）...", model_name)

    # 将结构化表格追加到文本末尾
    content = text
    if tables:
        content += "\n\n" + _format_tables_for_llm(tables)

    return client.chat.completions.create(
        model=model_name,
        response_model=PaperData,
        max_retries=settings.max_retries,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=settings.temperature,
    )


def review_extraction(data: PaperData, text: str, model: str | None = None, tables: list[dict] | None = None) -> PaperData | None:
    """自查：审核提取结果，如有遗漏/错误返回修正版。

    Args:
        data: 首次提取的结果。
        text: 论文全文。
        model: 模型名称，None 则使用默认配置。
        tables: 结构化表格数据，可选。

    Returns:
        修正后的 PaperData，或 None（表示无需修正）。
    """
    client = get_client()
    data_json = data.model_dump_json(indent=2)
    content = text
    if tables:
        content += "\n\n" + _format_tables_for_llm(tables)
    prompt = build_review_prompt(data_json, content, settings.review_text_limit)
    model_name = model or settings.model_name

    logger.info("正在进行自查审核...")

    result = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    raw = (result.choices[0].message.content or "").strip()

    # LLM 认为无需修改
    if raw.upper().startswith("OK"):
        logger.info("审核通过，无需修正。")
        return None

    # 解析修正后的 JSON（使用 Instructor 的 response_model 做二次提取）
    # 回退：直接解析 JSON
    return _parse_revised_json(raw)


def _format_tables_for_llm(tables: list[dict]) -> str:
    """把 detect_tables 的输出转成 LLM 可读的文本。"""
    parts: list[str] = ["## 结构化表格数据（精确列对齐，优先使用）\n"]
    for i, t in enumerate(tables, 1):
        parts.append(f"\n### 表格 {i}（第{t.get('page', '?')}页，{t['columns']}列）\n")
        # 表头
        parts.append("| " + " | ".join(t["header"]) + " |")
        parts.append("|" + "|".join("---" for _ in t["header"]) + "|")
        # 数据行
        for row in t["rows"]:
            cells = [str(v) if v is not None else "(空)" for v in row]
            # 补齐列数
            while len(cells) < len(t["header"]):
                cells.append("-")
            parts.append("| " + " | ".join(cells[:len(t["header"])]) + " |")
        parts.append("")
    return "\n".join(parts)


def _clean_schema(schema: dict) -> dict:
    """递归移除 JSON schema 中的 additionalProperties（Gemini 免费 API 限制）。"""
    schema = deepcopy(schema)

    def _clean(d: dict) -> None:
        d.pop("additionalProperties", None)
        for v in d.values():
            if isinstance(v, dict):
                _clean(v)
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        _clean(item)

    _clean(schema)
    return schema


def _parse_revised_json(raw: str) -> PaperData | None:
    """解析自查返回的修正 JSON。"""
    # 尝试提取代码块中的 JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)

    # 截取 JSON 对象
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        logger.warning("审查结果 JSON 解析失败: %s", e)
        return None
    except Exception as e:
        logger.warning("审查结果处理失败: %s", e)
        return None
    try:
        return PaperData(**parsed)
    except Exception as e:
        logger.warning("审查结果不符合 schema: %s", e)
        return None


def extract_with_review(
    text: str,
    model: str | None = None,
    tables: list[dict] | None = None,
) -> PaperData:
    """提取 + 自审：先提取，再审核修正。

    Args:
        text: 论文全文。
        model: 模型名称，None 则使用默认配置。
        tables: 结构化表格数据，可选。

    Returns:
        最终 PaperData。
    """
    data = extract_data(text, model=model, tables=tables)
    improved = review_extraction(data, text, model=model, tables=tables)
    if improved is not None:
        logger.info("已应用审查修正。")
        return improved
    return data


# ---------------------------------------------------------------------------
# 多模态提取（Gemini）
# ---------------------------------------------------------------------------

MULTIMODAL_SYSTEM_PROMPT = """\
You are a scientific data extraction specialist. Read the paper text AND the \
provided images (figures, charts, tables from the paper). Extract ALL \
structured information comprehensively.

## Core Principles

1. **READ THE IMAGES**: Each image is a figure or chart from the paper. Look \
at them directly — read axis labels, data points, legends, annotations. \
Extract exact numerical values from charts and graphs.

2. **TABLES**: The input contains structured table data in markdown format. \
"(空)" marks a genuinely empty cell. Keep it empty ("") in your output. \
Do NOT fill empty cells with 0, -, or any placeholder. \
If a row has fewer values than its header columns, those columns are empty. \
Never shift values sideways to fill gaps.

3. **FIGURES**: For EACH image, provide a detailed, structured description:
   - **Content**: What exactly is shown (samples, conditions, variables).
   - **Sub-figures**: Describe each (a)(b)(c) panel SEPARATELY.
   - **Trends**: Peaks, plateaus, inflection points, best/worst performers.
   - **Comparisons**: Between groups, with approximate magnitudes.
   - **Axis ranges**: X/Y labels, units, and approximate data ranges.
   - **Data points**: Every visible value — bars, markers, error bars, \
   percentages, thresholds. Be exhaustive, not vague.

4. **NUMERICAL VALUES**: Scan text AND images for quantitative results. \
Extract name, value, unit, and context for each. Be exhaustive.

5. **METHODS**: Summarize each experimental/analytical procedure in 1-3 \
sentences. List key techniques. Do NOT enumerate every detail.

6. **ACCURACY**: Only extract data explicitly stated in the paper text or \
visible in the provided images. Never invent or guess.

7. **LANGUAGE**: Papers may be in English or Chinese. Extract metadata in \
original language. Tags in English.

8. **COMPLETENESS**: When in doubt, include it.
"""


def extract_multimodal(
    text: str,
    image_paths: list[str],
    model: str | None = None,
    tables: list[dict] | None = None,
) -> PaperData:
    """多模态提取：文本 + 图片 → 结构化数据。

    Args:
        text: 论文全文（纯文本）。
        image_paths: 从 PDF 提取的图片文件路径列表。
        model: Gemini 模型名称，None 使用默认配置。
        tables: 结构化表格数据，可选。

    Returns:
        PaperData 实例。
    """
    from pathlib import Path

    from .client import get_gemini_client
    from google.genai import types

    client = get_gemini_client()
    model_name = model or settings.multimodal_model
    logger.info("多模态提取（模型: %s，%d 张图）...", model_name, len(image_paths))

    # 构建内容：文本 + 结构化表格 + 图片
    content = text
    if tables:
        content += "\n\n" + _format_tables_for_llm(tables)

    parts: list[types.Part] = [
        types.Part.from_text(text=MULTIMODAL_SYSTEM_PROMPT),
        types.Part.from_text(text=f"\n\n---\n\n{content}"),
    ]

    for img_path in image_paths:
        try:
            img_data = Path(img_path).read_bytes()
            parts.append(types.Part.from_bytes(data=img_data, mime_type="image/png"))
        except Exception as e:
            logger.debug("跳过图片 %s: %s", img_path, e)

    # 结构化输出（去掉 additionalProperties，Gemini 免费 API 不支持）
    schema = _clean_schema(PaperData.model_json_schema())

    response = client.models.generate_content(
        model=model_name,
        contents=types.Content(role="user", parts=parts),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
            response_schema=schema,
            max_output_tokens=32768,
        ),
    )

    if response.text is None:
        raise RuntimeError("Gemini 返回空响应")

    raw = response.text
    # 尝试从 markdown 代码块中提取 JSON
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", raw)
    if match:
        raw = match.group(1)
    # 截取 JSON 对象
    start, end = raw.find("{"), raw.rfind("}")
    if start >= 0 and end > start:
        raw = raw[start : end + 1]

    data = json.loads(raw)
    return PaperData(**data)


