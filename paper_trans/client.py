"""LLM 客户端 —— 封装 Instructor + OpenAI 兼容接口，以及 Gemini 多模态。"""

from openai import OpenAI
import instructor
from google import genai

from .config import settings


# ---------------------------------------------------------------------------
# DeepSeek（纯文本）
# ---------------------------------------------------------------------------

def create_client(model: str | None = None) -> instructor.Instructor:
    """创建 Instructor 包装的 DeepSeek 客户端。"""
    return instructor.patch(
        OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        ),
        mode=instructor.Mode.MD_JSON,
    )


_client: instructor.Instructor | None = None


def get_client() -> instructor.Instructor:
    """获取默认 DeepSeek 客户端（单例）。"""
    global _client
    if _client is None:
        _client = create_client()
    return _client


# ---------------------------------------------------------------------------
# Gemini（多模态）
# ---------------------------------------------------------------------------

_gemini_client: genai.Client | None = None


def get_gemini_client() -> genai.Client:
    """获取 Gemini 客户端（单例），用于多模态提取。"""
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(api_key=settings.gemini_api_key)
    return _gemini_client
