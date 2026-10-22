"""paper2data 配置 —— 从 .env 文件和环境变量加载。

独立于 paper_trans 主包：环境变量用 PAPER2DATA_ 前缀，避免两包共用配置打架。
"""

import os
from pathlib import Path

from dotenv import load_dotenv


def _find_env() -> str | None:
    """向上查找 .env 文件，返回路径或 None。"""
    cwd = Path.cwd()
    for parent in [cwd, *cwd.parents]:
        env_file = parent / ".env"
        if env_file.exists():
            return str(env_file)
    return None


# 自动加载 .env
env_path = _find_env()
if env_path:
    load_dotenv(env_path)


class Settings:
    """paper2data 配置。所有值从环境变量读取，有合理默认值。"""

    # --- LLM ---
    @property
    def deepseek_api_key(self) -> str:
        return os.getenv("DEEPSEEK_API_KEY", "")

    @property
    def deepseek_base_url(self) -> str:
        return os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

    @property
    def model_name(self) -> str:
        return os.getenv("PAPER2DATA_MODEL", "deepseek-v4-flash")

    @property
    def gemini_api_key(self) -> str:
        return os.getenv("GEMINI_API_KEY", "")

    @property
    def multimodal_model(self) -> str:
        return os.getenv("PAPER2DATA_MULTIMODAL_MODEL", "gemini-3.6-flash")

    @property
    def vision_model(self) -> str:
        return os.getenv("PAPER2DATA_VISION_MODEL", "deepseek-v4-flash-vision-exp")

    @property
    def max_retries(self) -> int:
        return int(os.getenv("PAPER2DATA_MAX_RETRIES", "3"))

    @property
    def temperature(self) -> float:
        return float(os.getenv("PAPER2DATA_TEMPERATURE", "0"))

    # --- 输出 ---
    @property
    def output_dir(self) -> str:
        return os.getenv("PAPER2DATA_OUTPUT_DIR", "./output")

    @property
    def image_dir(self) -> str:
        return os.getenv("PAPER2DATA_IMAGE_DIR", "./output/images")


# 全局单例
settings = Settings()
