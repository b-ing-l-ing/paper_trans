"""实验层：把 vision 调用结果按「图 bytes + prompt」哈希缓存到本地。

**为什么**：vision 服务端有抖动——同一张图两次跑，`detect_subfigures` 报的子图
bbox、`analyze_figure` 报的颜色都可能不同，导致对比实验每次结果都不一样，
「改前 vs 改后」没法比。这里在实验脚本里包一层缓存，同一张图复跑结果固定。

**粒度**：只缓存 `_vision_call`（vision 那一层），不缓存 `figure_to_text` 的最终
输出。所以 vision 固定了，但 `chart_extract` / `figure_extract` 的算法逻辑照常
跑——改代码，结果依然会变。这才是能用于迭代的粒度。

**不碰生产代码**：靠 monkeypatch 把 `paper2data.figure_extract._vision_call`
换成缓存版，`paper2data` 本身一行不改。

**怎么刷新**：删掉 `.vision_cache/` 目录，下次跑就会重新问 vision。
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

# 模块级抓原函数：import 时即绑定（此时还没 install，拿到的必然是原始版）
from paper2data.figure_extract import _vision_call as _real_vision_call

CACHE_DIR = Path(__file__).resolve().parent / ".vision_cache"


def cache_enabled() -> bool:
    """缓存开关（默认开）。设 COMPARE_VISION_CACHE=off 关闭——用来「每次重新抽」量抖动。"""
    return os.getenv("COMPARE_VISION_CACHE", "on").lower() not in (
        "off", "0", "false", "none", "no",
    )


def _key(image_bytes: bytes, prompt: str) -> str:
    # key 里带上模型名：换了 PAPER2DATA_VISION_MODEL 就该失效，否则会拿到旧模型的输出
    from paper2data.config import settings

    model = settings.vision_model.encode("utf-8")
    return hashlib.sha256(
        image_bytes + b"\x00" + model + b"\x00" + prompt.encode("utf-8")
    ).hexdigest()[:20]


def cached_vision_call(image_bytes: bytes, mime_type: str, prompt: str) -> str:
    """缓存版 vision 调用：命中缓存直接返回，未命中才真正调 API 并落盘。

    COMPARE_VISION_CACHE=off 时退化为完全不缓存（每次真调 API），用于测抖动。
    """
    if not cache_enabled():
        return _real_vision_call(image_bytes, mime_type, prompt)
    f = CACHE_DIR / f"{_key(image_bytes, prompt)}.txt"
    if f.exists():
        return f.read_text(encoding="utf-8")
    text = _real_vision_call(image_bytes, mime_type, prompt)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    f.write_text(text, encoding="utf-8")
    return text


def install() -> int:
    """把 paper2data 的 `_vision_call` 换成缓存版。返回缓存目录里已有的条目数。

    figure_extract 内部的 vision 调用按模块全局名解析，所以 patch 之后
    `figure_to_text` 里所有 vision 调用都会走缓存；`paper2data` 本身一行不改。
    """
    from paper2data import figure_extract

    figure_extract._vision_call = cached_vision_call
    return len(list(CACHE_DIR.glob("*.txt"))) if CACHE_DIR.exists() else 0
