"""PDF 结构化解析 —— 表格检测与单元格定位。

独立于 LLM 提取管线，只负责把 PDF 表格转成二维 JSON 结构。
"""

from .table_detector import detect_tables

__all__ = ["detect_tables"]
