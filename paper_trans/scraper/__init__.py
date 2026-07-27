"""网页爬取模块 —— 浏览器会话 + HTML 提取。"""

from .browser import ScraperBrowser, login_and_save
from .html_extractor import scrape_paper, scrape_paper_from_url, html_to_text

__all__ = [
    "ScraperBrowser",
    "login_and_save",
    "scrape_paper",
    "scrape_paper_from_url",
    "html_to_text",
]
