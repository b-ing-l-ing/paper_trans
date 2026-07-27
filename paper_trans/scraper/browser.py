"""浏览器会话管理 —— Playwright 封装。

支持：
- 手动登录 → 保存 cookie / session
- 后续自动加载已保存的登录态
- Headless / headed 切换
"""

from __future__ import annotations

import logging
from pathlib import Path

from playwright.sync_api import (
    sync_playwright,
    Browser,
    BrowserContext,
    Page,
    Playwright,
)
from playwright_stealth import Stealth

logger = logging.getLogger(__name__)

DEFAULT_SESSION = Path("./session.json")

_stealth = Stealth()


def _apply_stealth(page: Page) -> None:
    """抹除自动化指纹（navigator.webdriver 等），防反爬检测。"""
    try:
        _stealth.apply_stealth_sync(page)
    except Exception as e:
        logger.debug("stealth 应用失败: %s", e)


class ScraperBrowser:
    """带 session 持久化的浏览器封装。"""

    def __init__(
        self,
        session_file: str | Path = DEFAULT_SESSION,
        headless: bool = True,
    ):
        self.session_file = Path(session_file)
        self.headless = headless
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None

    # ------------------------------------------------------------------
    # 生命周期
    # ------------------------------------------------------------------

    def start(self) -> Page:
        """启动浏览器，自动加载已有 session。返回新页面。"""
        self._pw = sync_playwright().start()
        self._browser = self._pw.chromium.launch(
            headless=self.headless,
            channel="chrome",  # 用系统 Chrome，TLS 指纹与真人一致
        )

        if self.session_file.exists():
            logger.info("加载已保存的登录态: %s", self.session_file)
            self._context = self._browser.new_context(
                storage_state=str(self.session_file)
            )
        else:
            logger.info("无 session 文件，创建全新浏览器上下文")
            self._context = self._browser.new_context()

        page = self._context.new_page()
        _apply_stealth(page)
        return page

    def close(self) -> None:
        """关闭浏览器并保存 session。"""
        if self._context:
            self._context.storage_state(path=str(self.session_file))
            logger.info("session 已保存: %s", self.session_file)
            self._context.close()
        if self._browser:
            self._browser.close()
        if self._pw:
            self._pw.stop()

    def __enter__(self) -> Page:
        return self.start()

    def __exit__(self, *args) -> None:
        self.close()

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------

    def new_page(self) -> Page:
        """在当前上下文创建新页面。"""
        if not self._context:
            raise RuntimeError("浏览器未启动，请先调用 start()")
        page = self._context.new_page()
        _apply_stealth(page)
        return page

    @property
    def has_session(self) -> bool:
        return self.session_file.exists()

    def clear_session(self) -> None:
        self.session_file.unlink(missing_ok=True)
        logger.info("session 已清除")


def login_and_save(
    login_url: str,
    session_file: str | Path = DEFAULT_SESSION,
) -> None:
    """打开有头浏览器让用户手动登录，完成后保存 session。

    默认使用系统 Chrome（含你的书签、已登录状态等）。

    用法：
        from paper_trans.scraper.browser import login_and_save
        login_and_save("https://example.com/login")

    浏览器会打开，你手动登录后按 Enter 回到终端即可。
    """
    session_file = Path(session_file)

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=False, channel="chrome")
        context = browser.new_context()
        page = context.new_page()
        _apply_stealth(page)
        page.goto(login_url)

        logger.info("浏览器已打开: %s", login_url)
        logger.info("请在浏览器中完成登录，然后回到终端按 Enter...")
        input(">>> 登录完成后按 Enter 保存 session: ")

        context.storage_state(path=str(session_file))
        logger.info("session 已保存: %s", session_file)

        context.close()
        browser.close()
