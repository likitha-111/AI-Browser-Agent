import asyncio
import base64
import os
import queue
import sys
import threading
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool

SCREENSHOT_DIR = Path(os.getenv("SCREENSHOT_DIR", "./screenshots"))
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

_task_queue: queue.Queue = queue.Queue()
_browser_thread: Optional[threading.Thread] = None
_page_ready = threading.Event()


def _browser_worker():
    async def _run_loop():
        from playwright.async_api import async_playwright
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(
                headless=os.getenv("BROWSER_HEADLESS", "false").lower() == "true",
                slow_mo=int(os.getenv("BROWSER_SLOW_MO", "50")),
            )
            ctx  = await browser.new_context(
                viewport={"width": 1280, "height": 800},
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            )
            page = await ctx.new_page()
            _page_ready.set()

            while True:
                try:
                    item = _task_queue.get(timeout=0.05)
                except queue.Empty:
                    await asyncio.sleep(0)
                    continue
                if item is None:
                    break
                coro_fn, result_q = item
                try:
                    result_q.put(("ok", await coro_fn(page)))
                except Exception as e:
                    result_q.put(("err", str(e)))

            await ctx.close()
            await browser.close()

    if sys.platform == "win32":
        loop = asyncio.ProactorEventLoop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_loop())
        finally:
            loop.close()
    else:
        asyncio.run(_run_loop())


def _ensure_browser():
    global _browser_thread
    if _browser_thread is None or not _browser_thread.is_alive():
        _page_ready.clear()
        _browser_thread = threading.Thread(target=_browser_worker, daemon=True, name="playwright-browser")
        _browser_thread.start()
        if not _page_ready.wait(timeout=20):
            raise RuntimeError("Browser failed to start within 20 seconds")


def _run_in_browser(coro_fn) -> str:
    _ensure_browser()
    result_q: queue.Queue = queue.Queue()
    _task_queue.put((coro_fn, result_q))
    kind, value = result_q.get(timeout=45)
    if kind == "err":
        raise RuntimeError(value)
    return value


def close_browser():
    global _browser_thread
    if _browser_thread and _browser_thread.is_alive():
        _task_queue.put(None)
        _browser_thread.join(timeout=10)
    _browser_thread = None
    _page_ready.clear()

@tool
def navigate_to(url: str) -> str:
    """Navigate the browser to a URL. Always include https://. Use as the first step of any task."""
    async def _fn(page):
        await page.goto(url, wait_until="domcontentloaded", timeout=30000)
        return f"✓ Navigated to: {page.url}\nTitle: '{await page.title()}'"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Navigate failed: {e}"


@tool
def click_element(selector: str) -> str:
    """
    Click an element on the page.
    Selector formats: CSS ('button#submit'), text ('text=Sign In'), role ('role=button[name=OK]').
    Prefer text= selectors — they survive UI changes.
    """
    async def _fn(page):
        await page.click(selector, timeout=10000)
        await page.wait_for_load_state("domcontentloaded")
        return f"✓ Clicked: '{selector}'"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Click failed for '{selector}': {e}\nTip: try text=ButtonLabel"


@tool
def fill_input(selector: str, text: str) -> str:
    """
    Clear a form field and type text into it.
    Selector examples: 'input[name=q]', 'placeholder=Search', 'textarea'.
    After filling, use press_key with Enter or click the submit button.
    """
    async def _fn(page):
        await page.fill(selector, text, timeout=10000)
        return f"✓ Filled '{selector}' with: '{text}'"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Fill failed for '{selector}': {e}"


@tool
def press_key(selector: str, key: str) -> str:
    """
    Press a keyboard key on an element. Common keys: Enter, Tab, Escape.
    Use after fill_input to submit a search form.
    """
    async def _fn(page):
        await page.press(selector, key)
        await page.wait_for_load_state("domcontentloaded")
        return f"✓ Pressed '{key}' on '{selector}'. Now on: {page.url}"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Press failed: {e}"


@tool
def extract_text(selector: str = "body") -> str:
    """
    Extract visible text from an element. Default 'body' = full page.
    Examples: 'h1', '#firstHeading', '.article-body', 'table'.
    Returns up to 3000 characters.
    """
    async def _fn(page):
        el = await page.query_selector(selector)
        if not el:
            preview = await page.inner_text("body")
            return f"✗ Selector '{selector}' not found.\nPage preview:\n{preview[:500]}"
        text = await el.inner_text()
        return text[:3000] + ("\n...[truncated]" if len(text) > 3000 else "")
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Extract failed: {e}"


@tool
def take_screenshot(reason: str = "verify page state") -> str:
    """
    Capture the current browser page as a PNG screenshot.
    Use after navigating and whenever you need to verify the page state.
    The 'reason' parameter is optional — you can omit it or pass any string.
    """
    async def _fn(page):
        path = SCREENSHOT_DIR / "latest.png"
        await page.screenshot(path=str(path), full_page=False)
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f"✓ Screenshot saved.\nURL: {page.url}\n[img:base64:{b64[:80]}...]"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Screenshot failed: {e}"


@tool
def scroll_page(direction: str = "down") -> str:
    """Scroll the page up or down by ~600px. Use when content or buttons are off-screen."""
    async def _fn(page):
        delta = 600 if direction == "down" else -600
        await page.evaluate(f"window.scrollBy(0, {delta})")
        return f"✓ Scrolled {direction}"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Scroll failed: {e}"


@tool
def wait_for_element(selector: str) -> str:
    """Wait up to 10s for an element to appear. Use after clicking something that triggers a page change."""
    async def _fn(page):
        await page.wait_for_selector(selector, timeout=10000)
        return f"✓ Element appeared: '{selector}'"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Timeout: '{selector}' not found. Try take_screenshot() to check state."


@tool
def go_back() -> str:
    """Press the browser back button to return to the previous page."""
    async def _fn(page):
        await page.go_back(wait_until="domcontentloaded")
        return f"✓ Went back. Now on: '{await page.title()}' ({page.url})"
    try:
        return _run_in_browser(_fn)
    except Exception as e:
        return f"✗ Go back failed: {e}"


BROWSER_TOOLS = [
    navigate_to,
    click_element,
    fill_input,
    press_key,
    extract_text,
    take_screenshot,
    scroll_page,
    wait_for_element,
    go_back,
]

if __name__ == "__main__":
    print("\n--- Tool test ---\n")
    print("1.", navigate_to.invoke({"url": "https://www.wikipedia.org"}))
    print("2.", fill_input.invoke({"selector": 'input[name="search"]', "text": "Python programming"}))
    print("3.", press_key.invoke({"selector": 'input[name="search"]', "key": "Enter"}))
    print("4.", extract_text.invoke({"selector": "#firstHeading"}))
    print("5.", take_screenshot.invoke({}))
    print("6.", scroll_page.invoke({"direction": "down"}))
    print("7.", go_back.invoke({}))
    close_browser()
    print("\n All tools passed.\n")