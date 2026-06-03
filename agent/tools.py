"""
agent/tools.py — Windows-compatible version
Root cause fix: _page is thread-local when using ThreadPoolExecutor.
Solution: Store browser state in asyncio tasks, not module globals.
"""
import asyncio
import base64
import os
from pathlib import Path
from typing import Optional
from langchain_core.tools import tool
from playwright.async_api import async_playwright, Browser, Page, BrowserContext

# ─────────────────────────────────────────────────────────────────────────────
# Shared state — ONE event loop owns these objects the whole time
# The fix: never pass _page across threads. Keep everything in one loop.
# ─────────────────────────────────────────────────────────────────────────────
_browser:    Optional[Browser]        = None
_context:    Optional[BrowserContext] = None
_page:       Optional[Page]           = None
_playwright                           = None
_loop:       Optional[asyncio.AbstractEventLoop] = None  # <-- track the owning loop

SCREENSHOT_DIR = Path("./screenshots")
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)


async def get_page() -> Page:
    global _browser, _context, _page, _playwright, _loop
    if _page is None or _page.is_closed():
        _loop       = asyncio.get_running_loop()          # remember which loop owns browser
        _playwright = await async_playwright().start()
        _browser    = await _playwright.chromium.launch(
            headless=False,
            slow_mo=50,
        )
        _context = await _browser.new_context(
            viewport={"width": 1280, "height": 800},
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        )
        _page = await _context.new_page()
    return _page


async def close_browser():
    global _browser, _context, _page, _playwright, _loop
    try:
        if _page    and not _page.is_closed(): await _page.close()
        if _context: await _context.close()
        if _browser: await _browser.close()
        if _playwright: await _playwright.stop()
    except Exception:
        pass
    _browser = _context = _page = _playwright = _loop = None


# ─────────────────────────────────────────────────────────────────────────────
# Async/Sync bridge — Windows-safe version
#
# KEY INSIGHT: On Windows, asyncio uses ProactorEventLoop. You CANNOT run a
# second event loop in a ThreadPoolExecutor if Playwright objects were created
# in the first loop — they are bound to that loop's I/O handles.
#
# SOLUTION: Always use the SAME event loop. If one is running, schedule a
# coroutine on it using run_coroutine_threadsafe (safe cross-thread). If none
# is running, create one fresh with asyncio.run().
# ─────────────────────────────────────────────────────────────────────────────
def _run(coro):
    global _loop
    try:
        running_loop = asyncio.get_running_loop()
    except RuntimeError:
        running_loop = None

    if running_loop is not None:
        # We're inside an already-running loop (e.g. called from async test).
        # Cannot call run_until_complete here — just return the coroutine.
        # Caller must await it. This case only hits the manual test.
        raise RuntimeError(
            "Called a sync tool from inside an async context. "
            "Use 'await _inner()' directly instead of tool.invoke() in async code."
        )

    if _loop is not None and _loop.is_running():
        # LangGraph scenario: submit to the loop that owns the browser
        import concurrent.futures
        future = asyncio.run_coroutine_threadsafe(coro, _loop)
        return future.result(timeout=60)

    # Fresh call — no loop yet. Safe to use asyncio.run().
    return asyncio.run(coro)


# ─────────────────────────────────────────────────────────────────────────────
# Tools
# ─────────────────────────────────────────────────────────────────────────────

@tool
def navigate_to(url: str) -> str:
    """Navigate the browser to a URL. Always include https://. Use as first step."""
    async def _inner():
        page = await get_page()
        try:
            await page.goto(url, wait_until="domcontentloaded", timeout=30000)
            title   = await page.title()
            current = page.url
            return f"✓ Navigated to: {current}\nPage title: '{title}'"
        except Exception as e:
            return f"✗ Navigate failed: {str(e)}"
    return _run(_inner())


@tool
def click_element(selector: str) -> str:
    """
    Click an element. Selector formats:
      CSS:   'button.submit', '#login-btn'
      Text:  'text=Sign In'
      Role:  'role=button[name=Submit]'
    Prefer text= selectors — they survive UI redesigns.
    """
    async def _inner():
        page = await get_page()
        try:
            await page.click(selector, timeout=10000)
            await page.wait_for_load_state("domcontentloaded")
            return f"✓ Clicked: '{selector}'"
        except Exception as e:
            return (
                f"✗ Click failed for '{selector}': {str(e)}\n"
                f"Try: text=ButtonLabel  or  inspect page with extract_text()"
            )
    return _run(_inner())


@tool
def fill_input(selector: str, text: str) -> str:
    """
    Clear a form field and type text.
    selector examples: 'input[name=email]', 'placeholder=Search', 'label=Username'
    After filling a form, click the submit button separately.
    """
    async def _inner():
        page = await get_page()
        try:
            await page.fill(selector, text, timeout=10000)
            return f"✓ Filled '{selector}' with: '{text}'"
        except Exception as e:
            return (
                f"✗ Fill failed for '{selector}': {str(e)}\n"
                f"Try: placeholder=...  or  label=..."
            )
    return _run(_inner())


@tool
def extract_text(selector: str = "body") -> str:
    """
    Extract visible text from a page element.
    selector: CSS selector. Default 'body' = full page.
    Examples: 'h1', '#results', 'table', '.article-body'
    Returns max 3000 characters.
    """
    async def _inner():
        page = await get_page()
        try:
            element = await page.query_selector(selector)
            if not element:
                # Selector not found — show a page preview to help the LLM recover
                preview = await page.inner_text("body")
                return (
                    f"✗ No element found for: '{selector}'\n"
                    f"Page preview (500 chars):\n{preview[:500]}"
                )
            text = await element.inner_text()
            truncated = len(text) > 3000
            return text[:3000] + ("\n...[truncated]" if truncated else "")
        except Exception as e:
            return f"✗ Extract failed: {str(e)}"
    return _run(_inner())


@tool
def take_screenshot() -> str:
    """
    Capture the current browser page as PNG.
    Use after every navigate_to() and whenever you're unsure what's on screen.
    """
    async def _inner():
        page = await get_page()
        path = SCREENSHOT_DIR / "latest.png"
        try:
            await page.screenshot(path=str(path), full_page=False)
            with open(path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            return (
                f"✓ Screenshot saved.\n"
                f"URL: {page.url}\n"
                f"[img:base64:{b64[:80]}...]"
            )
        except Exception as e:
            return f"✗ Screenshot failed: {str(e)}"
    return _run(_inner())


@tool
def scroll_page(direction: str = "down") -> str:
    """Scroll page up or down by ~600px. Use when buttons or content are off-screen."""
    async def _inner():
        page = await get_page()
        delta = 600 if direction == "down" else -600
        await page.evaluate(f"window.scrollBy(0, {delta})")
        await asyncio.sleep(0.3)
        return f"✓ Scrolled {direction}"
    return _run(_inner())


@tool
def wait_for_element(selector: str) -> str:
    """Wait up to 10s for an element to appear. Use after actions that trigger page changes."""
    async def _inner():
        page = await get_page()
        try:
            await page.wait_for_selector(selector, timeout=10000)
            return f"✓ Element appeared: '{selector}'"
        except Exception:
            return f"✗ Timeout: '{selector}' not found in 10s. Try take_screenshot() to check state."
    return _run(_inner())


@tool
def go_back() -> str:
    """Press browser back button."""
    async def _inner():
        page = await get_page()
        await page.go_back(wait_until="domcontentloaded")
        return f"✓ Went back. Now on: '{await page.title()}' ({page.url})"
    return _run(_inner())


BROWSER_TOOLS = [
    navigate_to,
    click_element,
    fill_input,
    extract_text,
    take_screenshot,
    scroll_page,
    wait_for_element,
    go_back,
]



async def manual_test():
    print("\n--- Wikipedia Tool Test ---\n")

    page = await get_page()

    # 1. Navigate
    print("1. navigate_to → wikipedia.org")
    await page.goto(
        "https://www.wikipedia.org",
        wait_until="domcontentloaded"
    )

    print(f"   Title: {await page.title()}")

    # 2. Extract text
    print("\n2. extract_text")
    try:
        heading = await page.locator(".central-textlogo").text_content()
        print(f"   {heading}")
    except Exception as e:
        print(f"   Failed: {e}")

    # 3. Screenshot
    print("\n3. take_screenshot")
    path = SCREENSHOT_DIR / "wikipedia_home.png"
    await page.screenshot(path=str(path))
    print(f"   Saved: {path}")

    # 4. Fill search input
    print("\n4. fill_input")
    try:
        await page.fill(
            'input[name="search"]',
            "Python"
        )
        print("   Search filled ✓")
    except Exception as e:
        print(f"   Failed: {e}")

    # 5. Press Enter
    print("\n5. submit search")
    try:
        await page.press(
            'input[name="search"]',
            "Enter"
        )
        await page.wait_for_load_state("domcontentloaded")
        print(f"   URL: {page.url}")
    except Exception as e:
        print(f"   Failed: {e}")

    # 6. Extract article heading
    print("\n6. extract article title")
    try:
        article_title = await page.locator("#firstHeading").text_content()
        print(f"   {article_title}")
    except Exception as e:
        print(f"   Failed: {e}")

    # 7. Scroll
    print("\n7. scroll_page")
    await page.evaluate("window.scrollBy(0, 1000)")
    print("   Scrolled ✓")

    # 8. Click first internal link
    print("\n8. click_element")
    try:
        link = page.locator("#mw-content-text a").first
        await link.click()
        await page.wait_for_load_state("domcontentloaded")
        print(f"   Opened: {page.url}")
    except Exception as e:
        print(f"   Failed: {e}")

    # 9. Go back
    print("\n9. go_back")
    await page.go_back(wait_until="domcontentloaded")
    print(f"   Back: {page.url}")

    await close_browser()

    print("\n✅ Wikipedia test completed.\n")

if __name__ == "__main__":
    asyncio.run(manual_test())