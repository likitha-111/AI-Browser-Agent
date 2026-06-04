import base64
import json
import time
import requests
import streamlit as st

st.set_page_config(
    page_title="AI Browser Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

API_BASE = "http://localhost:8000"

st.markdown("""
<style>
.step-log {
    background: #0d1117;
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 14px 16px;
    font-family: 'Consolas', 'Fira Code', monospace;
    font-size: 12px;
    line-height: 1.75;
    height: 440px;
    overflow-y: auto;
    color: #c9d1d9;
    white-space: pre-wrap;
    word-break: break-word;
}
.step-log .nav   { color: #79c0ff; }
.step-log .fill  { color: #a5d6ff; }
.step-log .ext   { color: #7ee787; }
.step-log .shot  { color: #d2a8ff; }
.step-log .ok    { color: #56d364; }
.step-log .fail  { color: #f85149; }
.step-log .empty { color: #e3b341; }
.step-log .done  { color: #ffa657; font-weight: 600; }
.step-log .start { color: #8b949e; font-style: italic; }
.result-box {
    background: #161b22;
    border: 1px solid #30363d;
    border-left: 4px solid #388bfd;
    border-radius: 0 8px 8px 0;
    padding: 16px 18px;
    font-size: 14px;
    line-height: 1.85;
    color: #e6edf3;
    white-space: pre-wrap;
    max-height: 440px;
    overflow-y: auto;
    min-height: 100px;
}
.badge-run  { background:#e3b341;color:#000;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
.badge-done { background:#2ea043;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
.badge-err  { background:#da3633;color:#fff;padding:3px 10px;border-radius:12px;font-size:12px;font-weight:600; }
.badge-idle { background:#30363d;color:#8b949e;padding:3px 10px;border-radius:12px;font-size:12px; }
</style>
""", unsafe_allow_html=True)


defaults = {
    "task_input": "",  
    "result":     "",
    "task_log":   [],
    "steps":      0,
    "status":     "idle",
    "screenshot": None,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def api_health() -> bool:
    try:
        return requests.get(f"{API_BASE}/health", timeout=2).ok
    except Exception:
        return False

def get_memory_count() -> int:
    try:
        r = requests.get(f"{API_BASE}/memory/stats", timeout=2)
        return r.json().get("total_lessons", 0) if r.ok else 0
    except Exception:
        return 0

def fetch_screenshot() -> str | None:
    try:
        r = requests.get(f"{API_BASE}/screenshot?t={int(time.time())}", timeout=3)
        if r.ok:
            return base64.b64encode(r.content).decode()
    except Exception:
        pass
    return None

def colorize(line: str) -> str:
    if   "[Start]"      in line: return f'<span class="start">{line}</span>'
    elif "[Done]"       in line: return f'<span class="done">{line}</span>'
    elif "[Error]"      in line: return f'<span class="fail">{line}</span>'
    elif "[Obs] ✓"      in line: return f'<span class="ok">{line}</span>'
    elif "[Obs] ✗"      in line: return f'<span class="fail">{line}</span>'
    elif "[EMPTY"       in line: return f'<span class="empty">{line}</span>'
    elif "navigate_to"  in line: return f'<span class="nav">{line}</span>'
    elif "extract_text" in line: return f'<span class="ext">{line}</span>'
    elif "screenshot"   in line.lower(): return f'<span class="shot">{line}</span>'
    elif "fill_input"   in line or "press_key" in line: return f'<span class="fill">{line}</span>'
    return line

def render_log(ph, lines: list[str]):
    html = "<br>".join(colorize(l) for l in lines) if lines \
           else '<span class="start">Waiting for task…</span>'
    ph.markdown(f'<div class="step-log">{html}</div>', unsafe_allow_html=True)

def render_screen(ph, b64: str | None):
    if b64:
        ph.markdown(
            f'<img src="data:image/png;base64,{b64}" '
            f'style="width:100%;border-radius:8px;border:1px solid #30363d;display:block;"/>',
            unsafe_allow_html=True,
        )
    else:
        ph.markdown(
            '<div style="height:220px;background:#0d1117;border:1px solid #30363d;'
            'border-radius:8px;display:flex;align-items:center;justify-content:center;'
            'color:#8b949e;font-size:13px;">No screenshot yet</div>',
            unsafe_allow_html=True,
        )

def render_result(ph, text: str, status: str):
    if not text:
        ph.markdown(
            '<div class="result-box" style="color:#8b949e;font-style:italic;">'
            'Result will appear here</div>',
            unsafe_allow_html=True,
        )
        return
    border = "#da3633" if status == "error" else "#388bfd"
    ph.markdown(
        f'<div class="result-box" style="border-left-color:{border};">{text}</div>',
        unsafe_allow_html=True,
    )


with st.sidebar:
    st.markdown("## 🤖 AI Browser Agent")
    st.divider()

    alive = api_health()
    if alive:
        st.success("API connected ✅") 
    else:
        st.error("API offline ❌")
    if not alive:
        st.code("uvicorn api.main:app --reload --port 8000")

    st.metric("Lessons in memory", get_memory_count())
    if st.button("🗑️ Clear memory", use_container_width=True):
        try:
            requests.delete(f"{API_BASE}/memory", timeout=5)
            st.success("Cleared")
            st.rerun()
        except Exception:
            st.error("Failed")

    st.divider()
    st.markdown("### 💡 Examples")

    examples = [
        "Go to news.ycombinator.com and extract the top 5 news titles",
        "Go to github.com/langchain-ai/langgraph/releases and find the latest release version",
        "Go to wikipedia.org, search for 'Python programming language', extract the first paragraph",
        "Go to github.com/trending and list the top 3 trending repositories today",
        "Go to pypi.org/project/playwright and find the latest version number",
    ]
    for ex in examples:
        short = ex[:54] + "…" if len(ex) > 54 else ex
        if st.button(short, key=f"ex_{hash(ex)}", use_container_width=True):
            st.session_state["task_input"] = ex
            st.rerun()

    st.divider()
    st.caption("Groq · LangGraph · Playwright · ChromaDB")


st.markdown("# AI Browser Agent")
st.markdown("Control a real Chromium browser with plain English.")

col_task, col_run = st.columns([5, 1])
with col_task:
    st.text_area(
        "Task",
        height=80,
        placeholder="e.g. Go to news.ycombinator.com and extract the top 5 news titles",
        label_visibility="collapsed",
        key="task_input",
    )

with col_run:
    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    run_btn = st.button(
        "▶ Run", type="primary", use_container_width=True,
        disabled=not alive or st.session_state["status"] == "running",
    )

st.divider()

col_log, col_screen, col_result = st.columns([2, 2, 2])
with col_log:
    st.markdown("#### 📋 Step log")
    status_ph = st.empty()
    log_ph    = st.empty()
    steps_ph  = st.empty()
with col_screen:
    st.markdown("#### 🖥️ Browser")
    screen_ph = st.empty()
with col_result:
    st.markdown("#### ✅ Result")
    result_ph = st.empty()

status_map = {
    "idle":    '<span class="badge-idle">Idle</span>',
    "running": '<span class="badge-run">⏳ Running…</span>',
    "done":    '<span class="badge-done">✅ Done</span>',
    "error":   '<span class="badge-err">❌ Error</span>',
}
status_ph.markdown(status_map[st.session_state["status"]], unsafe_allow_html=True)
render_log(log_ph, st.session_state["task_log"])
render_screen(screen_ph, st.session_state["screenshot"])
render_result(result_ph, st.session_state["result"], st.session_state["status"])
if st.session_state["steps"]:
    steps_ph.caption(f"{st.session_state['steps']} steps")


if run_btn:
    task = st.session_state["task_input"].strip()

    if not task:
        st.warning("Please enter a task first.")
        st.stop()

    st.session_state["result"]     = ""
    st.session_state["task_log"]   = [f"[Start] {task}"]
    st.session_state["steps"]      = 0
    st.session_state["status"]     = "running"
    st.session_state["screenshot"] = None

    status_ph.markdown(status_map["running"], unsafe_allow_html=True)
    render_log(log_ph, st.session_state["task_log"])
    render_screen(screen_ph, None)
    render_result(result_ph, "", "running")

    errored = False

    try:
        with requests.post(
            f"{API_BASE}/run-task/stream",
            json={"task": task},
            stream=True,
            timeout=180,
        ) as resp:
            if not resp.ok:
                raise RuntimeError(f"API error {resp.status_code}: {resp.text[:200]}")

            for raw in resp.iter_lines():
                if not raw or not raw.startswith(b"data: "):
                    continue
                try:
                    event = json.loads(raw[6:])
                except json.JSONDecodeError:
                    continue

                etype = event.get("type", "")
                msg   = event.get("message", "")

                if etype == "log":
                    st.session_state["task_log"].append(msg)
                    render_log(log_ph, st.session_state["task_log"])

                    if msg.startswith("[Step"):
                        n = len([l for l in st.session_state["task_log"] if l.startswith("[Step")])
                        steps_ph.caption(f"Step {n} running…")

                    if msg.startswith("[Obs]") or "screenshot" in msg.lower():
                        b64 = fetch_screenshot()
                        if b64:
                            st.session_state["screenshot"] = b64
                            render_screen(screen_ph, b64)

                elif etype == "result":
                    st.session_state["result"] = msg
                    st.session_state["steps"]  = event.get("steps", 0)
                    st.session_state["status"] = "done"
                    st.session_state["task_log"].append(
                        f"[Done] Completed in {st.session_state['steps']} steps"
                    )
                    render_log(log_ph, st.session_state["task_log"])
                    b64 = fetch_screenshot()
                    if b64:
                        st.session_state["screenshot"] = b64
                        render_screen(screen_ph, b64)
                    break

                elif etype == "error":
                    errored = True
                    st.session_state["result"] = f"Error: {msg}"
                    st.session_state["status"] = "error"
                    st.session_state["task_log"].append(f"[Error] {msg}")
                    render_log(log_ph, st.session_state["task_log"])
                    break

    except requests.exceptions.Timeout:
        errored = True
        st.session_state["result"] = "Timeout — agent took longer than 3 minutes"
        st.session_state["status"] = "error"
    except Exception as e:
        errored = True
        st.session_state["result"] = str(e)
        st.session_state["status"] = "error"

    final_status = "error" if errored else "done"
    st.session_state["status"] = final_status
    status_ph.markdown(status_map[final_status], unsafe_allow_html=True)
    steps_ph.caption(f"{st.session_state['steps']} steps total")
    render_result(result_ph, st.session_state["result"], final_status)