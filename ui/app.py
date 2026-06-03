import time

import requests
import streamlit as st

API_BASE = "http://localhost:8000"

st.set_page_config(
    page_title="AI Browser Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("AI Browser Agent")
st.caption("Type a task in plain English — the agent will control a browser to complete it.")


with st.sidebar:
    st.header("Example tasks")
    examples = [
        "Go to news.ycombinator.com and extract the top 5 headlines",
        "Search for 'LangGraph tutorial' on Google and list the first 3 results",
        "Go to wikipedia.org and find the summary of 'Artificial intelligence'",
        "Open github.com/trending and extract the top 3 trending repos",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["task_input"] = ex

    st.divider()
    st.header("API status")
    try:
        r = requests.get(f"{API_BASE}/health", timeout=2)
        st.success("API connected") if r.ok else st.error("API error")
    except Exception:
        st.error("API not reachable — start FastAPI first")


col_input, col_log, col_screen = st.columns([2, 3, 2])

with col_input:
    st.subheader("Task")
    task = st.text_area(
        "What should the agent do?",
        value=st.session_state.get("task_input", ""),
        height=120,
        key="task_input",
        placeholder="e.g. Go to google.com and search for AI news",
    )
    run = st.button("Run agent", type="primary", use_container_width=True)

with col_log:
    st.subheader("Step log")
    log_box = st.empty()

with col_screen:
    st.subheader("Browser screenshot")
    screen_box = st.empty()

if run and task.strip():
    steps = []
    log_box.info("Agent starting...")

    with st.spinner("Agent working..."):
        try:
            with requests.post(
                f"{API_BASE}/run-task/stream",
                json={"task": task},
                stream=True,
                timeout=120,
            ) as resp:
                import json as _json
                for line in resp.iter_lines():
                    if line and line.startswith(b"data: "):
                        data = _json.loads(line[6:])
                        if "log" in data:
                            steps.append(data["log"])
                            log_box.code("\n".join(steps), language=None)
                        if "error" in data:
                            st.error(data["error"])
                        if data.get("done"):
                            st.success("Task complete!")
                            st.markdown("**Result:**")
                            st.write(data.get("result", ""))
                    try:
                        r = requests.get(f"{API_BASE}/screenshot", timeout=2)
                        if r.ok:
                            screen_box.image(r.content, use_container_width=True)
                    except Exception:
                        pass

        except requests.exceptions.ConnectionError:
            st.error("Cannot connect to API. Run: uvicorn api.main:app --reload")
