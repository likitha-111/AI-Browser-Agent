# AI Browser Agent

An AI-powered agent that controls a real Chromium browser to complete tasks from plain English instructions. No hardcoded scripts. No predefined selectors. Just describe what you want done.

```
"Search Artificial Intelligence in Wikipedia and get the first paragraph of it"
"Get top 5 news title from ynews combinator"
"search for Playwright Python in duckduckgo"
"Find the latest LangGraph release version on GitHub"
```

---

## What it does

- Opens websites and navigates multi-step flows
- Fills and submits forms
- Extracts and summarises page content
- Recovers from failed selectors automatically
- Learns from past runs — repeats tasks faster over time
- Streams live step-by-step progress to the UI

---

## Architecture

```
User (natural language task)
        │
        ▼
  Streamlit UI  ──────────────────────────────────────────────────────┐
  (ui/app.py)                                                         │
        │  POST /run-task/stream (SSE)                                │
        ▼                                                             │
  FastAPI Backend                                                     │
  (api/main.py)                                                       │  screenshot
        │                                                             │  refresh
        ▼                                                             │
  LangGraph ReAct Loop  ◄──── ChromaDB (past lessons) ────────────────┘
  (agent/graph.py)
        │
        │  tool calls
        ▼
  Playwright Browser Tools     ← thread-isolated, ProactorEventLoop
  (agent/tools.py)
        │
        ▼
  Chromium Browser
```

**ReAct loop:** the LLM reasons → picks a tool → observes the result → repeats until the task is complete.

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq (`openai/gpt-oss-120b`) |
| Agent framework | LangGraph |
| LLM integration | LangChain + `langchain-groq` |
| Browser | Playwright (async, Chromium) |
| Long-term memory | ChromaDB |
| API | FastAPI + uvicorn |
| UI | Streamlit |
| Package manager | `uv` |

---

## Project structure

```
ai_browser_agent/
├── agent/
│   ├── __init__.py
│   ├── tools.py       # 9 Playwright browser tools (thread-isolated)
│   ├── graph.py       # LangGraph ReAct agent loop
│   ├── memory.py      # Session memory + ChromaDB long-term memory
│   └── prompts.py     # System prompt with task-type detection
├── api/
│   ├── __init__.py
│   └── main.py        # FastAPI endpoints + SSE streaming
├── ui/
│   ├── __init__.py
│   └── app.py         # Streamlit frontend
├── screenshots/       # Latest browser screenshot (auto-created)
├── memory_store/      # ChromaDB persistent lessons (auto-created)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Clone and create virtual environment

```bash
git clone <your-repo-url>
cd ai_browser_agent
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Configure environment

```bash
cp .env.example .env
```

Open `.env` and set your Groq API key:

Get a free Groq API key at [console.groq.com](https://console.groq.com).


### 4. Start the API

```bash
uvicorn api.main:app --reload --port 8000
```

API docs available at `http://localhost:8000/docs`.

### 5. Start the UI (new terminal)

```bash
streamlit run ui/app.py
```

Opens at `http://localhost:8501`.

---

## API endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/run-task` | Run a task, return when complete |
| `POST` | `/run-task/stream` | Run a task, stream steps via SSE |
| `GET` | `/screenshot` | Latest browser screenshot (PNG) |
| `GET` | `/memory/stats` | Number of lessons stored |
| `DELETE` | `/memory` | Clear all stored lessons |

---

## Browser tools

| Tool | What it does |
|---|---|
| `navigate_to(url)` | Open a URL. Always the first step. |
| `take_screenshot()` | Capture the current page as PNG. |
| `fill_input(selector, text)` | Clear a field and type text into it. |
| `press_key(selector, key)` | Press Enter, Tab, Escape on an element. |
| `click_element(selector)` | Click a button or link. |
| `extract_text(selector)` | Read visible text from an element. |
| `scroll_page(direction)` | Scroll up or down by 600px. |
| `wait_for_element(selector)` | Wait up to 10s for an element to appear. |
| `go_back()` | Browser back button. |

---

This repository was created on HuggingFace Spaces.

---

title: Ai Browser Agent
emoji: 📚
colorFrom: pink
colorTo: purple
sdk: docker
pinned: false

---