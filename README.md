# AI Browser Agent

An AI agent that controls a real browser to complete tasks from natural language instructions.

## Stack
| Layer | Technology |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) |
| Agent | LangGraph ReAct loop |
| Browser | Playwright (Chromium) |
| Memory | LangChain in-memory + ChromaDB |
| API | FastAPI |
| UI | Streamlit |

## Quickstart

### 1. Clone and create venv
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
playwright install chromium
```

### 3. Set up environment
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY
```

### 4. Run Phase 1 sanity checks
```bash
python tests/test_tools.py
```

### 5. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```

### 6. Start the UI (in a second terminal)
```bash
streamlit run ui/app.py
```

## Project structure
```
ai_browser_agent/
├── agent/
│   ├── tools.py      # 8 Playwright browser tools
│   ├── graph.py      # LangGraph ReAct agent
│   ├── memory.py     # Session + ChromaDB memory
│   └── prompts.py    # System prompt
├── api/
│   └── main.py       # FastAPI endpoints
├── ui/
│   └── app.py        # Streamlit interface
├── tests/
│   └── test_tools.py # Phase 1 sanity checks
├── memory_store/     # ChromaDB persistent data
├── screenshots/      # Browser screenshots
├── .env.example
└── requirements.txt
```

## Example tasks
- "Go to news.ycombinator.com and extract the top 5 headlines"
- "Search for 'LangGraph tutorial' on Google and list the first 3 results"
- "Go to wikipedia.org and find the summary of Artificial intelligence"
