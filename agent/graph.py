import os
import time
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

from agent.tools import BROWSER_TOOLS, close_browser
from agent.memory import get_memory_context, reset_session, save_completed_task, longterm_memory

class AgentState(TypedDict):
    messages:       Annotated[list[BaseMessage], add_messages]
    task:           str
    task_log:       list[str]
    steps:          int
    memory_context: str
    consecutive_failures: int


SYSTEM_PROMPT = """You are an AI browser agent controlling a real Chromium browser.

## Your tools
- navigate_to(url)              → open a URL (always start here)
- click_element(selector)       → click buttons/links. Prefer text= selectors
- fill_input(selector, text)    → type into form fields
- press_key(selector, key)      → press Enter/Tab/Escape on an element
- extract_text(selector)        → read page content (default: 'body')
- take_screenshot()             → capture current page
- scroll_page(direction)        → scroll 'up' or 'down'
- wait_for_element(selector)    → wait for element to appear (use sparingly)
- go_back()                     → browser back button

## Workflow
1. navigate_to() the target URL
2. take_screenshot() to confirm the page loaded
3. Perform required actions one step at a time
4. extract_text('body') as fallback if specific selectors fail
5. Give a clear final summary of what was found or done

## Selector strategy (in order of preference)
1. Use selectors from the lesson below if available
2. Try text= selectors: text=Sign In, text=Search
3. Try simple CSS: input[name=search], button[type=submit]
4. Fall back to extract_text('body') — always works

## IMPORTANT: If extract_text returns content, that IS your answer.
## Do not keep retrying selectors after getting text from 'body'.

## Current task
{task}

{memory_context}
"""

MAX_STEPS = 30
RECURSION_LIMIT = 60 


def _get_llm():
    model = os.getenv("LLM_MODEL", "qwen-qwq-32b")
    return ChatGroq(
        model=model,
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,
    ).bind_tools(BROWSER_TOOLS)


def llm_node(state: AgentState) -> dict:
    llm = _get_llm()

    system = SystemMessage(content=SYSTEM_PROMPT.format(
        task=state["task"],
        memory_context=state["memory_context"],
    ))

    for attempt in range(3):
        try:
            response = llm.invoke([system] + state["messages"])
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 10 * (attempt + 1)
                print(f"[LLM] Rate limited — waiting {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    raise
            else:
                raise

    if response.tool_calls:
        tc  = response.tool_calls[0]
        log = f"[Step {state['steps']+1}] {tc['name']}({tc['args']})"
        failures = state["consecutive_failures"]
    else:
        log = f"[Done] {response.content[:300]}"
        failures = 0

    return {
        "messages":            [response],
        "task_log":            state["task_log"] + [log],
        "steps":               state["steps"] + 1,
        "consecutive_failures": failures,
    }


def tool_node_fn(state: AgentState) -> dict:
    executor = ToolNode(BROWSER_TOOLS)
    result   = executor.invoke(state)
    msgs     = result.get("messages", [])
    obs      = msgs[-1].content if msgs else "(no output)"
    obs_str  = str(obs)[:400]

    is_failure = "✗" in obs_str or "Timeout" in obs_str or "failed" in obs_str.lower()
    failures   = state["consecutive_failures"] + 1 if is_failure else 0

    return {
        "messages": msgs,
        "task_log": state["task_log"] + [f"[Obs] {obs_str}"],
        "consecutive_failures": failures,
    }


def should_continue(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
        print(f"[Agent] Reached MAX_STEPS ({MAX_STEPS}) — stopping")
        return "end"

    if state["consecutive_failures"] >= 4:
        print("[Agent] 4 consecutive failures — stopping to prevent infinite loop")
        return "end"

    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", tool_node_fn)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", "end": END})
    g.add_edge("tools", "llm")
    return g.compile()


def run_task(task: str) -> dict:
    reset_session()

    memory_ctx = get_memory_context(task)
    count      = longterm_memory.count()
    print(f"[Memory] {count} past tasks in DB" + (" — injecting lessons" if memory_ctx else " — starting fresh"))

    graph   = build_graph()
    initial: AgentState = {
        "messages": [HumanMessage(content=task)],
        "task":     task,
        "task_log": [f"[Start] {task}"],
        "steps": 0,
        "memory_context": memory_ctx,
        "consecutive_failures": 0,
    }
    try:
        final  = graph.invoke(initial, config={"recursion_limit": RECURSION_LIMIT})
        result = ""
        for msg in reversed(final["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                result = msg.content
                break

        save_completed_task(task, final["task_log"], result)
        return {
            "task":     task,
            "result":   result or "Task completed.",
            "task_log": final["task_log"],
            "steps":    final["steps"],
        }
    except Exception as e:
        import traceback
        return {
            "task": task,
            "result": f"Error: {e}",
            "task_log": [f"[Error] {traceback.format_exc()}"],
            "steps": 0,
        }
    finally:
        close_browser()


if __name__ == "__main__":
    TASK = "Go to DuckDuckGo and search for Playwright Python."

    print("\n" + "="*60)
    print("RUN 1 — no memory")
    print("="*60)
    r1 = run_task(TASK)
    for s in r1["task_log"]: print(s)
    print(f"\nSteps: {r1['steps']}\nResult: {r1['result'][:300]}\n")

    print("\n" + "="*60)
    print("RUN 2 — with lesson from run 1")
    print("="*60)
    r2 = run_task(TASK)
    for s in r2["task_log"]: print(s)
    print(f"\nSteps: {r2['steps']}\nResult: {r2['result'][:300]}\n")

    print(f"\nStep reduction: {r1['steps']} → {r2['steps']}")
    print(f"Total lessons stored: {longterm_memory.count()}")