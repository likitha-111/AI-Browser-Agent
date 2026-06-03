import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

from agent.tools import BROWSER_TOOLS, close_browser

class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    task:     str
    task_log: list[str]
    steps:    int


SYSTEM_PROMPT = """You are an AI browser agent controlling a real Chromium browser.

## Your tools
- navigate_to(url)              → open a URL (always start here)
- click_element(selector)       → click buttons/links. Prefer text= selectors
- fill_input(selector, text)    → type into form fields
- press_key(selector, key)      → press Enter/Tab/Escape on an element
- extract_text(selector)        → read page content (default: 'body')
- take_screenshot()             → capture current page — call with no arguments
- scroll_page(direction)        → scroll 'up' or 'down'
- wait_for_element(selector)    → wait for element to appear
- go_back()                     → browser back button

## Workflow
1. navigate_to() the target URL
2. take_screenshot() to confirm the page loaded — pass empty args {{}}
3. Perform required actions one step at a time
4. extract_text() to get the required information
5. Give a clear final summary of what was found or done

## Rules
- Call ONE tool per response
- For take_screenshot, always call it with no arguments
- If a selector fails try: text=Label, then CSS, then a simpler selector
- Never assume page state — verify with extract_text or take_screenshot
- Stop and explain if stuck after 3 retries on the same step

## Task
{task}
"""

MAX_STEPS = 25


def llm_node(state: AgentState) -> dict:
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,
    ).bind_tools(BROWSER_TOOLS)

    system   = SystemMessage(content=SYSTEM_PROMPT.format(task=state["task"]))
    response = llm.invoke([system] + state["messages"])

    if response.tool_calls:
        tc  = response.tool_calls[0]
        log = f"[Step {state['steps']+1}] {tc['name']}({tc['args']})"
    else:
        log = f"[Done] {response.content[:300]}"

    return {
        "messages": [response],
        "task_log": state["task_log"] + [log],
        "steps":    state["steps"] + 1,
    }


def tool_node_fn(state: AgentState) -> dict:
    executor = ToolNode(BROWSER_TOOLS)
    result   = executor.invoke(state)
    msgs     = result.get("messages", [])
    obs      = msgs[-1].content if msgs else "(no output)"
    return {
        "messages": msgs,
        "task_log": state["task_log"] + [f"[Obs] {str(obs)[:300]}"],
    }


def should_continue(state: AgentState) -> str:
    if state["steps"] >= MAX_STEPS:
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
    graph = build_graph()
    initial = {
        "messages": [HumanMessage(content=task)],
        "task":     task,
        "task_log": [f"[Start] {task}"],
        "steps":    0,
    }
    try:
        final  = graph.invoke(initial, config={"recursion_limit": MAX_STEPS + 5})
        result = ""
        for msg in reversed(final["messages"]):
            if isinstance(msg, AIMessage) and msg.content and not msg.tool_calls:
                result = msg.content
                break
        return {
            "task":     task,
            "result":   result or "Task completed.",
            "task_log": final["task_log"],
            "steps":    final["steps"],
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return {
            "task":     task,
            "result":   f"Error: {e}",
            "task_log": [f"[Error] {tb}"],
            "steps":    0,
        }
    finally:
        close_browser()


if __name__ == "__main__":
    task = "Go to wikipedia.org, search for 'Artificial Intelligence', and extract the first paragraph."
    print(f"\n{'='*60}\nTask: {task}\n{'='*60}\n")
    result = run_task(task)
    print("\n--- Step log ---")
    for s in result["task_log"]:
        print(s)
    print(f"\n--- Final result ({result['steps']} steps) ---")
    print(result["result"])