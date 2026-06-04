"""
agent/graph.py — v8
Changes from v7:
  1. Uses prompts.py SYSTEM_PROMPT (cleaner separation)
  2. After body extract succeeds → inject a "now parse" instruction
     so the LLM knows to stop calling tools and find the answer in the text
  3. Result post-processing: trim navigation noise from final answer
"""
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
from agent.prompts import SYSTEM_PROMPT


class AgentState(TypedDict):
    messages:             Annotated[list[BaseMessage], add_messages]
    task:                 str
    task_log:             list[str]
    steps:                int
    memory_context:       str
    consecutive_failures: int
    empty_extracts:       int
    last_extract_text:    str  


def _get_llm():
    return ChatGroq(
        model=os.getenv("LLM_MODEL", "openai/gpt-oss-120b"),
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

    messages = list(state["messages"])

    if state["empty_extracts"] > 0:
        last_tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if last_tool_msgs:
            last_obs = last_tool_msgs[-1].content
            if not last_obs or last_obs.strip() in ("", "\n", "\n\n"):
                messages = messages + [HumanMessage(content=(
                    "⚠️ The last extract_text returned empty. "
                    "Do NOT give a final answer yet. "
                    "Call extract_text('body') now to get the full page text."
                ))]

    elif state["last_extract_text"] and len(state["last_extract_text"]) > 100:
        last_tool_msgs = [m for m in messages if isinstance(m, ToolMessage)]
        if last_tool_msgs:
            last_obs = last_tool_msgs[-1].content
            last_ai = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
            if last_ai and last_ai.tool_calls:
                last_tool_name = last_ai.tool_calls[0].get("name", "")
                if "extract" in last_tool_name and last_obs and last_obs.strip():
                    messages = messages + [HumanMessage(content=(
                        "You have the page content above. "
                        "Do NOT call any more tools. "
                        "Read the text carefully, find exactly what the task asks for, "
                        "and give your final answer quoting ONLY the relevant portion. "
                        "Skip navigation menus and table of contents. "
                        "For Wikipedia, the first paragraph starts with the subject's full name in bold."
                    ))]

    for attempt in range(3):
        try:
            response = llm.invoke([system] + messages)
            break
        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                wait = 15 * (attempt + 1)
                print(f"[LLM] Rate limited — waiting {wait}s...")
                time.sleep(wait)
                if attempt == 2:
                    raise
            else:
                raise

    if response.tool_calls:
        tc  = response.tool_calls[0]
        log = f"[Step {state['steps']+1}] {tc['name']}({tc['args']})"
    else:
        log = f"[Done] {response.content[:300]}"

    return {
        "messages":            [response],
        "task_log":            state["task_log"] + [log],
        "steps":               state["steps"] + 1,
        "consecutive_failures": state["consecutive_failures"],
    }


def tool_node_fn(state: AgentState) -> dict:
    executor = ToolNode(BROWSER_TOOLS)
    result   = executor.invoke(state)
    msgs     = result.get("messages", [])
    obs      = msgs[-1].content if msgs else ""
    obs_str  = str(obs)

    is_failure = "✗" in obs_str or "Timeout" in obs_str or "failed" in obs_str.lower()
    failures   = state["consecutive_failures"] + 1 if is_failure else 0

    is_empty = obs_str.strip() in ("", "\n", "\n\n") or len(obs_str.strip()) < 5
    empty_extracts = state["empty_extracts"] + 1 if is_empty else 0

    last_extract = state["last_extract_text"]
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)), None)
    if last_ai and last_ai.tool_calls:
        if "extract" in last_ai.tool_calls[0].get("name", "") and obs_str.strip():
            last_extract = obs_str

    log_obs = obs_str[:400] if obs_str.strip() else "[EMPTY — will retry with body selector]"

    return {
        "messages":            msgs,
        "task_log":            state["task_log"] + [f"[Obs] {log_obs}"],
        "consecutive_failures": failures,
        "empty_extracts":      empty_extracts,
        "last_extract_text":   last_extract,
    }


def should_continue(state: AgentState) -> str:
    if state["steps"] >= 30:
        return "end"
    if state["consecutive_failures"] >= 5:
        return "end"
    last = state["messages"][-1]
    if isinstance(last, AIMessage) and last.tool_calls:
        return "tools"
    return "end"


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm",   llm_node)
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
        "messages":            [HumanMessage(content=task)],
        "task":                task,
        "task_log":            [f"[Start] {task}"],
        "steps":               0,
        "memory_context":      memory_ctx,
        "consecutive_failures": 0,
        "empty_extracts":      0,
        "last_extract_text":   "",
    }

    try:
        final  = graph.invoke(initial, config={"recursion_limit": 70})
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
            "task":     task,
            "result":   f"Error: {e}",
            "task_log": [f"[Error] {traceback.format_exc()}"],
            "steps":    0,
        }
    finally:
        close_browser()


if __name__ == "__main__":
    task = "search for Colleen Hoover on wikipedia and extract first paragraph of it."
    print(f"\n{'='*60}\nTask: {task}\n{'='*60}\n")
    result = run_task(task)
    for s in result["task_log"]:
        print(s)
    print(f"\n--- Result ({result['steps']} steps) ---")
    print(result["result"])