import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent.tools import BROWSER_TOOLS, close_browser, init_browser
from agent.memory import recall_similar_tasks, reset_session, save_task_to_memory
from agent.prompts import SYSTEM_PROMPT

load_dotenv()

class AgentState(TypedDict):
    messages:   Annotated[list, add_messages]
    task:       str
    task_log:   list[str] 
    screenshot: str


def _build_llm():
    return ChatGroq(
        model=os.getenv("LLM_MODEL", "llama-3.3-70b-versatile"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
        api_key=os.getenv("GROQ_API_KEY"),
    ).bind_tools(BROWSER_TOOLS)


async def llm_node(state: AgentState) -> AgentState:
    llm = _build_llm()
    response = await llm.ainvoke(state["messages"])
    print("\n=== LLM RESPONSE ===")
    print(response)
    print("====================\n")
    if response.tool_calls:
        log_entry = f"[Tool Call] {response.tool_calls}"
    else:
        log_entry = f"[Agent] {response.content}"
    return {
        "messages":   [response],
        "task_log":   state["task_log"] + [log_entry],
        "screenshot": state["screenshot"],
    }


from langchain_core.messages import ToolMessage

TOOL_MAP = {
    tool.name: tool
    for tool in BROWSER_TOOLS
}

async def sequential_tools(state: AgentState):
    last_message = state["messages"][-1]

    outputs = []

    for tool_call in last_message.tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        print(f"\n=== EXECUTING TOOL ===")
        print(tool_name)
        print(tool_args)

        tool = TOOL_MAP[tool_name]

        result = await tool.ainvoke(tool_args)

        outputs.append(
            ToolMessage(
                content=str(result),
                tool_call_id=tool_call["id"],
            )
        )

    return {
        "messages": outputs
    }


def should_continue(state: AgentState) -> str:
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return END


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("llm", llm_node)
    g.add_node("tools", sequential_tools)
    g.set_entry_point("llm")
    g.add_conditional_edges("llm", should_continue, {"tools": "tools", END: END})
    g.add_edge("tools", "llm")
    return g.compile()


def run_task(task: str) -> dict:
    """Run a natural-language task and return the result dict."""
    import asyncio

    async def _run():
        reset_session()
        memory_context = recall_similar_tasks(task)
        system_prompt = SYSTEM_PROMPT.format(memory_context=memory_context)

        from langchain_core.messages import HumanMessage, SystemMessage

        initial_state: AgentState = {
            "messages": [
                SystemMessage(content=system_prompt),
                HumanMessage(content=task)
            ],
            "task": task,
            "task_log": [f"[Start] Task: {task}"],
            "screenshot": "",
        }

        graph = build_graph()

        await init_browser()

        try:
            final_state = await graph.ainvoke(
                initial_state,
                {"recursion_limit": 40}
            )

            result_text = final_state["messages"][-1].content

            save_task_to_memory(
                task,
                final_state["task_log"],
                result_text
            )

            return {
                "task": task,
                "result": result_text,
                "task_log": final_state["task_log"],
                "screenshot": final_state["screenshot"],
            }

        finally:
            await close_browser()

    return asyncio.run(_run())
