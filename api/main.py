import json
import os
import sys
import traceback
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

load_dotenv()
sys.path.insert(0, str(Path(__file__).parent.parent))

app = FastAPI(
    title="AI Browser Agent API",
    description="Control a real browser with natural language",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCREENSHOT_PATH = Path("./screenshots/latest.png")


class TaskRequest(BaseModel):
    task: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"task": "Go to wikipedia.org and search for Artificial Intelligence"}
            ]
        }
    }


class TaskResponse(BaseModel):
    task:     str
    result:   str
    task_log: list[str]
    steps:    int
    status:   str = "success"


class MemoryStats(BaseModel):
    total_lessons: int
    storage_path:  str



@app.get("/")
def root():
    return {
        "name":    "AI Browser Agent API",
        "version": "1.0.0",
        "docs":    "/docs",
        "status":  "running",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "model": os.getenv("LLM_MODEL", "openai/gpt-oss-120b")}


@app.post("/run-task", response_model=TaskResponse)
def run_task_endpoint(req: TaskRequest):
    """
    Run a browser task synchronously.
    Blocks until the agent finishes (can take 30–120s for complex tasks).
    Use /run-task/stream for a live step-by-step view.
    """
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")
    try:
        from agent.graph import run_task
        result = run_task(req.task)
        status = "error" if result["result"].startswith("Error:") else "success"
        return TaskResponse(
            task=result["task"],
            result=result["result"],
            task_log=result["task_log"],
            steps=result["steps"],
            status=status,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-task/stream")
def run_task_stream(req: TaskRequest):
    """
    Run a browser task and stream progress via Server-Sent Events.

    SSE event format:
      data: {"type": "log",    "message": "[Step 1] navigate_to(...)"}
      data: {"type": "log",    "message": "[Obs] ✓ Navigated to ..."}
      data: {"type": "result", "message": "Final answer...", "steps": 7, "done": true}
      data: {"type": "error",  "message": "...", "done": true}

    Streamlit polls this endpoint and renders each event in real time.
    """
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")
    def event_stream():
        try:

            import queue as q_mod
            import threading
            step_queue: q_mod.Queue = q_mod.Queue()
            result_holder = {}

            def _patched_run():
                """Run the agent in a thread; push each log line to step_queue."""
                from agent import graph as graph_module
                original_llm  = graph_module.llm_node
                original_tool = graph_module.tool_node_fn

                def patched_llm(state):
                    out = original_llm(state)
                    for entry in out.get("task_log", [])[len(state["task_log"]):]:
                        step_queue.put({"type": "log", "message": entry})
                    return out

                def patched_tool(state):
                    out = original_tool(state)
                    for entry in out.get("task_log", [])[len(state["task_log"]):]:
                        step_queue.put({"type": "log", "message": entry})
                    return out

                graph_module.llm_node   = patched_llm
                graph_module.tool_node_fn = patched_tool

                try:
                    from agent.graph import run_task
                    result = run_task(req.task)
                    result_holder["data"] = result
                except Exception as e:
                    result_holder["error"] = str(e)
                finally:
                    graph_module.llm_node     = original_llm
                    graph_module.tool_node_fn = original_tool
                    step_queue.put(None)

            t = threading.Thread(target=_patched_run, daemon=True)
            t.start()

            while True:
                try:
                    item = step_queue.get(timeout=120)
                except q_mod.Empty:
                    yield f"data: {json.dumps({'type': 'error', 'message': 'Timeout', 'done': True})}\n\n"
                    break

                if item is None:
                    if "error" in result_holder:
                        payload = {
                            "type":    "error",
                            "message": result_holder["error"],
                            "done":    True,
                        }
                    else:
                        r = result_holder.get("data", {})
                        payload = {
                            "type":    "result",
                            "message": r.get("result", ""),
                            "steps":   r.get("steps", 0),
                            "done":    True,
                        }
                    yield f"data: {json.dumps(payload)}\n\n"
                    break
                else:
                    yield f"data: {json.dumps(item)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e), 'done': True})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
@app.get("/screenshot")
def get_screenshot():
    if not SCREENSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="No screenshot captured yet")
    return FileResponse(
        str(SCREENSHOT_PATH),
        media_type="image/png",
        headers={"Cache-Control": "no-cache"},
    )
@app.get("/memory/stats", response_model=MemoryStats)
def memory_stats():
    """How many lessons are stored in ChromaDB."""
    from agent.memory import longterm_memory
    return MemoryStats(
        total_lessons=longterm_memory.count(),
        storage_path=os.getenv("CHROMA_PERSIST_DIR", "./memory_store"),
    )


@app.delete("/memory")
def clear_memory():
    from agent.memory import longterm_memory
    longterm_memory.clear_all()
    return {"status": "cleared", "lessons_remaining": 0}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", "8000")),
        reload=True,
    )