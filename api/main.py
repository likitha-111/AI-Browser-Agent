import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from agent.graph import run_task

app = FastAPI(title="AI Browser Agent API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SCREENSHOT_PATH = Path("./screenshots/latest.png")


class TaskRequest(BaseModel):
    task: str


class TaskResponse(BaseModel):
    task:     str
    result:   str
    task_log: list[str]


@app.get("/")
def root():
    return {"status": "ok", "message": "AI Browser Agent is running"}


@app.post("/run-task", response_model=TaskResponse)
def run_task_endpoint(req: TaskRequest):
    if not req.task.strip():
        raise HTTPException(status_code=400, detail="Task cannot be empty")
    try:
        result = run_task(req.task)
        return TaskResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/run-task/stream")
def run_task_stream(req: TaskRequest):
    """Stream task logs line-by-line as the agent works."""
    def event_generator():
        # This is a simplified streamer — Phase 5 will make this truly async
        try:
            result = run_task(req.task)
            for step in result["task_log"]:
                yield f"data: {json.dumps({'log': step})}\n\n"
            yield f"data: {json.dumps({'result': result['result'], 'done': True})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/screenshot")
def get_screenshot():
    if not SCREENSHOT_PATH.exists():
        raise HTTPException(status_code=404, detail="No screenshot yet")
    return FileResponse(str(SCREENSHOT_PATH), media_type="image/png")


@app.get("/health")
def health():
    return {"status": "healthy"}
