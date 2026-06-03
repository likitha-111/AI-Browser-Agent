import os
import re
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./memory_store")


def extract_lessons(task: str, steps: list[str], result: str) -> str:
    """
    Distill a raw step log into a compact lesson focused on what WORKED.
    Filters out failed steps, keeps only the successful action sequence.

    Output format:
        TASK: <task>
        SITE: <domain>
        WORKING STEPS:
          1. navigate_to(url)
          2. fill_input(selector, text)
          ...
        RESULT: <summary>
    """
    working_steps = []
    skip_next_obs = False

    for i, step in enumerate(steps):
        # Skip error observations
        if step.startswith("[Obs] ✗") or "Timeout" in step or "failed" in step.lower():
            # Also remove the preceding [Step] that caused this failure
            if working_steps and working_steps[-1].startswith("[Step"):
                working_steps.pop()
            continue

        # Skip wait_for_element steps — they're noisy and fail often
        if "wait_for_element" in step:
            continue

        # Keep successful steps and observations
        if step.startswith("[Step") or (step.startswith("[Obs] ✓") and "Screenshot" not in step):
            working_steps.append(step)

    # Extract the domain from the task for context
    urls = re.findall(r'https?://[^\s,\)\'\"]+', " ".join(steps))
    domain = urls[0] if urls else "unknown"

    lesson = (
        f"TASK: {task}\n"
        f"SITE: {domain}\n"
        f"WORKING STEPS (use these exact selectors):\n"
        + "\n".join(f"  {s}" for s in working_steps[:15])
        + f"\nRESULT: {result[:300]}"
    )
    return lesson


class SessionMemory:
    def __init__(self):
        self._messages: list[BaseMessage] = []
        self._scratchpad: dict = {}

    def add_human(self, text: str):   self._messages.append(HumanMessage(content=text))
    def add_ai(self, text: str):      self._messages.append(AIMessage(content=text))
    def note(self, k: str, v: str):   self._scratchpad[k] = v
    def get_note(self, k: str):       return self._scratchpad.get(k)
    def get_messages(self):           return list(self._messages)
    def clear(self):                  self._messages.clear(); self._scratchpad.clear()
    def __len__(self):                return len(self._messages)


class LongTermMemory:
    def __init__(self):
        self._client     = None
        self._collection = None

    def _get_collection(self):
        if self._collection is None:
            self._client = chromadb.PersistentClient(path=CHROMA_DIR)
            self._collection = self._client.get_or_create_collection(
                name="browser_agent_lessons_v2", 
                embedding_function=DefaultEmbeddingFunction(),
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    def save(self, task: str, steps: list[str], result: str):
        """Save a distilled lesson (not the raw log)."""
        col    = self._get_collection()
        lesson = extract_lessons(task, steps, result)
        doc_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        col.add(
            documents=[lesson],
            ids=[doc_id],
            metadatas=[{
                "task":      task[:200],
                "timestamp": datetime.now().isoformat(),
            }],
        )
        print(f"[Memory] Lesson saved (id={doc_id})")
        print(f"[Memory] Lesson preview:\n{lesson[:400]}\n")

    def recall(self, task: str, n: int = 2) -> str:
        """Retrieve top-n similar lessons, formatted for the system prompt."""
        col   = self._get_collection()
        total = col.count()
        if total == 0:
            return ""

        results = col.query(query_texts=[task], n_results=min(n, total))
        docs    = results.get("documents", [[]])[0]
        if not docs:
            return ""

        lines = [
            "## Lessons from past similar tasks — USE THESE EXACT SELECTORS:",
            "## Do NOT try other selectors if these are available for this site.\n",
        ]
        for i, doc in enumerate(docs, 1):
            lines.append(f"### Lesson {i}:\n{doc}\n")

        return "\n".join(lines)

    def count(self) -> int:
        return self._get_collection().count()

    def clear_all(self):
        col     = self._get_collection()
        all_ids = col.get()["ids"]
        if all_ids:
            col.delete(ids=all_ids)
        print(f"[Memory] Cleared {len(all_ids)} lessons")


session_memory  = SessionMemory()
longterm_memory = LongTermMemory()


def reset_session():
    session_memory.clear()

def save_completed_task(t, steps, result):
    longterm_memory.save(t, steps, result)
    
def get_memory_context(task):
    return longterm_memory.recall(task)


if __name__ == "__main__":
    raw_steps = [
        "[Start] Go to wikipedia.org, search for AI, extract first paragraph.",
        "[Step 1] navigate_to({'url': 'https://www.wikipedia.org'})",
        "[Obs] ✓ Navigated to: https://www.wikipedia.org/",
        "[Step 2] take_screenshot({})",
        "[Obs] ✓ Screenshot saved.",
        "[Step 3] fill_input({'selector': 'input[name=search]', 'text': 'Artificial Intelligence'})",
        "[Obs] ✓ Filled 'input[name=search]' with: 'Artificial Intelligence'",
        "[Step 4] press_key({'key': 'Enter', 'selector': 'input[name=search]'})",
        "[Obs] ✓ Pressed 'Enter' on 'input[name=search]'.",
        "[Step 5] wait_for_element({'selector': '#mw-content-text .mw-parser-output > p'})",
        "[Obs] ✗ Timeout: '#mw-content-text .mw-parser-output > p' not found.",
        "[Step 6] take_screenshot({})",
        "[Obs] ✓ Screenshot saved.",
        "[Step 7] extract_text({'selector': 'body'})",
        "[Obs] ✓ Jump to content... Artificial intelligence (AI) is...",
        "[Done] First paragraph found.",
    ]

    print("=== Raw lesson extractor test ===\n")
    lesson = extract_lessons(
        "Go to wikipedia.org, search for AI, extract first paragraph.",
        raw_steps,
        "Artificial intelligence (AI) is the capability of..."
    )
    print(lesson)

    print("\n=== Verifying failed selectors are removed ===")
    assert "mw-parser-output > p" not in lesson, "Failed selector still in lesson!"
    assert "wait_for_element"     not in lesson, "wait_for_element still in lesson!"
    assert "input[name=search]"   in lesson,     "Working selector missing!"
    assert "extract_text"         in lesson,     "Working extract_text missing!"
    print("PASS ✓ — failed selectors removed, working selectors kept\n")