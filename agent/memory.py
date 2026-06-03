import os
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions
from langchain_classic.memory import ConversationBufferMemory

session_memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
)


def reset_session():
    """Call this at the start of each new task."""
    session_memory.clear()


CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./memory_store")

_chroma_client: Optional[chromadb.PersistentClient] = None
_collection = None


def _get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=CHROMA_DIR)
        ef = embedding_functions.DefaultEmbeddingFunction()
        _collection = _chroma_client.get_or_create_collection(
            name="task_memory",
            embedding_function=ef,
        )
    return _collection


def save_task_to_memory(task: str, steps: list[str], result: str):
    """Persist a completed task so the agent can recall it later."""
    col = _get_collection()
    doc = f"TASK: {task}\nSTEPS:\n" + "\n".join(steps) + f"\nRESULT: {result}"
    col.add(
        documents=[doc],
        ids=[f"task_{datetime.now().isoformat()}"],
        metadatas=[{"task": task, "timestamp": datetime.now().isoformat()}],
    )


def recall_similar_tasks(task: str, n: int = 2) -> str:
    """Retrieve the n most similar past tasks to inject into the system prompt."""
    col = _get_collection()
    if col.count() == 0:
        return ""
    results = col.query(query_texts=[task], n_results=min(n, col.count()))
    if not results["documents"] or not results["documents"][0]:
        return ""
    docs = results["documents"][0]
    header = "## Relevant past tasks you have completed:\n"
    return header + "\n\n---\n".join(docs)
