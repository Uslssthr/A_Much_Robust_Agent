"""
任务类型定义 & 处理器注册表
支持的异步任务类型：
  - document_ingest   : 文档向量化（长耗时）
  - batch_summarize   : 批量摘要
  - memory_extract    : LLM 记忆提取
  - long_tool_call    : 长耗时工具调用
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)


class TaskType(str, Enum):
    DOCUMENT_INGEST = "document_ingest"
    BATCH_SUMMARIZE = "batch_summarize"
    MEMORY_EXTRACT  = "memory_extract"
    LONG_TOOL_CALL  = "long_tool_call"


# 处理器注册表: task_type -> async_handler
_TASK_HANDLERS: dict[str, Callable[[dict], Awaitable[dict]]] = {}


def task_handler(task_type: TaskType):
    """装饰器：注册任务处理器"""
    def decorator(fn: Callable[[dict], Awaitable[dict]]):
        _TASK_HANDLERS[task_type.value] = fn
        logger.info(f"[Tasks] 注册任务处理器：{task_type.value}")
        return fn
    return decorator


def get_handler(task_type: str) -> Callable[[dict], Awaitable[dict]] | None:
    return _TASK_HANDLERS.get(task_type)

# ————————————
# 具体任务处理器
# ————————————

@task_handler(TaskType.MEMORY_EXTRACT)
async def handle_memory_extract(payload: dict) -> dict:
    """
    LLM 记忆提取任务
    从对话中提取值得长期记忆的信息
    """
    from langchain_openai import ChatOpenAI
    from backend.agent.prompts import MEMORY_EXTRACTION_PROMPT
    from backend.memory.long_term import long_term_memory
    from backend.config import settings
    import json, re

    user_id    = payload["user_id"]
    session_id = payload["session_id"]
    user_input = payload["user_input"]
    assistant  = payload["assistant_response"]

    llm = ChatOpenAI(
        model=settings.llm.model,
        temperature=0,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )

    chain = MEMORY_EXTRACTION_PROMPT | llm
    response = await chain.ainvoke({
        "user_input": user_input,
        "assistant_response": assistant,
    })

    # 解析JSON
    saved = 0
    try:
        match = re.search(r"\{.*\}", response.content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            for mem in data.get("memories", []):
                long_term_memory.save(
                    user_id=user_id,
                    session_id=session_id,
                    content=mem["content"],
                    memory_type=mem.get("type", "fact"),
                    importance=float(mem.get("importance", 0.7))
                )
                saved += 1
    except Exception as e:
        logger.warning(f"[Task:memory_extract] 解析失败：{e}")

    return {"extracted": saved}


@task_handler(TaskType.DOCUMENT_INGEST)
async def handle_document_ingest(payload: dict) -> dict:
    """
    文档向量化任务（从队列消费，替代之前的 asyncio.create_task）
    """
    from backend.rag.ingestion import ingestion_pipline

    file_path  = payload["file_path"]
    file_type  = payload["file_type"]
    doc_id     = payload["doc_id"]
    filename   = payload["filename"]
    collection = payload["collection"]

    await ingestion_pipline._process_file(
        file_path=file_path,
        file_type=file_type,
        doc_id=doc_id,
        filename=filename,
        collection=collection,
    )
    return {"doc_id": doc_id, "status": "processed"}


@task_handler(TaskType.BATCH_SUMMARIZE)
async def handle_batch_summarize(payload: dict) -> dict:
    """批量会话摘要任务"""
    from langchain_openai import ChatOpenAI
    from backend.agent.prompts import SUMMARIZATION_PROMPT
    from backend.db.sqlite_manager import db
    from backend.config import settings

    session_id = payload["session_id"]
    messages = db.get_messages(session_id, limit=100)

    history_text = "\n".join(
        f"[{m['role']}]: {m['content']}" for m in messages
    )

    llm = ChatOpenAI(
        model=settings.llm.model,
        temperature=0,
        api_key=settings.llm.api_key,
        base_url=settings.llm.base_url,
    )
    chain = SUMMARIZATION_PROMPT | llm
    response = await chain.ainvoke({"history_text": history_text})

    db.update_session_summary(session_id, response.content)
    return {"session_id": session_id, "summary_len": len(response.content)}


