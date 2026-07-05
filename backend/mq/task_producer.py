"""
任务生产者
提供便捷函数，供 API / 节点将任务推入队列
"""
from __future__ import annotations

import logging

from backend.mq.redis_client import redis_client
from backend.mq.tasks import TaskType

logger = logging.getLogger(__name__)


class TaskProducer:

    async def submit_memory_extract(
        self,
        user_id: str,
        session_id: str,
        user_input: str,
        assistant_response: str,
    ) -> str:
        return await redis_client.enqueue_task({
            "type": TaskType.MEMORY_EXTRACT.value,
            "user_id": user_id,
            "session_id": session_id,
            "user_input": user_input,
            "assistant_response": assistant_response,
        })

    async def submit_document_ingest(
        self,
        file_path: str,
        file_type: str,
        doc_id: str,
        filename: str,
        collection: str,
    ) -> str:
        return await redis_client.enqueue_task({
            "type": TaskType.DOCUMENT_INGEST.value,
            "file_path": file_path,
            "file_type": file_type,
            "doc_id": doc_id,
            "filename": filename,
            "collection": collection,
        })

    async def submit_batch_summarize(self, session_id: str) -> str:
        return await redis_client.enqueue_task({
            "type": TaskType.BATCH_SUMMARIZE.value,
            "session_id": session_id,
        })


task_producer = TaskProducer()
