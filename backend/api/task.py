"""
任务状态查询 API
GET /api/v1/tasks/{task_id}        → 查询任务状态
GET /api/v1/tasks/{task_id}/result → 查询任务结果
GET /api/v1/tasks/mq/stats      → 队列统计
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.mq.redis_client import redis_client

router = APIRouter(prefix="/api/v1/tasks", tags=["Tasks"])


@router.get("/{task_id}")
async def get_task_status(task_id: str):
    """查询任务状态"""
    status = await redis_client.get_task_status(task_id)
    if status is None:
        raise HTTPException(status_code=404, detail="任务不存在或已过期")
    return {"task_id": task_id, "status": status}


@router.get("/{task_id}/result")
async def get_task_result(task_id: str):
    """查询任务结果"""
    result = await redis_client.get_task_result(task_id)
    if result is None:
        status = await redis_client.get_task_status(task_id)
        if status in ("pending", "running"):
            return {"task_id": task_id, "status": status, "result": None}
        raise HTTPException(status_code=404, detail="任务结果不存在或已过期")
    return {"task_id": task_id, "status": result.get("status"), "result": result}


@router.get("/mq/stats")
async def get_queue_stats():
    """队列统计信息"""
    length = await redis_client.queue_length()
    return {
        "queue_length": length,
        "redis_alive": await redis_client.ping(),
    }

