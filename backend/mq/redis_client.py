"""
Redis 统一客户端
四大用途：
  1. 任务队列（List）        —— 异步任务分发
  2. 结果存储（String + TTL）—— 任务结果缓存
  3. 流式中继（Pub/Sub）     —— Worker → API 的实时进度推送
  4. 会话缓存 & 分布式锁      —— 减少 SQLite 压力 / 防重复执行
"""
from __future__ import annotations

import json
import logging
import uuid
from typing import Optional, Any, AsyncIterator

import redis.asyncio as aioredis

from backend.config import settings

logger = logging.getLogger(__name__)


class RedisClient:
    """异步 Redis 客户端封装（全局单例）"""

    # Key命名空间
    TASK_QUEUE = "agent:task:mq"
    TASK_RESULT = "agent:task:result:"          # + task_id
    TASK_STATUS = "agent:task:status:"          # + task_id
    STREAM_CHANNEL = "agent:stream:"            # + session_id
    SESSION_CACHE = "agent:session:cache:"      # + session_id
    LOCK_PREFIX = "agent:lock:"                 # + key
    RATE_LIMIT = "agent:ratelimit:"             # + user_id
    SEARCH_CACHE = "agent:search:cache:"        # + query_hash

    def __init__(self):
        self._redis: Optional[aioredis.Redis] = None


    # 连接管理

    async def connect(self):
        """建立连接（应用启动时调用）"""
        if self._redis is None:
            self._redis = aioredis.from_url(
                settings.db.redis_url,
                decode_responses=True,
                max_connections=50,
                socket_connect_timeout=10,
                socket_keepalive=True,
                health_check_interval=30,
            )
            # 测试连接
            await self._redis.ping()
            logger.info(f"[Redis] 连接成功：{settings.db.redis_url}")
        return self._redis

    async def disconnect(self):
        """关闭连接（应用退出时调用）"""
        if self._redis:
            await self._redis.aclose()
            self._redis = None
            logger.info("[Redis] 连接已关闭")

    @property
    def redis(self) -> aioredis.Redis:
        if self._redis is None:
            raise RuntimeError("Redis 未连接，请先调用connect()")
        return self._redis

    async def ping(self) -> bool:
        try:
            return await self._redis.ping()
        except Exception:
            return False


    # 1.任务队列

    async def enqueue_task(self, task: dict[str, Any]) -> str:
        """
        推送任务到队列
        返回 task_id
        """
        task_id = task.get("task_id") or str(uuid.uuid4())
        task["task_id"] = task_id

        # 入队 + 初始化状态
        await self.redis.rpush(self.TASK_QUEUE, json.dumps(task, ensure_ascii=False))
        await self.set_task_status(task_id, "pending")

        logger.info(f"[Redis] 任务入队：task_id={task_id} type={task.get('type')}")
        return task_id

    async def dequeue_task(self, timeout: int = 5) -> Optional[dict]:
        """
        阻塞式取出任务（Worker 调用）
        timeout: 阻塞等待秒数，0=永久阻塞
        """
        result = await self.redis.blpop(self.TASK_QUEUE, timeout=timeout)
        if result is None:
            return None
        _, raw = result
        return json.loads(raw)

    async def queue_length(self) -> int:
        """当前队列长度"""
        return await self.redis.llen(self.TASK_QUEUE)


    # 2.任务状态&结果

    async def set_task_status(self, task_id: str, status: str, ttl: int = 3600):
        """
        状态: pending / running / success / failed
        """
        await self.redis.setex(
            f"{self.TASK_STATUS}{task_id}", ttl, status
        )

    async def get_task_status(self, task_id: str) -> Optional[str]:
        return await self.redis.get(f"{self.TASK_STATUS}{task_id}")

    async def set_task_result(
        self, task_id: str, result: dict, ttl: int = 3600
    ):
        await self.redis.setex(
            f"{self.TASK_RESULT}{task_id}",
            ttl,
            json.dumps(result, ensure_ascii=False, default=str),
        )

    async def get_task_result(self, task_id: str) -> Optional[dict]:
        raw = await self.redis.get(f"{self.TASK_RESULT}{task_id}")
        return json.loads(raw) if raw else None


    # 3.流式中继(Pub/Sub)

    async def publish_stream(self, session_id: str, event: dict):
        """Worker 发布进度事件"""
        await self.redis.publish(
            f"{self.STREAM_CHANNEL}{session_id}",
            json.dumps(event, ensure_ascii=False),
        )

    async def subscribe_stream(
        self, session_id: str
    ) -> AsyncIterator[dict]:
        """
        API 订阅进度事件（异步生成器）
        收到 {"type":"done"} 时自动结束
        """
        pubsub = self.redis.pubsub()
        await pubsub.subscribe(f"{self.STREAM_CHANNEL}{session_id}")
        try:
            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                event = json.loads(message["data"])
                yield event
                if event.get("type") == "done":
                    break
        finally:
            await pubsub.unsubscribe(f"{self.STREAM_CHANNEL}{session_id}")
            await pubsub.aclose()


    # 4.会话缓存
    async def cache_session(self, session_id: str, data: dict, ttl: int = 1800):
        await self.redis.setex(
            f"{self.SESSION_CACHE}{session_id}",
            ttl,
            json.dumps(data, ensure_ascii=False, default=str),
        )

    async def get_cached_session(self, session_id:str) -> Optional[dict]:
        raw = await self.redis.get(f"{self.SESSION_CACHE}{session_id}")
        return json.loads(raw) if raw else None

    async def invalidate_session(self, session_id: str):
        await self.redis.delete(f"{self.SESSION_CACHE}{session_id}")


    # 5.分布式锁

    async def acquire_lock(self, key: str, ttl: int = 30) -> bool:
        """SET NX EX 实现分布式锁"""
        return bool(await self.redis.set(
            f"{self.LOCK_PREFIX}{key}","1", nx=True, ex=ttl
        ))

    async def release_lock(self, key: str):
        await self.redis.delete(f"{self.LOCK_PREFIX}{key}")

    # 6.速率限制（滑动窗口）

    async def check_rate_limit(
        self, user_id: str, max_requests: int = 30, window: int = 60
    ) -> tuple[bool, int]:
        """
        滑动窗口限流
        返回: (是否允许, 剩余配额)
        """
        key = f"{self.RATE_LIMIT}{user_id}"
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, window)
        remaining = max(0, max_requests - current)
        allowed = current <= max_requests
        return allowed, remaining


    # 7.搜索结果缓存

    async def cache_research(self, query: str, result: str, ttl: int = 86400):
        import hashlib
        key = hashlib.md5(query.encode()).hexdigest()
        await self.redis.setex(f"{self.SEARCH_CACHE}{key}", ttl, result)

    async def get_cached_search(self, query: str) -> Optional[str]:
        import hashlib
        key = hashlib.md5(query.encode()).hexdigest()
        return await self.redis.get(f"{self.SEARCH_CACHE}{key}")


# 全局单例
redis_client = RedisClient()

