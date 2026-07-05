"""
监控装饰器：一行代码为函数添加指标追踪
"""
from __future__ import annotations

import functools
import logging
import time
from typing import Callable

from .metrics import (
    tool_calls_total, tool_duration,
    llm_call_duration, llm_tokens_total,
    rag_retrieval_duration,
)

logger = logging.getLogger(__name__)


def track_tool(tool_name: str):
    """追踪工具调用指标"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.time()
            status = "success"
            try:
                result = await fn(*args, **kwargs)
                return result
            except TimeoutError:
                status = "timeout"
                raise
            except Exception:
                status = "error"
                raise
            finally:
                elapsed = time.time() - start
                tool_calls_total.labels(tool=tool_name, status=status).inc()
                tool_duration.labels(tool=tool_name, status=status).observe(elapsed)
        return wrapper
    return decorator


def track_llm(model: str):
    """追踪 LLM 调用延迟"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = await fn(*args, **kwargs)
                # 尝试提取token使用量
                if hasattr(result, "usage_metadata"):
                    usage = result.usage_metadata or {}
                    llm_tokens_total.labels(model=model, type="input").inc(
                        usage.get("input_tokens", 0)
                    )
                    llm_tokens_total.labels(model=model, type="output").inc(
                        usage.get("output_tokens", 0)
                    )
                return result
            finally:
                llm_call_duration.labels(model=model).observe(time.time() - start)
        return wrapper
    return decorator


def track_rag():
    """追踪 RAG 检索延迟"""
    def decorator(fn: Callable):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            start = time.time()
            try:
                return await fn(*args, **kwargs)
            finally:
                rag_retrieval_duration.observe(time.time() - start)
        return wrapper
    return decorator
