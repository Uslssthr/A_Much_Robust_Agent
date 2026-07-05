"""
轻量级链路追踪
为每个 Agent 请求生成 trace，记录各节点耗时
（简化版，如需完整可观测性可接入 OpenTelemetry / LangSmith）
"""
from __future__ import annotations

import logging
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class Span:
    name: str
    start: float
    end: float | None = None
    metadata: dict = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        if self.end is None:
            return 0.0
        return (self.end - self.start) * 1000.0


@dataclass
class Trace:
    trace_id: str
    session_id: str
    spans: list[Span] = field(default_factory=list)
    start: float = field(default_factory=time.time)

    @contextmanager
    def span(self, name: str, **metadata):
        """记录一个节点的执行 span"""
        s = Span(name=name, start=time.time(), metadata=metadata)
        try:
            yield s
        finally:
            s.end = time.time()
            self.spans.append(s)
            logger.debug(f"[Trace:{self.trace_id[:8]}] {name}: {s.duration_ms:.2f}ms")

    def summary(self) -> dict:
        total_ms = (time.time() - self.start) * 1000.0
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "total_ms": round(total_ms, 2),
            "spans": [
                {
                    "name": s.name,
                    "duration_ms": round(s.duration_ms, 2),
                    "metadata": s.metadata,
                }
                for s in self.spans
            ]
        }


def new_trace(session_id: str) -> Trace:
    return Trace(trace_id=str(uuid.uuid4()), session_id=session_id)
