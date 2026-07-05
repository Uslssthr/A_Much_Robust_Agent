import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from backend.monitoring.metrics import http_requests_total, http_requests_duration

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径、耗时、状态码，并上报 Prometheus 指标"""

    # 不统计指标的路径（避免 /metrics 自身污染数据）
    SKIP_PATHS = {"/metrics", "/api/v1/ping", "/favicon.ico"}

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # 注入 request_id（方便日志追踪）
        request.state.request_id = request_id

        path = self._normalize_path(request.url.path)

        logger.info(
            f"[{request_id}] -> {request.method} {request.url.path}"
            f"client={request.client.host if request.client.host else 'unknown'}"
        )

        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        except Exception as e:
            logger.error(f"[{request_id}] ✗ 未捕获异常: {e}", exc_info=True)
            raise
        finally:
            elapsed = (time.time() - start) * 1000
            logger.info(
                f"[{request_id}] <- {status_code}"
                f"{elapsed:.2f}ms {request.url.path}"
            )

            # 上报指标（跳过无意义路径）
            if request.url.path not in self.SKIP_PATHS:
                http_requests_total.labels(
                    method=request.method,
                    endpoint=path,
                    status=str(status_code),
                ).inc()
                http_requests_duration.labels(
                    method=request.method,
                    endpoint=path,
                ).observe(elapsed)

    @staticmethod
    def _normalize_path(path: str) -> str:
        """
        将路径中的动态参数替换为占位符
        避免 Prometheus 标签基数爆炸
        /api/v1/chat/history/abc-123  →  /api/v1/chat/history/{id}
        /api/v1/knowledge/docs/xyz    →  /api/v1/knowledge/docs/{id}
        """
        import re
        # UUID格式
        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}", path
        )
        # 纯数字ID
        path = re.sub(r"\d+", "{id}", path)
        return path


