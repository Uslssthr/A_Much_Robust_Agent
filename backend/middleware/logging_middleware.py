import logging
import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger(__name__)

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """记录每个请求的方法、路径、耗时、状态码"""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = str(uuid.uuid4())[:8]
        start = time.time()

        # 注入 request_id（方便日志追踪）
        request.state.request_id = request_id

        logger.info(
            f"[{request_id}] -> {request.method} {request.url.path}"
            f"client={request.client.host if request.client.host else 'unknown'}"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"[{request_id}] ✗ 未捕获异常: {e}", exc_info=True)
            raise

        elapsed = (time.time() - start) * 1000
        logger.info(
            f"[{request_id}] <- {response.status_code}"
            f"{elapsed:.1f}ms {request.url.path}"
        )

        response.headers["X-Request-ID"] = request_id
        return response
