"""
FastAPI 应用入口
"""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from backend.agent.graph import build_agent_graph_with_checkpointer
from backend.config import settings
from backend.db.sqlite_manager import db
from backend.middleware.logging_middleware import RequestLoggingMiddleware

logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./data/agent.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# 生命周期
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动 & 关闭生命周期管理"""

    # 启动
    logger.info("=" * 50)
    logger.info("🚀 Agent 服务启动中...")

    # 1. 创建必要目录
    for d in ["./data", "./data/uploads", "./data/chroma"]:
        os.makedirs(d, exist_ok=True)

    # 2. 初始化SQLite表结构
    db.initialize()
    logger.info("✅ 数据库初始化完成")

    # 3. 初始化AsyncSqliteSaver + 编译Agent Graph
    db_path = settings.db.sqlite_path
    async with AsyncSqliteSaver.from_conn_string(db_path) as checkpointer:
        app.state.agent_graph = build_agent_graph_with_checkpointer(checkpointer)
        logger.info("✅ Agent Graph 编译完成")
        logger.info("🟢 服务就绪，等待请求...")
        logger.info("=" * 50)

        yield  # ← 应用运行期间阻塞在此，等待请求


    # 关闭
    logger.info("🔴 服务关闭，资源已释放")


# 应用实例
app = FastAPI(
    title="通用Agent API",
    description="支持 RAG / 工具调用 / 多轮对话 / 长期记忆的通用Agent",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)


# 中间件
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # 生产环境替换为前端域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["X-Session-ID", "X-Request-ID"],
)
app.add_middleware(RequestLoggingMiddleware)


# 全局异常处理
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"未处理异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "服务器内部错误", "detail": str(exc)},
    )


# 注册路由
from backend.api.chat import router as chat_router
from backend.api.knowledge import router as knowledge_router
from backend.api.health import router as health_router

app.include_router(chat_router)
app.include_router(knowledge_router)
app.include_router(health_router)


# 启动入口
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
        log_level="debug" if settings.debug else "info",
    )
