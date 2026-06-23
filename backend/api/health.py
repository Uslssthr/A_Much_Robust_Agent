import logging
import sqlite3
import time

from fastapi import APIRouter, Request

from backend.config import settings

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1", tags=["Health"])

START_TIME = time.time()


@router.get("/health")
async def health_check(request: Request):
    """
    健康检查接口
    检查：SQLite / Redis / ChromaDB / Agent Graph
    """
    checks = {}

    # 1. SQLite 连接
    try:
        conn = sqlite3.connect(settings.db.sqlite_path)
        conn.execute("SELECT 1").fetchone()
        conn.close()
        checks["sqlite"] = {"status": "ok"}

    except Exception as e:
        checks["sqlite"] = {"status": "error", "error": str(e)}


    # 2. Redis 检查
    try:
        import redis
        r = redis.from_url(settings.db.redis_url, socket_connect_timeout=5)
        r.ping()
        checks["redis"] = {"status": "ok"}
    except Exception as e:
        checks["redis"] = {"status": "error", "error": str(e)}


    # 3. Agent Graph 检查
    try:
        graph = request.app.state.agent_graph
        checks["agent_graph"] = {
            "status": "ok" if graph else "not_initialized",
        }
    except Exception as e:
        checks["agent_graph"] = {"status": "error", "error": str(e)}


    # 4. ChromaDB 检查
    try:
        import chromadb
        client = chromadb.PersistentClient(path=settings.rag.chroma_dir)
        client.heartbeat()
        checks["chromadb"] = {"status": "ok"}
    except Exception as e:
        checks["chromadb"] = {"status": "error", "error": str(e)}

    overall = "healthy" if all(
        v.get("status") == "ok" for v in checks.values()
    ) else "degraded"

    return {
        "status": overall,
        "uptime_s": round(time.time() - START_TIME, 2),
        "checks": checks,
        "version": "1.0.0",
    }


@router.get("/ping")
async def ping():
    """最简心跳"""
    return {"pong": True, "ts": time.time()}

