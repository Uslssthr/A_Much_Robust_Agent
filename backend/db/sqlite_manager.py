"""
SQLite 统一管理
- 会话表
- 消息表
- 知识库文档元数据表
- 长期记忆表
所有建表语句集中在这里，启动时一次性初始化
"""
from __future__ import annotations

import logging
import os.path
import sqlite3
from contextlib import contextmanager

from backend.config import settings

logger = logging.getLogger(__name__)

# DDL：所有建表语句
INIT_SQL = """
-- 会话表
CREATE TABLE IF NOT EXISTS sessions (
    session_id    TEXT PRIMARY KEY,
    user_id       TEXT NOT NULL DEFAULT 'anonymous',
    title         TEXT,
    summary       TEXT,
    route_stats   TEXT DEFAULT '{}',      -- JSON: {react:N, rag:N, direct:N}
    message_count INTEGER NOT NULL DEFAULT 0,
    token_total   INTEGER NOT NULL DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 消息快照表（LangGraph checkpointer 之外的业务备份）
CREATE TABLE IF NOT EXISTS messages (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL REFERENCES sessions(session_id),
    role          TEXT NOT NULL CHECK(role IN ('user','assistant','tool','system')),
    content       TEXT NOT NULL,
    tool_name     TEXT,
    tool_input    TEXT,                   -- JSON
    token_count   INTEGER DEFAULT 0,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 知识库文档元数据表
CREATE TABLE IF NOT EXISTS kb_documents (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id        TEXT NOT NULL UNIQUE,   -- uuid
    filename      TEXT NOT NULL,
    file_type     TEXT NOT NULL,          -- pdf/md/txt/docx
    file_size     INTEGER DEFAULT 0,      -- bytes
    chunk_count   INTEGER DEFAULT 0,
    status        TEXT NOT NULL DEFAULT 'processing'
                      CHECK(status IN ('processing','active','failed','deleted')),
    error_msg     TEXT,
    collection    TEXT NOT NULL DEFAULT 'default',
    uploaded_by   TEXT DEFAULT 'anonymous',
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 长期记忆表
CREATE TABLE IF NOT EXISTS long_term_memory (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id       TEXT NOT NULL,
    session_id    TEXT NOT NULL,
    memory_type   TEXT NOT NULL
                      CHECK(memory_type IN ('preference','fact','decision','summary')),
    content       TEXT NOT NULL,
    importance    REAL NOT NULL DEFAULT 0.8
                      CHECK(importance BETWEEN 0.0 AND 1.0),
    access_count  INTEGER NOT NULL DEFAULT 0,
    last_access   DATETIME,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- 工具调用日志表
CREATE TABLE IF NOT EXISTS tool_call_logs (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    tool_name     TEXT NOT NULL,
    tool_input    TEXT,                   -- JSON
    tool_output   TEXT,
    error         TEXT,
    elapsed_ms    REAL,
    created_at    DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- ── 索引 ────────────────────────────────────────────────────────────────────
CREATE INDEX IF NOT EXISTS idx_messages_session
    ON messages(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_kb_docs_status
    ON kb_documents(status, collection);

CREATE INDEX IF NOT EXISTS idx_ltm_user
    ON long_term_memory(user_id, importance DESC, last_access DESC);

CREATE INDEX IF NOT EXISTS idx_tool_logs_session
    ON tool_call_logs(session_id, created_at);

CREATE INDEX IF NOT EXISTS idx_sessions_user
    ON sessions(user_id, updated_at DESC);
"""


class SQLiteManager:
    """线程安全的 SQLite 管理器（同步版，供非异步代码使用）"""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db.sqlite_path
        self._ensure_dir()

    def _ensure_dir(self):
        dir_path = os.path.dirname(self.db_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)

    def get_conn(self) -> sqlite3.Connection:
        """获取新连接（每次调用返回新连接，避免多线程共享）"""
        conn = sqlite3.connect(
            self.db_path,
            check_same_thread=False,
            isolation_level=None,       # autocommit
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")     # 提升并发读写性能
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    @contextmanager
    def transaction(self):
        """手动事务上下文（需要原子操作时使用）"""
        conn = self.get_conn()
        conn.execute("BEGIN")
        try:
            yield conn
            conn.execute("COMMIT")
        except Exception:
            conn.execute("ROLLBACK")
            raise
        finally:
            conn.close()

    def initialize(self):
        """建表 & 索引（应用启动时调用一次）"""
        conn = self.get_conn()
        try:
            conn.executescript(INIT_SQL)
            logger.info(f"[DB] 数据库初始化完成：{self.db_path}")
        finally:
            conn.close()

    # sessions 相关操作

    def upsert_session(
        self,
        session_id: str,
        user_id: str = "anonymous",
        title: str | None = None,
        message_count: int = 0,
    ):
        conn = self.get_conn()
        try:
            conn.execute("""
                INSERT INTO sessions (session_id, user_id, title, message_count)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    message_count = message_count + excluded.message_count,
                    updated_at = CURRENT_TIMESTAMP
            """, (
                session_id, user_id, title, message_count
            ))
        finally:
            conn.close()

    def get_sessions(self, user_id: str, limit: int = 20) -> list[dict]:
        conn = self.get_conn()
        try:
            rows = conn.execute("""
                SELECT session_id, title, summary, message_count,
                        token_total, created_at, updated_at
                FROM sessions
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
            """, (user_id, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def update_session_summary(self, session_id: str, summary: str, title: str = ""):
        conn = self.get_conn()
        try:
            conn.execute("""
                UPDATE sessions SET summary=?, title=COALESCE(NULLIF(?,''),(
                    SELECT title FROM sessions WHERE session_id=?
                )), updated_at=CURRENT_TIMESTAMP
                WHERE session_id=?
            """, (summary, title, session_id, session_id))
        finally:
            conn.close()

    def delete_session(self, session_id: str):
        conn = self.get_conn()
        try:
            conn.execute("DELETE FROM messages WHERE session_id=?", (session_id,))
            conn.execute("DELETE FROM sessions WHERE session_id=?", (session_id,))
        finally:
            conn.close()


    # messages 相关操作

    def save_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tool_name: str | None = None,
        tool_input: str | None = None,
        token_count: int = 0,
    ):
        conn = self.get_conn()
        try:
            conn.execute("""
                INSERT INTO messages
                    (session_id, role, content, tool_name, tool_input, token_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, role, content, tool_name, tool_input, token_count))
        finally:
            conn.close()

    def get_messages(self, session_id: str, limit: int = 100) -> list[dict]:
        conn = self.get_conn()
        try:
            rows = conn.execute("""
                SELECT role, content, tool_name, created_at
                FROM messages
                WHERE session_id = ? 
                ORDER BY created_at ASC
                LIMIT ?
            """, (session_id, limit)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()


    # 知识库相关

    def create_kb_doc(
        self,
        doc_id: str,
        filename: str,
        file_type: str,
        file_size: int,
        collection: str = "default",
        uploaded_by: str = "anonymous",
    ):
        conn = self.get_conn()
        try:
            conn.execute("""
                INSERT INTO kb_documents
                    (doc_id, filename, file_type, file_size, collection, uploaded_by)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (doc_id, filename, file_type, file_size, collection, uploaded_by))
        finally:
            conn.close()

    def update_kb_doc_status(
        self,
        doc_id: str,
        status: str,
        chunk_count: int = 0,
        error_msg: str | None = None,
    ):
        conn = self.get_conn()
        try:
            conn.execute("""
                UPDATE kb_documents
                SET status=?, chunk_count=?, error_msg=?, updated_at=CURRENT_TIMESTAMP
                WHERE doc_id=?
            """, (status, chunk_count, error_msg, doc_id))
        finally:
            conn.close()

    def list_kb_docs(
        self,
        collection: str = "default",
        status: str = "active",
    ) -> list[dict]:
        conn = self.get_conn()
        try:
            rows = conn.execute("""
                SELECT doc_id, filename, file_type, file_size,
                        chunk_count, status, created_at
                FROM kb_documents
                WHERE collection=? AND status=?
                ORDER BY created_at DESC
            """, (collection, status)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def soft_delete_kb_doc(self, doc_id: str) -> bool:
        conn = self.get_conn()
        try:
            cur = conn.execute("""
                UPDATE kb_documents SET status='deleted', updated_at=CURRENT_TIMESTAMP
                WHERE doc_id=? AND status='active'
            """, (doc_id,))
            return cur.rowcount > 0
        finally:
            conn.close()

    # Tool Call 日志相关

    def save_tool_log(
        self,
        session_id: str,
        tool_name: str,
        tool_input: str | None = None,
        tool_output: str | None = None,
        error: str | None = None,
        elapsed_ms: float = 0.0,
    ):
        conn = self.get_conn()
        try:
            conn.execute("""
                INSERT INTO tool_call_logs
                    (session_id, tool_name, tool_input, tool_output, error, elapsed_ms)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (session_id, tool_name, tool_input, tool_output, error, elapsed_ms))
        finally:
            conn.close()

# 数据库实例
db = SQLiteManager()
