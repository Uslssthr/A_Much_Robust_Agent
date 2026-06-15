"""
长期记忆：跨会话存储在 SQLite
存储：用户偏好、重要事实、关键决策
"""
from __future__ import annotations

import logging
import sqlite3
from typing import Optional

from backend.config import settings

logger = logging.getLogger(__name__)


class LongTermMemory:

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or settings.db.sqlite_path
        self._conn = None
        self._ensure_tables()

    def _get_conn(self) -> sqlite3.Connection:
        """获取（或重建）数据库连接"""
        if self._conn is None:
            self._conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None,       # autocommit
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _ensure_tables(self):
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS long_term_memory (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id         TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                memory_type     TEXT NOT NULL CHECK(memory_type IN ('preference', 'fact', 'decision', 'summary')),
                content         TEXT NOT NULL,
                importance      REAL NOT NULL DEFAULT 1.0 CHECK(importance BETWEEN 0 AND 1),
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_access     DATETIME,
                access_count    INTEGER DEFAULT 0
            );
            
            CREATE TABLE IF NOT EXISTS conversation_sessions (
                session_id      TEXT PRIMARY KEY,
                user_id         TEXT NOT NULL,
                title           TEXT,
                summary         TEXT,
                message_count   INTEGER DEFAULT 0,
                created_at      DATETIME DEFAULT CURRENT_TIMESTAMP,
                updated_at      DATETIME DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS idx_ltm_user_type
                ON long_term_memory (user_id, memory_type, importance DESC);
                
            CREATE INDEX IF NOT EXISTS idx_sessions_user
                ON conversation_sessions (user_id, updated_at DESC);
        """)
        logger.info("[LongTermMemory] 数据库表初始化完成")

    def save(
        self,
        user_id : str,
        session_id: str,
        content: str,
        memory_type: str = "fact",
        importance: float = 0.8,
    ) -> int:
        """保存一条记忆，返回记录 ID"""
        conn = self._get_conn()
        cursor = conn.execute("""
            INSERT INTO long_term_memory
                (user_id, session_id, memory_type, content, importance)
            VALUES (?, ?, ?, ?, ?)
        """, (user_id, session_id, memory_type, content, importance))
        logger.debug(
            f"[LongTermMemory] 保存记忆: user={user_id} "
            f"type={memory_type} importance={importance}"
        )
        return cursor.lastrowid

    def recall(
        self,
        user_id: str,
        top_k: int = 5,
        min_importance: float = 0.5,
    ) -> list[str]:
        """
        召回最重要的记忆
        同时更新访问时间和访问次数
        """
        conn = self._get_conn()

        # 查询
        rows = conn.execute("""
            SELECT id, content, memory_type, importance
            FROM long_term_memory
            WHERE user_id = ?
                AND importance >= ?
            ORDER BY importance DESC, last_access DESC NULLS LAST
            LIMIT ?
        """, (user_id, min_importance, top_k)).fetchall()

        if not rows:
            return []

        # 批量更新访问记录
        ids = [row["id"] for row in rows]
        conn.execute(f"""
            UPDATE long_term_memory
            SET last_access = CURRENT_TIMESTAMP,
                access_count = access_count + 1
            WHERE id IN ({", ".join('?' * len(ids))})
        """, ids)

        return [
            f"{row['memory_type']}: {row['content']}"
            for row in rows
        ]

    def recall_as_text(self, user_id: str, top_k: int = 5) -> Optional[str]:
        """召回记忆并格式化为文本（注入 Prompt）"""
        memories = self.recall(user_id, top_k=top_k)
        if not memories:
            return None
        return "关于该用户的已知信息：\n" + "\n".join(
            f" • {m}" for m in memories
        )

    def save_session_summary(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        title: str = "",
    ):
        """保存会话摘要"""
        conn = self._get_conn()
        conn.execute("""
            INSERT INTO conversation_sessions
                (session_id, user_id, title, summary)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (session_id) DO UPDATE SET
                summary = excluded.summary,
                title = excluded.title,
                updated_at = CURRENT_TIMESTAMP
        """, (session_id, user_id, title, summary))

    def get_recent_sessions(
        self, user_id: str, limit: int = 10,
    ) -> list[dict]:
        """获取用户最近的会话列表"""
        conn = self._get_conn()
        rows = conn.execute("""
            SELECT session_id, title, summary, message_count, updated_at
            FROM conversation_sessions
            WHERE user_id = ?
            ORDER BY updated_at DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        return [dict(row) for row in rows]

    def delete_memory(self, memory_id: int, user_id: str) -> bool:
        """删除指定记忆（需要验证 user_id 防止越权）"""
        conn = self._get_conn()
        cursor = conn.execute("""
            DELETE FROM long_term_memory
            WHERE id = ? AND user_id = ?
        """, (memory_id, user_id))
        return cursor.rowcount > 0


# 全局单例
long_term_memory = LongTermMemory()
