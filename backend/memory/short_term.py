"""
短期记忆：由 LangGraph Checkpointer 自动管理
此文件提供辅助函数，方便操作 messages
"""
from __future__ import annotations

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langgraph.checkpoint.sqlite import SqliteSaver

from backend.config import settings


def get_checkpointer(db_path: str | None = None) -> SqliteSaver:
    """获取 SQLite Checkpointer 实例"""
    path = db_path or settings.db.sqlite_path
    return SqliteSaver.from_conn_string(path)


def build_thread_config(session_id: str) -> dict:
    """构建 LangGraph 运行配置（用于多轮对话隔离）"""
    return {
        "configurable": {
            "thread_id": session_id,
        },
    }


async def get_session_messages(
        graph,
        session_id: str,
) -> list[BaseMessage]:
    """从 Checkpointer 中读取指定会话的消息历史"""
    config = build_thread_config(session_id)
    state = await graph.get_state(config)
    if state is None:
        return []
    return list(state.values.get("messages", []))


def format_messages_for_display(
        messages: list[BaseMessage],
) -> list[dict]:
    """将消息列表转为前端友好的格式"""
    result = []
    for msg in messages:
        role = {
            HumanMessage: "user",
            AIMessage: "assistant",
        }.get(type(msg), "system")
        result.append({
            "role": role,
            "content": msg.content if isinstance(msg.content, str) else str(msg.content),
        })
    return result


