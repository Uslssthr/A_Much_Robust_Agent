from __future__ import annotations
from enum import Enum
from typing import TypedDict, Any, Optional, Annotated, Sequence

from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class RouteType(str, Enum):
    """路由类型枚举"""
    DIRECT = "direct"       # 直接回答（无需工具/RAG）
    RAG = "rag"             # 需要知识库检索
    REACT = "react"         # 需要 ReAct 工具推理
    HYBRID = "hybrid"       # RAG + 工具混合


class SafetyLevel(str, Enum):
    """安全风险级别"""
    NONE    = "none"
    LOW     = "low"
    MEDIUM  = "medium"
    HIGH    = "high"


class RetrievedDoc(TypedDict):
    """检索到的文档片段"""
    content:    str
    source:     str
    score:      float
    chunk_id:   str
    metadata:   dict[str, Any]


class ToolCall(TypedDict):
    """工具调用记录"""
    tool_name:  str
    tool_input: dict[str, Any]
    tool_output: Optional[str]
    error:      Optional[str]
    elapsed_ms: Optional[float]
    timestamp:  str


class AgentState(TypedDict):
    """
    LangGraph 全局状态
    所有节点共享此状态，通过读取/写入状态来通信
    """

    # 基础信息
    session_id:     str             # 会话唯一 ID
    user_id:        str             # 用户 ID
    user_input:     str             # 当前用户输入（原始）

    # 消息历史（LangGraph自动append，不可直接覆盖）
    # add_messages reducer: 自动追加，支持去重更新
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # 安全检查
    safety_passed:      bool
    safety_reason:      Optional[str]
    safety_level:       SafetyLevel

    # 上下文工程
    context_token_count:    int
    context_overflow:       bool            # 是否触发了压缩
    context_summary:        Optional[str]   # 历史摘要（压缩后）
    compression_strategy:    Optional[str]   # 使用的压缩策略名

    # 路由
    route:  RouteType

    # RAG检索
    collection:         str                     # 知识库集合名称
    retrieved_docs:     list[RetrievedDoc]      # 检索到的文档列表
    rag_context:        Optional[str]           # 格式化后的 RAG 上下文文本

    # ReAct推理
    current_thought:    Optional[str]           # 当前思考过程
    needs_tool:         bool                    # 是否需要调用工具
    pending_tool:       Optional[dict]          # 待执行的工具（名称+参数）
    iteration_count:    int                     # 当前迭代次数

    # 工具调用
    tool_calls_history: list[ToolCall]           # 本轮对话工具调用历史

    # 记忆系统
    long_term_memory:  Optional[str]            # 从长期记忆中召回的内容

    # 最终输出
    final_answer:      Optional[str]            # 最终回答
    error:             Optional[str]            # 错误信息


def create_initial_state(
        user_input: str,
        session_id: str,
        user_id:    str = "anonymous",
        collection: str = "default",
) -> AgentState:
    """构造初始状态，所有字段给予安全默认值"""
    return AgentState(
        session_id=session_id,
        user_id=user_id,
        user_input=user_input,
        messages=[],
        collection=collection,
        safety_passed=False,
        safety_reason=None,
        safety_level=SafetyLevel.NONE,
        context_token_count=0,
        context_overflow=False,
        context_summary=None,
        compression_strategy=None,
        route=RouteType.DIRECT,
        retrieved_docs=[],
        rag_context=None,
        current_thought=None,
        needs_tool=False,
        pending_tool=None,
        iteration_count=0,
        tool_calls_history=[],
        long_term_memory=None,
        final_answer=None,
        error=None,
    )
