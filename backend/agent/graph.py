"""
LangGraph 图定义 —— 完整 Agent 流程编排
"""
from __future__ import annotations
import logging

from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph import StateGraph, END

from backend.agent.state import AgentState, RouteType, SafetyLevel
from backend.agent.nodes import (
    safety_checker,
    context_manager,
    router,
    rag_retriever,
    react_loop,
    tool_executor,
)
from backend.config import settings

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════
# 条件边函数（每个函数决定下一跳到哪个节点）
# ══════════════════════════════════════════════════════════════════

def after_safety_check(state: AgentState) -> str:
    """
    安全检查后的路由：
    通过 → 进入上下文管理
    未通过 → 直接结束（FastAPI层会读取 safety_reason 返回给用户）
    """
    if state["safety_passed"]:
        return "context_manage"
    else:
        level = state.get("safety_level", SafetyLevel.NONE)
        logger.warning(
            f"[Graph] 安全检查未通过: "
            f"level={level} reason={state.get('safety_reason')}"
        )
        return END

def after_router(state: AgentState) -> str:
    """
    路由节点后的分发：
    direct  → 直接进入 react_think（会走 _direct_answer 分支）
    rag     → 先检索知识库
    react   → 直接进入 react_think
    hybrid  → 先检索知识库，再 react
    """
    route = state.get("route", RouteType.REACT)
    logger.info(f"[Graph] after_router -> {route}")

    if route == RouteType.DIRECT:
        return "react_think"
    elif route == RouteType.RAG:
        return "rag_retrieve"
    elif route == RouteType.REACT:
        return "react_think"
    elif route == RouteType.HYBRID:
        return "rag_retrieve"
    else:
        return "react_think"


def after_react_think(state: AgentState) -> str:
    """
    ReAct 推理后的路由：
    需要工具 → 执行工具
    不需要工具 → 结束（final_answer 已设置）
    超过迭代次数 → 强制结束
    """
    needs_tool = state.get("needs_tool", False)
    iteration = state.get("iteration_count", 0)
    has_answer = bool(state.get("final_answer"))

    if has_answer:
        logger.info(f"[Graph] react_think -> END (has_answer)")
        return END

    if needs_tool and iteration < settings.agent.max_iterations:
        logger.info(f"[Graph] react_think -> tool_execute (iter={iteration})")
        return "tool_execute"

    if iteration >= settings.agent.max_iterations:
        logger.warning(f"[Graph] 超过最大迭代次数 {settings.agent.max_iterations}")
        logger.info(f"[Graph] react_think -> END (iter={iteration})")

    return END


def after_tool_execute(state: AgentState) -> str:
    """
    工具执行后的路由：
    始终回到 react_think 进行下一轮推理
    （迭代次数限制在 after_react_think 中控制）
    """
    iteration = state.get("iteration_count", 0)
    logger.info(f"[Graph] tool_execute → react_think (iter={iteration})")
    return "react_think"


# ══════════════════════════════════════════════════════════════
# 图结构构建
# ══════════════════════════════════════════════════════════════

def _build_graph_structure() -> StateGraph:
    """只构建节点和边，不 compile，方便复用"""
    builder = StateGraph(AgentState)

    # 注册节点
    builder.add_node("safety_check",   safety_checker.run)
    builder.add_node("context_manage", context_manager.run)
    builder.add_node("router",         router.run)
    builder.add_node("rag_retrieve",   rag_retriever.run)
    builder.add_node("react_think",    react_loop.run)
    builder.add_node("tool_execute",   tool_executor.run)

    # 入口
    builder.set_entry_point("safety_check")

    # 边
    builder.add_conditional_edges(
        "safety_check", after_safety_check,
        {"context_manage": "context_manage", END: END},
    )
    builder.add_edge("context_manage", "router")
    builder.add_conditional_edges(
        "router", after_router,
        {"react_think": "react_think", "rag_retrieve": "rag_retrieve"},
    )
    builder.add_edge("rag_retrieve", "react_think")
    builder.add_conditional_edges(
        "react_think", after_react_think,
        {"tool_execute": "tool_execute", END: END},
    )
    builder.add_conditional_edges(
        "tool_execute", after_tool_execute,
        {"react_think": "react_think"},
    )

    return builder

# ══════════════════════════════════════════════════════════════
# 对外暴露的两种构建方式
# ══════════════════════════════════════════════════════════════

def build_agent_graph_with_checkpointer(checkpointer: AsyncSqliteSaver):
    """
    由外部（FastAPI lifespan / 测试）传入已初始化的 checkpointer
    推荐生产环境使用
    """
    builder = _build_graph_structure()
    graph   = builder.compile(checkpointer=checkpointer)
    logger.info("[Graph] 构建完成（外部 checkpointer）")
    return graph


def build_agent_graph_no_persist():
    """
    不带持久化（InMemorySaver）
    适合：单元测试、CI、快速调试
    """
    from langgraph.checkpoint.memory import InMemorySaver
    builder = _build_graph_structure()
    graph   = builder.compile(checkpointer=InMemorySaver())
    logger.info("[Graph] 构建完成（InMemorySaver，无持久化）")
    return graph

