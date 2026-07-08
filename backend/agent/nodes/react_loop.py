"""
ReAct 推理节点
负责：
  1. 综合所有上下文（RAG/记忆/历史）生成推理
  2. 决定是直接回答 or 调用工具
  3. 解析 LLM 的 tool_call 输出
  4. 超限保护（防止无限循环）
"""
from __future__ import annotations

import json
import logging
import time

from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI

from backend.agent.prompts import DIRECT_ANSWER_PROMPT, RAG_SYNTHESIS_PROMPT, REACT_PROMPT
from backend.agent.state import AgentState, ToolCall, RouteType
from backend.config import settings
from backend.monitoring.metrics import llm_call_duration, llm_tokens_total, agent_iterations

from backend.tools.registry import get_all_tools, format_tool_description

logger = logging.getLogger(__name__)

class ReActLoopNode:

    def __init__(self):
        # 绑定工具的 LLM（支持 function calling）
        self.tools: list[BaseTool] = get_all_tools()
        self.llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            streaming=True,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        ).bind_tools(self.tools)

        # 无工具版本（用于直接回答和 RAG 合成）
        self.llm_no_tools = ChatOpenAI(
            model=settings.llm.model,
            temperature=settings.llm.temperature,
            max_tokens=settings.llm.max_tokens,
            streaming=True,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

        self.max_iterations = settings.agent.max_iterations

    def _format_tool_history(self, tool_calls: list[ToolCall]) -> str:
        """将工具调用历史格式化为可读字符串"""
        if not tool_calls:
            return "暂无工具调用记录"
        lines = []
        for tc in tool_calls:
            status = "✅" if not tc.get("error") else "❌"
            lines.append(
                f"{status} {tc['tool_name']}("
                f"{json.dumps(tc['tool_input'], ensure_ascii=False)}) "
                f"→ {tc.get('tool_output', tc.get('error', ''))[:200]}"
            )

        return "\n".join(lines)

    def _build_react_prompt_vars(self, state: AgentState) -> dict:
        """构建 ReAct Prompt 的变量"""
        return {
            "messages": list(state["messages"]),
            "user_input": state["user_input"],
            "tool_descriptions": format_tool_description(self.tools),
            "rag_context": state.get("rag_context", "暂无知识库上下文"),
            "long_term_memory": state.get("long_term_memory", "暂无长期记忆"),
            "context_summary": state.get("context_summary", ""),
            "max_iterations": str(self.max_iterations),
            "tool_calls_history": self._format_tool_history(
                state.get("tool_calls_history", [])
            )
        }

    async def _direct_answer(self, state: AgentState, config: RunnableConfig) -> dict:
        """直接回答（不需要工具）"""
        chain = DIRECT_ANSWER_PROMPT | self.llm_no_tools
        start_time = time.time()
        response = await chain.ainvoke({
            "messages": list(state["messages"]),
            "user_input": state["user_input"],
            "long_term_memory": state.get("long_term_memory") or "暂无",
            "context_summary": state.get("context_summary") or "",
            },
            config=config,
        )
        llm_elapsed = time.time() - start_time

        # 指标上报
        llm_call_duration.labels(model=settings.llm.model).observe(llm_elapsed)
        agent_iterations.observe(1)         # direct 路由只有 1 次 LLM 调用

        return {
            "messages":[
                HumanMessage(content=state["user_input"]),
                AIMessage(content=response.content),
            ],
            "final_answer": response.content,
            "needs_tool": False,
        }

    async def _rag_answer(self, state:AgentState, config: RunnableConfig) -> dict:
        """RAG 知识库回答"""
        if not state.get("rag_context"):
            logger.warning("[ReAct] RAG 上下文为空, 降级为直接回答")
            return await self._direct_answer(state)

        chain = RAG_SYNTHESIS_PROMPT | self.llm_no_tools
        start_time = time.time()
        response = await chain.ainvoke({
            "messages": list(state["messages"]),
            "user_input": state["user_input"],
            "rag_context": state.get("rag_context", "暂无知识库上下文"),
            "long_term_memory": state.get("long_term_memory") or "暂无",
            "context_summary": state.get("context_summary") or "",
            },
            config=config,
        )
        llm_elapsed = time.time() - start_time

        # 指标上报
        llm_call_duration.labels(model=settings.llm.model).observe(llm_elapsed)
        agent_iterations.observe(1)         # RAG 检索路由只有 1 次 LLM 调用

        return {
            "messages": [
                HumanMessage(content=state["user_input"]),
                AIMessage(content=response.content),
            ],
            "final_answer": response.content,
            "needs_tool": False,
        }

    async def _react_think(self, state: AgentState, config: RunnableConfig) -> dict:
        """
        ReAct 推理主逻辑：
        调用绑定工具的 LLM，解析是否需要工具调用
        """
        iteration = state.get("iteration_count", 0)

        # 超限保护
        if iteration >= self.max_iterations:
            logger.warning(
                f"[ReAct] 超过最大迭代次数 {self.max_iterations}, 强制结束"
            )
            return {
                "needs_tool": False,
                "final_answer": "已达到最大推理步数，基于已有信息给出最终回答。",
                "iteration_count": iteration,
            }

        chain = REACT_PROMPT | self.llm

        start_time = time.time()
        response = await chain.ainvoke(self._build_react_prompt_vars(state), config=config)
        llm_elapsed = time.time() - start_time

        # 上报LLM推理时间
        llm_call_duration.labels(model=settings.llm.model).observe(llm_elapsed)

        # 上报Token用量（LangChain 的 usage_metadata）
        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            llm_tokens_total.labels(
                model=settings.llm.model, type="input"
            ).inc(usage.get("input_tokens", 0))
            llm_tokens_total.labels(
                model=settings.llm.model, type="output"
            ).inc(usage.get("output_tokens", 0))


        logger.info(
            f"[ReAct] iteration={iteration} "
            f"has_tool_calls={bool(response.tool_calls)}"
            f" llm_elapsed={llm_elapsed * 1000:.4f} ms"
        )

        # 情况1：LLM 决定调用工具
        if response.tool_calls:
            # 只处理第一个工具调用（ReAct 每步只执行一个 Action）
            tool_call = response.tool_calls[0]
            pending_tool = {
                "id": tool_call["id"],
                "name": tool_call["name"],
                "args": tool_call["args"],
            }
            logger.info(
                f"[ReAct] 工具调用: {pending_tool['name']} "
                f"args={json.dumps(pending_tool['args'], ensure_ascii=False)[:100]}"
            )
            return {
                "messages": [
                    HumanMessage(content=state["user_input"]),
                    response,  # AIMessage with tool_calls
                ],
                "needs_tool": True,
                "pending_tool": pending_tool,
                "current_thought": response.content or f"准备调用工具: {pending_tool['name']}",
                "iteration_count": iteration + 1,
            }

        # 情况2：LLM 直接给出最终答案
        else:
            # 在最终回答时上报总迭代次数
            agent_iterations.observe(iteration + 1)

            logger.info(f"[ReAct] 直接给出答案，content长度={len(response.content)}")
            return {
                "messages": [
                    HumanMessage(content=state["user_input"]),
                    AIMessage(content=response.content),
                ],
                "needs_tool": False,
                "final_answer": response.content,
                "iteration_count": iteration + 1,
            }

    async def run(self, state: AgentState, config: RunnableConfig) -> dict:
        route = state.get("route", RouteType.REACT)

        logger.info(
            f"[ReAct] 运行: session={state['session_id']} "
            f"route={route} iteration={state.get('iteration_count', 0)}"
        )

        try:
            if route == RouteType.DIRECT:
                return await self._direct_answer(state, config)
            elif route == RouteType.RAG:
                return await self._rag_answer(state, config)
            else:
                # REACT / HYBRID：走完整的 ReAct 循环
                return await self._react_think(state, config)

        except Exception as e:
            logger.error(f"[ReAct] 推理异常: {e}", exc_info=True)
            return {
                "needs_tool": False,
                "final_answer": "抱歉，推理过程中发生了错误，请稍后重试。",
                "error": str(e),
            }


react_loop_node = ReActLoopNode()

async def run(state: AgentState, config: RunnableConfig) -> dict:
    return await react_loop_node.run(state, config)

