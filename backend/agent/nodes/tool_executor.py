"""
工具执行节点
职责：
  1. 从 state.pending_tool 中取出待执行工具
  2. 带超时、异常捕获地执行工具
  3. 将执行结果写回 messages（ToolMessage）
  4. 更新 tool_calls_history
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Optional

from langchain_core.messages import ToolMessage
from langchain_core.tools import BaseTool

from backend.agent.state import AgentState, ToolCall
from backend.config import settings
from backend.monitoring.metrics import tool_calls_total, tool_duration
from backend.tools.registry import get_tool_by_name

logger = logging.getLogger(__name__)


class ToolExecutorNode:

    def __init__(self):
        self.timeout = settings.agent.tool_timeout

    async def _execute_tool(
            self,
            tool: BaseTool,
            args: dict,
    ) -> tuple[str, Optional[str]]:
        """
        执行工具，带超时控制
        返回: (output_str, error_str_or_None)
        """
        start_time = time.time()
        status = "success"
        try:
            # 带超时的异步执行
            if hasattr(tool, 'arun'):
                result = await asyncio.wait_for(
                    tool.arun(args),
                    timeout=self.timeout,
                )
            else:
                # 同步工具转异步执行
                result = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(
                        None, lambda: tool.run(args)
                    ),
                    timeout=self.timeout
                )

            # 统一转字符串
            if isinstance(result, str):
                return result, None
            return json.dumps(result, ensure_ascii=False, indent=2), None

        except asyncio.TimeoutError:
            status = "timeout"
            err = f"工具 {tool.name} 执行超时（>{self.timeout}s）"
            logger.error(f"[ToolExecutorNode] {err}")
            return "", err

        except Exception as e:
            status = "error"
            err = f"工具 {tool.name} 执行失败: {type(e).__name__}: {str(e)}"
            logger.error(f"[ToolExecutorNode] {err}", exc_info=True)
            return "", err
        finally:
            # 上报工具调用指标
            elapsed = time.time() - start_time
            tool_calls_total.labels(tool=tool.name, status=status).inc()
            tool_duration.labels(tool=tool.name).observe(elapsed)

    async def run(self, state: AgentState) -> dict:
        pending = state.get("pending_tool")

        # 没有待执行工具
        if not pending:
            logger.warning("[ToolExecutor] pending_tool 为空，跳过")
            return {"need_tool": False}

        tool_id = pending.get("id", "")
        tool_name = pending.get("name", "")
        tool_args = pending.get("args", {})

        logger.info(
            f"[ToolExecutor] 执行工具: {tool_name} "
            f"args={json.dumps(tool_args, ensure_ascii=False)[:100]}"
        )

        start_time = time.time()

        # 1. 查找工具
        tool = get_tool_by_name(tool_name)
        if tool is None:
            error_msg = f"工具 '{tool_name}' 不存在，可用工具请查看工具列表"
            logger.error(f"[ToolExecutor] {error_msg}")
            tool_output = error_msg
            error = error_msg
            elapsed_ms = 0.0
        else:
            # 2. 执行
            tool_output, error = await self._execute_tool(tool, tool_args)
            elapsed_ms = (time.time() - start_time) * 1e3

        logger.info(
            f"[ToolExecutor] 完成: tool={tool_name} "
            f"elapsed={elapsed_ms:.1f}ms "
            f"success={error is None} "
            f"output_len={len(tool_output)}"
        )

        # 3. 构建 ToolMessage（写回 messages，让 LLM 看到工具结果）
        tool_msg_content = tool_output if not error else f"[错误] {error}"
        tool_message = ToolMessage(
            content=tool_msg_content,
            tool_call_id=tool_id,
            name=tool_name,
        )

        # 4. 构建工具调用记录
        tool_call_record = ToolCall(
            tool_name=tool_name,
            tool_input=tool_args,
            tool_output=tool_output if not error else None,
            error=error,
            elapsed_ms=elapsed_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

        # 5. 更新历史
        history = list(state.get("tool_calls_history", []))
        history.append(tool_call_record)

        return {
            "messages": tool_message,
            "tool_calls_history": history,
            "pending_tool": None,   # 清空，等待下一步决策
            "needs_tool": False,  # 由 react_loop 在下一步决定
        }


tool_executor_node = ToolExecutorNode()

async def run(state: AgentState) -> dict:
    return await tool_executor_node.run(state)
