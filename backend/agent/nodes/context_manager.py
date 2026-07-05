"""
上下文管理节点：多级压缩策略
────────────────────────────────────────
策略 1：无需压缩（< soft_limit）       → 原样通过
策略 2：滑动窗口（soft_limit ~ hard_limit）→ 保留最近 N token
策略 3：摘要压缩（hard_limit ~ max_limit）→ 旧消息摘要化
策略 4：Focus 折叠（> max_limit）       → 强制折叠 + 摘要
"""

from __future__ import annotations

import logging

import tiktoken
from deepseek_tokenizer import ds_token
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_openai import ChatOpenAI

from backend.agent.prompts import SUMMARIZATION_PROMPT
from backend.agent.state import AgentState
from backend.config import settings
from backend.monitoring.metrics import context_token_count, context_compressions_total

logger = logging.getLogger(__name__)


class ContextManagerNode:

    def __init__(self):
        cfg = settings.context
        self.soft_limit         = cfg.soft_limit
        self.hard_limit         = cfg.hard_limit
        self.max_limit          = cfg.max_limit
        self.keep_last_n        = cfg.keep_last_n
        self.sliding_tokens     = cfg.sliding_window_tokens

        self.llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

        # tiktoken 编码器（gpt-4o 兼容）
        try:
            self.encoder = ds_token
        except KeyError:
            self.encoder = tiktoken.get_encoding("cl100k_base")

    # Token 计数
    def _count_tokens(self, messages: list[BaseMessage]) -> int:
        total = 0
        for msg in messages:
            if isinstance(msg.content, str):
                total += len(self.encoder.encode(msg.content))
            # 多模态消息（list content）只计文字部分
            elif isinstance(msg.content, list):
                for block in msg.content:
                    if isinstance(block, dict) and block.get("type") == "text":
                        total += len(self.encoder.encode(block["text"]))

        return total


    def _msg_to_text(self, messages: list[BaseMessage]) -> str:
        """将消息列表转为可读文本（用于摘要）"""
        lines = []
        for msg in messages:
            role = {
                HumanMessage:   "用户",
                AIMessage:      "助手",
                SystemMessage:  "系统",
            }.get(type(msg), "未知")
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            lines.append(f"[{role}]: {content}")

        return "\n".join(lines)


    # 策略 1： 无操作
    def _pass_through(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], str]:
        return messages, "none"

    # 策略 2： 滑动窗口
    def _sliding_window(self, messages: list[BaseMessage]) -> tuple[list[BaseMessage], str]:
        """
        从最新消息向前回溯，直到 token 数达到上限
        始终保留 SystemMessage
        """
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        result = []
        tokens = self._count_tokens(system_msgs)

        # 从后向前遍历
        for msg in reversed(other_msgs):
            t = len(self.encoder.encode(
                msg.content if isinstance(msg.content, str) else ""
            ))
            if tokens + t > self.sliding_tokens:
                break
            result.insert(0, msg)
            tokens += t

        final = system_msgs + result
        logger.info(
            f"[ContextManager] sliding_window: "
            f"{len(messages)} msgs → {len(final)} msgs, "
            f"~{tokens} tokens"
        )
        return final, "sliding_window"

    # 策略 3： 摘要压缩
    async def _summarize_compress(
            self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], str, str]:
        """
        摘要前 N 条消息，保留最近 keep_last_n 条
        返回: (压缩后消息列表, 策略名, 摘要文本)
        """
        system_msgs = [m for m in messages if isinstance(m, SystemMessage)]
        other_msgs = [m for m in messages if not isinstance(m, SystemMessage)]

        if len(other_msgs) <= self.keep_last_n:
            return messages, "none", ""

        old_msgs = other_msgs[:-self.keep_last_n]
        keep_msgs = other_msgs[-self.keep_last_n:]

        # 调用 LLM 生成摘要
        history_text = self._msg_to_text(old_msgs)
        chain = SUMMARIZATION_PROMPT | self.llm
        response = await chain.ainvoke({"history_text": history_text})
        summary = response.content

        # 将摘要注入为 System 消息
        summary_msg = SystemMessage(
            content=f"## 历史对话摘要（已压缩）\n{summary}"
        )

        final = system_msgs + [summary_msg] + keep_msgs
        logger.info(
            f"[ContextManager] summarize: "
            f"{len(old_msgs)} old msgs → 1 summary msg, "
            f"kept {len(keep_msgs)} recent msgs"
        )
        return final, "summarize", summary

    # 策略 4： Focus 折叠（强制）
    async def _focus_fold(
            self, messages: list[BaseMessage]
    ) -> tuple[list[BaseMessage], str, str]:
        """
        将所有旧消息折叠，只保留最近 keep_last_n 条
        适用于超长上下文的强制裁剪
        """
        logger.warning(
            f"[ContextManager] focus_fold triggered! "
            f"messages={len(messages)}"
        )
        # 调用摘要压缩（keep_last_n 减半，更激进）
        self.keep_last_n = max(2, self.keep_last_n // 2)
        result, strategy, summary = await self._summarize_compress(messages)
        self.keep_last_n = settings.context.keep_last_n     # 恢复
        return result, "focus_fold", summary

    async def run(self, state: AgentState) -> dict:
        messages = list(state["messages"])
        token_count = self._count_tokens(messages)

        logger.info(
            f"[ContextManager] session={state['session_id']} "
            f"token_count={token_count} "
            f"thresholds=({self.soft_limit}/{self.hard_limit}/{self.max_limit})"
        )

        # 记录压缩前的token分布
        context_token_count.observe(token_count)

        summary = state.get("context_summary")
        overflow = False

        if token_count < self.soft_limit:
            # 无需处理
            messages, strategy = self._pass_through(messages)

        elif token_count < self.hard_limit:
            # 滑动窗口
            messages, strategy = self._sliding_window(messages)
            overflow = True

        elif token_count < self.max_limit:
            # 摘要压缩
            messages, strategy, summary = await self._summarize_compress(messages)
            overflow = True

        else:
            # Focus 折叠（强制）
            messages, strategy, summary = await self._focus_fold(messages)
            overflow = True

        # 上报压缩策略指标（none 不上报，只关心真正发生压缩的情况）
        if strategy != "none":
            context_compressions_total.labels(strategy=strategy).inc()
            logger.info(f"[ContextManager] 压缩触发：strategy={strategy}")

        new_token_count = self._count_tokens(messages)

        return {
            "messages": messages,
            "context_token_count": new_token_count,
            "context_overflow": overflow,
            "context_summary": summary,
            "compression_strategy": strategy,
        }


context_manager_node = ContextManagerNode()

async def run(state: AgentState) -> dict:
    return await context_manager_node.run(state)

