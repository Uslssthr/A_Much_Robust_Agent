from __future__ import annotations

import json
import logging
import re

from langchain_openai import ChatOpenAI

from backend.agent.prompts import ROUTER_PROMPT
from backend.agent.state import RouteType, AgentState
from backend.config import settings

logger = logging.getLogger(__name__)


# 基于关键词的规则路由（快速、低成本）

RULE_BASED_ROUTES: dict[RouteType, list[str]] = {
    RouteType.DIRECT: [
        "你好", "hi", "hello", "谢谢", "thanks",
        "再见", "bye", "今天几号", "几点了",
        "你是谁", "你叫什么", "介绍一下你自己",
    ],
    RouteType.REACT: [
        "计算", "搜索", "查询", "执行", "运行",
        "帮我找", "最新", "实时", "现在的价格",
        "calculate", "search", "find", "execute", "run",
        "current price", "latest news",
    ],
    RouteType.RAG: [
        "根据文档", "知识库", "文件里", "资料中",
        "我们的产品", "公司规定", "内部文档",
        "根据规定", "政策是什么",
    ],
}


class RouterNode:

    def __init__(self):
        self.llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

    def _rule_based_route(self, user_input: str) -> RouteType | None:
        """基于规则的快速路由（优先于 LLM，节省成本）"""
        text_lower = user_input.lower()
        for route, keywords in RULE_BASED_ROUTES.items():
            for kw in keywords:
                if kw.lower() in text_lower:
                    logger.info(f"[Router] rule_match: '{kw}' -> {route}")
                    return route
        return None

    def _parse_llm_response(self, raw: str) -> tuple[RouteType, str, float]:
        """解析 LLM JSON 输出，带容错"""
        # 尝试提取 JSON 块
        json_match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if not json_match:
            logger.warning(f"[Router] LLM 返回非 JSON: {raw[:100]}")
            return RouteType.REACT, "解析失败，默认 ReAct", 0.5

        try:
            data = json.loads(json_match.group())
            route_str = data.get("route", "react").lower()
            reasoning = data.get("reasoning", "")
            confidence = float(data.get("confidence", 0.7))

            # 映射到枚举
            route_map = {
                "direct": RouteType.DIRECT,
                "react": RouteType.REACT,
                "rag": RouteType.RAG,
                "hybrid": RouteType.HYBRID,
            }
            route = route_map.get(route_str, RouteType.REACT)
            return route, reasoning, confidence

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"[Router] 解析 JSON 失败: {e}")
            return RouteType.REACT, "解析异常", 0.5


    async def run(self, state: AgentState) -> dict:
        user_input = state["user_input"]
        long_term_memory = state["long_term_memory"] or "暂无长期记忆"

        # 1. 先尝试规则路由（轻量快速）
        rule_route = self._rule_based_route(user_input)
        if rule_route is not None:
            return {
                "route": rule_route,
                "current_thought": f"规则路由 -> {rule_route}",
            }


        # 2. 规则未命中，使用 LLM 路由
        try:
            chain = ROUTER_PROMPT | self.llm
            response = await  chain.ainvoke({
                "messages": list(state["messages"]),
                "user_input": user_input,
                "long_term_memory": long_term_memory,
            })
            route, reasoning, confidence = self._parse_llm_response(response.content)

            logger.info(
                f"[Router] LLM决策: {route} "
                f"confidence={confidence:.2f} "
                f"reason={reasoning}"
            )

            # 置信度低于 0.5 时降级为 react（最通用）
            if confidence < 0.5:
                route = RouteType.REACT
                reasoning = "置信度不足，降级为 ReAct 路由"

            return {
                "route": route,
                "current_thought": f"LLM路由 -> {route} {reasoning}",
            }

        except Exception as e:
            logger.error(f"[Router] LLM 路由失败: {e}")
            # 兜底：默认 react 路径
            return {
                "route": RouteType.REACT,
                "current_thought": f"[Router异常，降级] error={str(e)}",
            }


router_node = RouterNode()

async def run(state: AgentState) -> dict:
    return await router_node.run(state)

