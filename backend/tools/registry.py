"""
工具注册表：统一管理所有工具的注册、查找、描述
"""
from __future__ import annotations

import logging
from typing import Optional

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

# 全局工具注册表
_tool_registry: dict[str, BaseTool] = {}


def register_tool(tool: BaseTool) -> None:
    """注册工具"""
    _tool_registry[tool.name] = tool
    logger.info(f"[ToolRegistry] 注册工具: {tool.name}")


def get_tool_by_name(name: str) -> Optional[BaseTool]:
    """按名称查找工具"""
    return _tool_registry.get(name)


def get_all_tools() -> list[BaseTool]:
    """获取所有已注册的工具"""
    return list(_tool_registry.values())


def format_tool_description(tools: list[BaseTool]) -> str:
    """格式化工具描述，注入 Prompt"""
    if not tools:
        return "当前没有可用工具"
    lines = []
    for tool in tools:
        args_schema = ""
        if tool.args_schema:
            schema = tool.args_schema.schema()
            props = schema.get("properties", {})
            args_schema = ", ".join([
                f"{k}({v.get('type', 'any')}): {v.get('description', '')}"
                for k, v in props.items()
            ])
        lines.append(
            f"• **{tool.name}**: {tool.description}\n"
            f"  参数: {args_schema or '无'}"
        )
    return "\n".join(lines)


def _auto_register_all():
    """自动注册所有工具（应用启动时调用）"""
    from backend.tools.search_tool import SearchTool
    from backend.tools.calculator_tool import CalculatorTool
    from backend.tools.knowledge_tool import KnowledgeQueryTool

    register_tool(SearchTool())
    register_tool(CalculatorTool())
    register_tool(KnowledgeQueryTool())
    logger.info(f"[Registry] 共注册 {len(_tool_registry)} 个工具")


# 模块加载时自动注册
_auto_register_all()
