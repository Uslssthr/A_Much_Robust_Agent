"""
工具基类
"""
import logging
from abc import ABC, abstractmethod

from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)

class BaseAgentTool(BaseTool, ABC):
    """
    所有 Agent 工具的基类
    统一提供：日志记录、错误处理、输入验证
    """

    @abstractmethod
    def _run(self, *args, **kwargs) -> str:
        """同步执行（子类必须实现）"""
        ...

    async def _arun(self, *args, **kwargs) -> str:
        """
        默认异步实现：包装同步版本
        需要真正异步的工具应重写此方法
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None, lambda: self._run(*args, **kwargs)
        )

    def run(self, tool_input: dict | str, **kwargs) -> str:
        """带统一日志的 run"""
        logger.info(f"[Tool:{self.name}] input={str(tool_input)[:100]}")
        try:
            result = super().run(tool_input, **kwargs)
            logger.info(f"[Tool:{self.name}] success output_len={len(str(result))}")
            return result
        except Exception as e:
            logger.error(f"[Tool:{self.name}] error: {e}")
            raise e
