# backend/security/output_filter.py
"""
输出安全过滤器
防止 LLM 在回答中泄露敏感信息（系统提示、API Key、内部路径等）
"""
from __future__ import annotations
import re
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class OutputFilterResult:
    content:    str
    filtered:   bool          # 是否触发了过滤
    reason:     str = ""


# ── 敏感信息正则 ──────────────────────────────────────────────────────────────
_SENSITIVE_PATTERNS: list[tuple[str, str, str]] = [
    # (pattern, replacement, description)
    (
        r"(sk-[A-Za-z0-9]{20,})",
        "[API_KEY_REDACTED]",
        "OpenAI API Key",
    ),
    (
        r"(tvly-[A-Za-z0-9]{20,})",
        "[TAVILY_KEY_REDACTED]",
        "Tavily API Key",
    ),
    (
        r"([A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,})",
        "[EMAIL_REDACTED]",
        "邮箱地址",
    ),
    (
        r"\b(\d{11})\b",            # 手机号（11位数字）
        "[PHONE_REDACTED]",
        "手机号",
    ),
    (
        r"\b(\d{17}[\dXx])\b",      # 身份证号
        "[ID_REDACTED]",
        "身份证号",
    ),
    (
        # Windows/Linux 绝对路径
        r"([C-Z]:\\(?:[^\\/:*?\"<>|\r\n]+\\)*[^\\/:*?\"<>|\r\n]*"
        r"|/(?:home|root|etc|var|usr)/\S+)",
        "[PATH_REDACTED]",
        "系统路径",
    ),
    (
        # 私有 IP 地址段
        r"\b((?:192\.168|10\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01]))"
        r"\.\d{1,3}\.\d{1,3})\b",
        "[PRIVATE_IP_REDACTED]",
        "内网IP",
    ),
]


class OutputSafetyFilter:
    """
    输出安全过滤器
    在 LLM 回答返回给用户之前执行过滤

    用法：
        filter = OutputSafetyFilter()
        result = filter.filter(llm_output)
        return result.content   # 过滤后的安全内容
    """

    def filter(self, content: str) -> OutputFilterResult:
        """
        对 LLM 输出执行脱敏处理

        Args:
            content: LLM 原始输出

        Returns:
            OutputFilterResult，content 字段为处理后的安全文本
        """
        filtered   = False
        reasons    = []
        result     = content

        for pattern, replacement, desc in _SENSITIVE_PATTERNS:
            new_result, count = re.subn(pattern, replacement, result)
            if count > 0:
                filtered = True
                reasons.append(f"脱敏 {desc}（{count} 处）")
                result = new_result
                logger.warning(
                    f"[OutputFilter] 脱敏 {desc}: {count} 处"
                )

        return OutputFilterResult(
            content=  result,
            filtered= filtered,
            reason=   "；".join(reasons),
        )

    def is_safe(self, content: str) -> bool:
        """快速判断是否需要过滤（不做实际替换）"""
        for pattern, _, _ in _SENSITIVE_PATTERNS:
            if re.search(pattern, content):
                return False
        return True