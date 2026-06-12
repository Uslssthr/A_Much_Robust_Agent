"""
安全检查节点：Pipeline 的第一道防线
职责：
  1. 输入长度检查
  2. Prompt 注入检测
  3. 敏感内容过滤
  4. 编码/混淆攻击检测
"""
from __future__ import annotations

import logging
import re
import unicodedata

from backend.agent.state import SafetyLevel, AgentState
from backend.config import settings

logger = logging.getLogger(__name__)


# 注入攻击正则模式（中英文双语）
INJECTION_PATTERNS: list[tuple[str, SafetyLevel]] = [
    # 高危：直接指令覆盖
    (r"ignore\s+(all\s+)?previous\s+instructions?",            SafetyLevel.HIGH),
    (r"disregard\s+(all\s+)?previous",                         SafetyLevel.HIGH),
    (r"forget\s+(all\s+)?previous",                            SafetyLevel.HIGH),
    (r"you\s+are\s+now\s+",                                    SafetyLevel.HIGH),
    (r"act\s+as\s+if\s+you\s+(are|have\s+no)",                 SafetyLevel.HIGH),
    (r"jailbreak",                                             SafetyLevel.HIGH),
    (r"\bdan\b.*mode",                                         SafetyLevel.HIGH),
    (r"do\s+anything\s+now",                                   SafetyLevel.HIGH),
    # 中文注入
    (r"忽略.{0,10}(之前|前面|上面).{0,10}(指令|规则|设定)",         SafetyLevel.HIGH),
    (r"你(现在|从现在起).{0,5}(是|变成|扮演)",                     SafetyLevel.HIGH),
    (r"忘记你.{0,10}(是|的|所有)",                               SafetyLevel.HIGH),
    # 中危：角色扮演绕过
    (r"pretend\s+(you\s+are|to\s+be)",                        SafetyLevel.MEDIUM),
    (r"roleplay\s+as",                                        SafetyLevel.MEDIUM),
    (r"simulate\s+(being|a)",                                 SafetyLevel.MEDIUM),
    (r"扮演.{0,10}(没有|不受|不受限制)",                          SafetyLevel.MEDIUM),
    # 低危：系统提示探测
    (r"(show|reveal|print|output)\s+(your\s+)?system\s+prompt", SafetyLevel.LOW),
    (r"(显示|输出|打印).{0,5}(系统|你的).{0,5}(提示|prompt)",        SafetyLevel.LOW),
    (r"<\s*(system|SYSTEM)\s*>",                                SafetyLevel.LOW),
    (r"\[SYSTEM\]",                                             SafetyLevel.LOW),
]


# 违禁内容关键词（高危)
FORBIDDEN_KEYWORDS: list[str] = [
    "炸弹制作", "爆炸物合成", "违禁药物合成", "制造毒品",
    "黑客攻击教程", "儿童色情", "人肉搜索",
    "how to make bomb", "synthesize drugs",
    "child exploitation",
]


class SafetyCheckNode:
    """安全检查节点"""

    def __init__(self):
        self.max_input_length = settings.safety.max_input_length

    def _normalize_text(self, text: str) -> str:
        """
        规范化文本：
        - Unicode 规范化（防止同形字符绕过）
        - 去除零宽字符
        - 转小写
        """
        # 去除零宽字符
        text = re.sub(r'[\u200b-\u200f\u202a-\u202a\ufeff]', '', text)
        # Unicode 规范化
        text = unicodedata.normalize('NFKC', text)
        return text.lower()

    def _check_length(self, text: str) -> tuple[bool, str, SafetyLevel]:
        """检查输入长度"""
        if len(text) > self.max_input_length:
            return (
                False,
                f"输入内容过长（{len(text)} 字符），最大允许 {self.max_input_length} 字符",
                SafetyLevel.LOW
            )
        return True, "", SafetyLevel.NONE

    def _check_injection(self, text: str) -> tuple[bool, str, SafetyLevel]:
        """检测 Prompt 注入攻击"""
        normalized = self._normalize_text(text)
        for pattern, level in INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE | re.DOTALL):
                return (
                    False,
                    f"检测到潜在的 Prompt 注入攻击（模式：{pattern[:30]}...）",
                    level,
                )

        return True, "", SafetyLevel.NONE

    def _check_forbidden(self, text: str) -> tuple[bool, str, SafetyLevel]:
        """检测违禁内容"""
        normalized = self._normalize_text(text)
        for keyword in FORBIDDEN_KEYWORDS:
            if keyword.lower() in normalized:
                return (
                    False,
                    f"输入包含违禁内容，已被系统拦截",
                    SafetyLevel.HIGH,
                )
        return True, "", SafetyLevel.NONE

    def _check_encoding_attack(self, text: str) -> tuple[bool, str, SafetyLevel]:
        """检测编码攻击（Base64/十六进制嵌套指令）"""
        import base64
        # 检测可疑的长 Base64 片段
        b64_pattern = r'[A-Za-z0-9+/]{50,}={0,2}'
        matches = re.findall(b64_pattern, text)
        for match in matches:
            try:
                decoded = base64.b64decode(match).decode('utf-8', errors='ignore')
                # 如果解码后包含注入特征
                for pattern, level in INJECTION_PATTERNS[:6]:       # 只检查高危模式
                    if re.search(pattern, decoded, re.IGNORECASE):
                        return (
                            False,
                            "检测到编码混淆攻击",
                            SafetyLevel.HIGH,
                        )
            except Exception:
                pass

        return True, "", SafetyLevel.NONE

    async def run(self, state: AgentState) -> dict:
        """节点入口：执行所有安全检查"""
        user_input = state["user_input"]
        logger.info(f"[SafetyCheck] session={state['session_id']} input_len={len(user_input)}")

        # 依次执行各项检查（短路：发现问题立即返回）
        checks = [
            self._check_length,
            self._check_injection,
            self._check_forbidden,
            self._check_encoding_attack,
        ]

        for check_fn in checks:
            passed, reason, level = check_fn(user_input)
            if not passed:
                logger.warning(
                    f"[SafetyCheck] BLOCKED session={state['session_id']} "
                    f"reason={reason} level={level}"
                )
                return {
                    "safety_passed": False,
                    "safety_reason": reason,
                    "safety_level": level,
                }

        logger.info(f"[SafetyCheck] PASSED session={state['session_id']}")
        return {
            "safety_passed": True,
            "safety_reason": None,
            "safety_level": SafetyLevel.NONE,
        }


# 节点实例（全局单例，供 graph.py 使用）
safety_check_node = SafetyCheckNode()

async def run(state: AgentState) -> dict:
    return await safety_check_node.run(state)

