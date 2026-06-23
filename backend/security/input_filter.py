# backend/security/input_filter.py
"""
输入安全过滤器
与 agent/nodes/safety_checker.py 的区别：
  - safety_checker.py  是 LangGraph 节点，在 Agent 图内部运行
  - input_filter.py    是 FastAPI 层的前置检查，在请求进入 Agent 之前运行
两者共用同一套检测逻辑，但职责分层：
  API层(input_filter) → Agent图层(safety_checker)
"""
from __future__ import annotations
import re
import unicodedata
import logging
from dataclasses import dataclass

from backend.config import settings

logger = logging.getLogger(__name__)


@dataclass
class SafetyResult:
    """安全检查结果"""
    passed:     bool
    reason:     str       = ""
    risk_level: str       = "none"   # none / low / medium / high


# ── Prompt 注入正则模式 ───────────────────────────────────────────────────────
_INJECTION_PATTERNS: list[tuple[str, str]] = [
    # 高危
    (r"ignore\s+(all\s+)?previous\s+instructions?",         "high"),
    (r"disregard\s+(all\s+)?previous",                      "high"),
    (r"forget\s+(everything|all).{0,20}(above|previous)",   "high"),
    (r"you\s+are\s+now\s+",                                 "high"),
    (r"act\s+as\s+if\s+you\s+(are|have\s+no)",              "high"),
    (r"\bjailbreak\b",                                       "high"),
    (r"\bdan\b.{0,10}mode",                                 "high"),
    (r"do\s+anything\s+now",                                 "high"),
    # 中文高危
    (r"忽略.{0,15}(之前|前面|上面).{0,10}(指令|规则|设定|限制)", "high"),
    (r"(你现在|从现在起|请你).{0,10}(是|变成|扮演|假装)",    "high"),
    (r"忘记你.{0,10}(是|的设定|所有规则)",                  "high"),
    # 中危
    (r"pretend\s+(you\s+are|to\s+be)",                      "medium"),
    (r"roleplay\s+as",                                       "medium"),
    (r"扮演.{0,10}(没有限制|不受约束|自由)",                 "medium"),
    # 低危
    (r"(show|reveal|print).{0,10}system\s+prompt",          "low"),
    (r"(显示|输出|打印).{0,5}(系统提示|system prompt)",     "low"),
    (r"<\s*(system|SYSTEM)\s*>",                             "low"),
    (r"\[SYSTEM\]",                                          "low"),
]

# ── 违禁关键词 ────────────────────────────────────────────────────────────────
_FORBIDDEN_KEYWORDS: list[str] = [
    "炸弹制作", "爆炸物合成", "违禁药物合成", "制造毒品",
    "儿童色情", "人肉搜索",
    "how to make a bomb", "synthesize illegal drugs",
    "child pornography",
]


class InputSafetyFilter:
    """
    API 层输入安全过滤器
    在请求进入 Agent 之前快速拦截明显的恶意输入

    用法：
        filter = InputSafetyFilter()
        result = filter.check(user_input)
        if not result.passed:
            raise HTTPException(400, result.reason)
    """

    def __init__(self):
        self.max_length = settings.safety.max_input_length

    # ── 内部检查方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(text: str) -> str:
        """
        规范化文本：
        1. 去除零宽字符（常用于绕过关键词检测）
        2. Unicode NFKC 规范化（处理同形字符）
        3. 转小写
        """
        # 去除零宽字符
        text = re.sub(r"[\u200b-\u200f\u202a-\u202e\ufeff\u00ad]", "", text)
        # Unicode 规范化
        text = unicodedata.normalize("NFKC", text)
        return text.lower()

    def _check_length(self, text: str) -> SafetyResult | None:
        """输入长度检查"""
        if len(text) > self.max_length:
            return SafetyResult(
                passed=False,
                reason=(
                    f"输入内容过长（{len(text)} 字符），"
                    f"最大允许 {self.max_length} 字符"
                ),
                risk_level="low",
            )
        if len(text.strip()) == 0:
            return SafetyResult(
                passed=False,
                reason="输入内容不能为空",
                risk_level="none",
            )
        return None

    def _check_injection(self, text: str) -> SafetyResult | None:
        """Prompt 注入检测"""
        normalized = self._normalize(text)
        for pattern, level in _INJECTION_PATTERNS:
            if re.search(pattern, normalized, re.IGNORECASE | re.DOTALL):
                logger.warning(
                    f"[InputFilter] 注入检测命中: pattern={pattern[:40]} level={level}"
                )
                return SafetyResult(
                    passed=False,
                    reason="检测到潜在的 Prompt 注入攻击，请求已被拦截",
                    risk_level=level,
                )
        return None

    def _check_forbidden(self, text: str) -> SafetyResult | None:
        """违禁内容检测"""
        normalized = self._normalize(text)
        for keyword in _FORBIDDEN_KEYWORDS:
            if keyword.lower() in normalized:
                logger.warning(f"[InputFilter] 违禁词命中: {keyword}")
                return SafetyResult(
                    passed=False,
                    reason="输入包含违禁内容，请求已被拦截",
                    risk_level="high",
                )
        return None

    def _check_encoding_attack(self, text: str) -> SafetyResult | None:
        """
        检测 Base64 / URL编码 混淆攻击
        攻击者常将恶意指令 Base64 编码后注入
        """
        import base64

        b64_pattern = r"[A-Za-z0-9+/]{40,}={0,2}"
        for match in re.findall(b64_pattern, text):
            try:
                decoded = base64.b64decode(match + "==").decode("utf-8", errors="ignore")
                decoded_norm = self._normalize(decoded)
                # 用高危模式检测解码后内容
                for pattern, level in _INJECTION_PATTERNS:
                    if level == "high" and re.search(
                        pattern, decoded_norm, re.IGNORECASE
                    ):
                        logger.warning("[InputFilter] Base64 编码攻击被拦截")
                        return SafetyResult(
                            passed=False,
                            reason="检测到编码混淆攻击，请求已被拦截",
                            risk_level="high",
                        )
            except Exception:
                pass
        return None

    # ── 主入口 ────────────────────────────────────────────────────────────────

    def check(self, text: str) -> SafetyResult:
        """
        执行完整安全检查，短路求值（发现问题立即返回）

        Args:
            text: 用户输入文本

        Returns:
            SafetyResult(passed=True)  → 通过
            SafetyResult(passed=False) → 被拦截，包含原因和风险等级
        """
        # 按严重程度从轻到重排列，先过滤简单情况
        checks = [
            self._check_length,
            self._check_forbidden,
            self._check_injection,
            self._check_encoding_attack,
        ]

        for check_fn in checks:
            result = check_fn(text)
            if result is not None:
                return result

        return SafetyResult(passed=True, risk_level="none")

    def check_batch(self, texts: list[str]) -> list[SafetyResult]:
        """批量检查（用于批量接口）"""
        return [self.check(t) for t in texts]