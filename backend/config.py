import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv

load_dotenv()

@dataclass
class LLMConfig:
    model: str = os.getenv("LLM_MODEL", "deepseek-chat")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.7"))
    max_tokens: int = int(os.getenv("LLM_MAX_TOKENS", "2048"))
    streaming: bool = True
    api_key : str = field(default_factory=lambda : os.getenv("DEEPSEEK_API_KEY", ""))
    base_url : Optional[str] = os.getenv("DEEPSEEK_BASE_URL", None)

@dataclass
class ContextConfig:
    # Token 阈值
    soft_limit: int = 6000      # 触发轻度压缩
    hard_limit: int = 8000      # 触发重度压缩
    max_limit: int = 10000      # 强制截断
    # 压缩后保留最近的消息条数
    keep_last_n: int = 6
    # 滑动窗口保留token数
    sliding_window_tokens: int = 4000

@dataclass
class AgentConfig:
    # ReAct 最大迭代次数
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "20"))
    # 工具执行超时（秒）
    tool_timeout: int = int(os.getenv("TOOL_TIMEOUT", "30"))
    # 是否启用长期记忆
    enable_long_term_memory:bool = True
    # 是否启用 RAG
    enable_rag: bool = True

@dataclass
class RAGConfig:
    top_k: int = 4                              # 检索 Top-K 文档
    score_threshold: float = 0.3                # 相关性阈值
    chunk_size: int = 512
    chunk_overlap: int = 64
    embedding_model: str = "BGE-M3"
    base_url: str = "https://oai.endpoints.kepler.ai.cloud.ovh.net/v1"
    chroma_dir: str = os.getenv("CHROMA_DIR", "./data/chroma")

@dataclass
class DatabaseConfig:
    sqlite_path: str = os.getenv("SQLITE_PATH", "./data/agent.db")
    redis_url: str = os.getenv("REDIS_URL", "redis://localhost:6379")

@dataclass
class SafetyConfig:
    max_input_length: int = 4000
    enable_injection_detection: bool = True
    enable_sensitive_word_filter: bool = True

@dataclass
class ToolsConfig:
    tavily_api_key:          str  = field(default_factory=lambda: os.getenv("TAVILY_API_KEY", ""))
    tavily_max_results:      int  = int(os.getenv("TAVILY_MAX_RESULTS", "3"))
    tavily_search_depth:     str  = os.getenv("TAVILY_SEARCH_DEPTH", "basic")
    tavily_include_answer:   bool = True
    tavily_include_raw_content: bool = False
    tavily_topic:            str  = "general"

    # ✅ 新增：代理配置
    http_proxy:  str = field(default_factory=lambda: os.getenv("HTTP_PROXY",  ""))
    https_proxy: str = field(default_factory=lambda: os.getenv("HTTPS_PROXY", ""))

    @property
    def proxy_url(self) -> str | None:
        """优先取 https_proxy，其次 http_proxy，都没有返回 None"""
        return self.https_proxy or self.http_proxy or None

@dataclass
class AppConfig:
    llm : LLMConfig = field(default_factory=LLMConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    rag: RAGConfig = field(default_factory=RAGConfig)
    db: DatabaseConfig = field(default_factory=DatabaseConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    debug: bool = bool(os.getenv("DEBUG", "false").lower() == "true")


# 全局单例
settings = AppConfig()