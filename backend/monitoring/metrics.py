"""
Prometheus 监控指标定义
所有指标集中在此，供各模块 import 使用
"""
from __future__ import annotations
from prometheus_client import (
    Counter, Histogram, Gauge, Info,
    generate_latest, CONTENT_TYPE_LATEST,
)

# HTTP请求指标

http_requests_total = Counter(
    "agent_http_requests_total",
    "HTTP 请求总数",
    ["method", "endpoint", "status"]
)

http_requests_duration = Histogram(
    "agent_http_request_duration_seconds",
    "HTTP请求延迟",
    ["method", "endpoint"],
    buckets=[0.05, 0.1, 0.3, 0.5, 1, 2, 5, 10, 30],
)


# Agent核心指标

agent_requests_total = Counter(
    "agent_requests_total",
    "Agent 处理请求总数",
    ["route", "status"],
)

agent_iterations = Histogram(
    "agent_react_iterations",
    "ReAct 迭代次数分布",
    buckets=[1, 2, 3, 4, 5, 6, 8, 10],
)

active_sessions = Gauge(
    "agent_active_sessions",
    "当前活跃会话数",
)


# 路由分布

route_distribution = Counter(
    "agent_route_total",
    "路由决策分布",
    ["route"],      # direct / rag / react / hybrid
)


# LLM&Token指标
llm_tokens_total = Counter(
    "agent_llm_tokens_total",
    "LLM Token 消耗总数",
    ["model", "type"],      # model_name, type (input/output)
)

llm_call_duration = Histogram(
    "agent_llm_call_duration_seconds",
    "LLM调用延迟",
    ["model"],
    buckets=[0.5, 1, 2, 3, 4, 10, 20, 30],
)


# 工具调用指标

tool_calls_total = Counter(
    "agent_tool_calls_total",
    "工具调用总数",
    ["tool", "status"],         # tool_name, status (success/error/timeout)
)

tool_duration = Histogram(
    "agent_tool_duration_seconds",
    "工具执行延迟",
    ["tool"],
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30],
)


# RAG指标
rag_retrieval_duration = Histogram(
    "agent_rag_retrieval_seconds",
    "RAG 检索延迟",
    buckets=[0.05, 0.1, 0.3, 0.5, 1, 2],
)

rag_retrieved_docs = Histogram(
"agent_rag_retrieved_docs",
    "每次检索返回文档数",
    buckets=[0, 1, 2, 3, 4, 5, 10],
)


# 上下文工程指标

context_compressions_total = Counter(
    "agent_context_compressions_total",
    "上下文压缩触发总数",
    ["strategy"],           # sliding_window / summarize / focus_fold
)

context_token_count = Histogram(
    "agent_context_tokens",
    "上下文Token数分布",
    buckets=[500, 1000, 2000, 4000, 6000, 8000, 10000],
)


# 安全指标

safety_blocks_total = Counter(
    "agent_safety_blocks_total",
    "安全拦截次数",
    ["risk_level"],         # low / medium / high
)


# 任务队列指标

task_processed_total = Counter(
    "agent_task_processed_total",
    "已处理任务总数",
    ["type", "status"],
)

task_duration_seconds = Histogram(
    "agent_task_duration_seconds",
    "任务处理延迟",
    ["type"],
    buckets=[0.5, 1, 2, 5, 10, 30, 60, 120],
)

task_queue_length = Gauge(
    "agent_task_queue_length",
    "任务队列长度",
)


# 应用信息

app_info = Info("agent_app", "应用信息")
app_info.info({"version": "1.0.0", "name": "universal-react-agent"})


def get_metrics() -> tuple[bytes, str]:
    """返回 Prometheus 格式指标（供 /metrics 接口使用）"""
    return generate_latest(), CONTENT_TYPE_LATEST