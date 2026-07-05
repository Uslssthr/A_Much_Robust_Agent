"""
对话 API
POST /api/v1/chat         → 流式 SSE 对话
GET  /api/v1/chat/history/{session_id} → 获取消息历史
GET  /api/v1/chat/sessions             → 获取会话列表
DELETE /api/v1/chat/sessions/{session_id} → 删除会话
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from datetime import datetime

from fastapi import APIRouter, Request, HTTPException
from fastapi.params import Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend.agent.state import create_initial_state
from backend.db.sqlite_manager import db
from backend.memory.long_term import long_term_memory
from backend.mq.redis_client import redis_client
from backend.mq.task_producer import task_producer
from backend.security.input_filter import InputSafetyFilter, SafetyResult
from backend.monitoring.metrics import (
    agent_requests_total, route_distribution,
    safety_blocks_total, active_sessions,
)

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/chat", tags=["Chat"])
safety = InputSafetyFilter()


# 请求/响应模型

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    session_id: str | None = Field(None, description="会话ID， 为空则自动创建新会话")
    user_id: str = Field("anonymous", description="用户ID， 为空则默认匿名用户")
    collection: str = Field("default", description="知识库集合名， 为空则使用默认集合")


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    route: str
    tool_calls: list[dict]
    token_count: int


# SSE事件构建函数
def _sse(event_type: str, data: dict) -> str:
    """构建 SSE 消息帧"""
    payload = json.dumps({"type": event_type, **data}, ensure_ascii=False)
    return f"data: {payload}\n\n"

def _sse_token(content: str) -> str:
    return _sse("token", {"content": content})

def _sse_tool_start(tool: str, args:dict) -> str:
    return _sse("tool_start", {"tool": tool, "args": args})

def _sse_tool_end(tool: str, output: str, elapsed_ms: float) -> str:
    return _sse("tool_end", {
        "tool": tool,
        "output": output[:300],
        "elapsed_ms": round(elapsed_ms, 2),
    })

def _sse_meta(session_id: str, route: str, iteration: int, overflow: bool) -> str:
    return _sse("meta", {
        "session_id": session_id,
        "route": route,
        "iteration": iteration,
        "compressed": overflow,
    })

def _sse_error(msg: str) -> str:
    return _sse("error", {"message": msg})

def _sse_done(session_id: str) -> str:
    return _sse("done", {"session_id": session_id})



# 主对话接口

@router.post("")
async def chat(req: ChatRequest, request: Request):
    """
    流式对话接口（SSE）
    客户端接收格式：
      data: {"type":"token","content":"..."}\n\n
      data: {"type":"tool_start","tool":"...","args":{...}}\n\n
      data: {"type":"tool_end","tool":"...","output":"...","elapsed_ms":123}\n\n
      data: {"type":"meta","session_id":"...","route":"...",...}\n\n
      data: {"type":"done","session_id":"..."}\n\n
    """
    # 0. 速率限制
    try:
        allowed, remaining = await redis_client.check_rate_limit(
            req.user_id, max_requests=30, window_size=60
        )
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail="请求过于频繁，请稍后再试",
            )
    except HTTPException:
        raise
    except Exception:
        pass

    # 1. 安全检查
    safety_result = safety.check(req.message)
    if not safety_result.passed:
        safety_blocks_total.labels(risk_level=safety_result.risk_level).inc()
        raise HTTPException(
            status_code=400,
            detail={
                "error": safety_result.reason,
                "risk_level": safety_result.risk_level,
            }
        )

    session_id = req.session_id or str(uuid.uuid4())
    agent = request.app.state.agent_graph
    config = {"configurable": {"thread_id": session_id}}

    active_sessions.inc()       # 活跃会话 +1

    # 2. 确保会话存在
    db.upsert_session(
        session_id=session_id,
        user_id=req.user_id,
        message_count=1,
    )

    # 3. 召回长期记忆
    long_memory_text = long_term_memory.recall_as_text(req.user_id)

    # 4. 构造初始状态
    init_state = create_initial_state(
        user_input=req.message,
        session_id=session_id,
        user_id=req.user_id,
    )
    if long_memory_text:
        init_state["long_term_memory"] = long_memory_text

    # 5. 保存用户消息
    db.save_message(
        session_id=session_id,
        role="user",
        content=req.message,
    )

    async def event_generator():
        full_answer = []
        tool_calls_log = []
        final_state = {}

        try:
            # 使用 astream_events 获取细粒度事件流
            async for event in agent.astream_events(
                init_state,
                config=config,
                version="v2",
            ):
                kind = event["event"]
                name = event.get("name", "")

                # Token 流式输出

                if kind == "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    token = chunk.content if hasattr(chunk, "content") else ""
                    if token:
                        full_answer.append(token)
                        yield _sse_token(token)

                # 工具开始执行

                elif kind == "on_tool_start":
                    tool_input = event["data"].get("input", {})
                    tool_calls_log.append({
                        "tool": name,
                        "args": tool_input,
                        "start": datetime.utcnow().isoformat(),
                    })
                    yield _sse_tool_start(name, tool_input)

                # 工具执行完成

                elif kind == "on_tool_end":
                    output = str(event["data"].get("output", ""))
                    # 计算耗时（简单估算）
                    if tool_calls_log:
                        last = tool_calls_log[-1]
                        last["output"] = output
                        elapsed_ms = (
                            datetime.utcnow() -
                            datetime.fromisoformat(last["start"])
                        ).total_seconds() * 1000
                    else:
                        elapsed_ms = -1
                    yield _sse_tool_end(name, output, elapsed_ms)

                # 图运行结束，读取最终状态

                elif kind == "on_chain_end" and name == "LangGraph":
                    final_state = event["data"].get("output", {})

        except asyncio.CancelledError:
            logger.info(f"[Chat] 客户端断开连接：session_id={session_id}")
            return

        except Exception as e:
            logger.error(f"[Chat] 流式生成异常：{e}", exc_info=True)
            yield _sse_error(f"生成过程中发生错误：{str(e)[:100]}")
            return
        finally:
            active_sessions.dec()           # 活跃会话 -1

        # 流结束后的后处理

        answer = "".join(full_answer) or final_state.get("final_answer", "")
        route = str(final_state.get("route", "unknown"))
        route_distribution.labels(route=route).inc()
        agent_requests_total.labels(route=route, status="success").inc()

        # 发送元数据帧
        yield _sse_meta(
            session_id=session_id,
            route=route,
            iteration=final_state.get("iteration_count", 0),
            overflow=final_state.get("context_overflow", False),
        )

        # 保存助手回复到 SQLite
        if answer:
            db.save_message(
                session_id=session_id,
                role="assistant",
                content=answer,
            )

        # 保存工具调用日志
        for tc in (final_state.get("tool_call_history") or []):
            import json as _json
            db.save_tool_log(
                session_id=session_id,
                tool_name=tc.get("tool_name", ""),
                tool_input=_json.dumps(tc.get("tool_input", {}), ensure_ascii=False),
                tool_output=tc.get("tool_output", ""),
                error=tc.get("error"),
                elapsed_ms=tc.get("elapsed_ms", -1),
            )

        # 提取长期记忆（异步，不阻塞响应）
        try:
            await task_producer.submit_memory_extract(
                user_id=req.user_id,
                session_id=session_id,
                user_input=req.message,
                assistant_response=answer,
            )
        except Exception as e:
            logger.warning(f"[Chat] 记忆提取任务提交失败: {e}")

        yield _sse_done(session_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",      # 禁止nginx缓冲
            "X-Session-ID": session_id,
        }
    )

# 非流式接口
@router.post("/sync", response_model=ChatResponse)
async def chat_sync(req: ChatRequest, request: Request):
    """非流式对话（适合测试或简单场景）"""
    safety_result = safety.check(req.message)
    if not safety_result.passed:
        raise HTTPException(status_code=400, detail=safety_result.reason)

    session_id = req.session_id or str(uuid.uuid4())
    agent = request.app.state.agent_graph
    config = {"configurable": {"thread_id": session_id}}

    init_state = create_initial_state(
        user_input=req.message,
        session_id=session_id,
        user_id=req.user_id,
    )
    long_memory = long_term_memory.recall_as_text(req.user_id)
    if long_memory:
        init_state["long_term_memory"] = long_memory

    result = await agent.ainvoke(init_state, config=config)

    answer = result.get("final_answer", "")
    db.upsert_session(session_id, req.user_id, message_count=2)
    db.save_message(session_id, "user", req.message)
    db.save_message(session_id, "assistant", answer)

    return ChatResponse(
        session_id=session_id,
        answer=answer,
        route=str(result.get("route", "")),
        tool_calls=[
            {
                "tool": tc.get("tool_name", ""),
                "input": tc.get("tool_input", {}),
                "output": tc.get("tool_output", "")[:200],
            }
            for tc in (result.get("tool_call_history") or [])
        ],
        token_count=result.get("context_token_count", 0),
    )


# 历史查询接口

@router.get("/history/{session_id}")
async def get_history(
    session_id: str,
    limit: int = Query(default=50, ge=1, le=200),
):
    """获取指定会话的消息历史"""
    messages = db.get_messages(session_id, limit=limit)
    if not messages:
        raise HTTPException(status_code=404, detail="会话不存在或无消息记录")
    return {
        "session_id": session_id,
        "count": len(messages),
        "messages": messages,
    }


@router.get("/sessions")
async def list_sessions(
    user_id: str = Query(default="anonymous"),
    limit: int = Query(default=20, ge=1, le=100),
):
    """获取用户的会话列表"""
    sessions = db.get_sessions(user_id=user_id, limit=limit)
    return {
        "user_id": user_id,
        "count": len(sessions),
        "sessions": sessions,
    }


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    """删除指定会话"""
    db.delete_session(session_id)
    return {"message": f"会话 {session_id} 已删除成功"}


# 内部辅助函数
async def _extract_and_save_memory(
    user_id: str,
    session_id: str,
    user_input: str,
    assistant_response: str,
):
    """
    后台任务：从对话中提取值得长期记忆的信息
    使用简单规则提取（生产环境可接入 LLM 做更精准提取）
    """
    try:
        # 检测用户偏好表达
        preference_keywords = [
            "我喜欢", "我不喜欢", "我习惯", "我偏好", "请你以后",
            "记住我", "我是", "我的职业", "我在"
        ]
        for kw in preference_keywords:
            if kw in user_input:
                long_term_memory.save(
                    user_id=user_id,
                    session_id=session_id,
                    content=f"用户说：{user_input}",
                    memory_type="preference",
                    importance=0.8,
                )
                break
    except Exception as e:
        logger.warning(f"[Memory] 记忆提取失败：{e}")
