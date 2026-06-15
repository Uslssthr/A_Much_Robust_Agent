# backend/tools/search_tool.py
from __future__ import annotations
import logging
from typing import ClassVar, Literal, Any

import httpx
from pydantic import BaseModel, Field

from backend.tools.base_tool import BaseAgentTool
from backend.config import settings

logger = logging.getLogger(__name__)

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class SearchInput(BaseModel):
    query: str = Field(description="搜索关键词或自然语言问题")
    max_results: int = Field(default=3, description="返回结果数量，1-10", ge=1, le=10)
    topic: Literal["general", "news", "finance"] = Field(
        default="general",
        description="general(通用) / news(新闻) / finance(金融)",
    )
    search_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="basic(快速,1 credit) / advanced(深度,2 credits)",
    )


class SearchTool(BaseAgentTool):
    name: str = "web_search"
    description: str = (
        "使用 Tavily 搜索引擎查询互联网。适用于：实时新闻、最新数据、"
        "当前价格/天气、最新技术资讯等需要实时联网的场景。"
    )
    args_schema: type[BaseModel] = SearchInput

    REQUEST_TIMEOUT: ClassVar[float] = 20.0

    # ── 工具方法 ────────────────────────────────────────────────────────────

    def _build_payload(
        self,
        query:        str,
        max_results:  int,
        topic:        str,
        search_depth: str,
    ) -> dict[str, Any]:
        return {
            "api_key":             settings.tools.tavily_api_key,
            "query":               query,
            "max_results":         max_results,
            "topic":               topic,
            "search_depth":        search_depth,
            "include_answer":      settings.tools.tavily_include_answer,
            "include_raw_content": settings.tools.tavily_include_raw_content,
        }

    def _build_client_kwargs(self) -> dict[str, Any]:
        """
        ✅ 核心修复：兼容 httpx 新旧版本的代理参数
        - httpx < 0.28: proxies={"https://": url}
        - httpx >= 0.28: proxy=url  或  mounts={"https://": httpx.AsyncHTTPTransport(proxy=url)}
        """
        proxy_url = settings.tools.proxy_url
        kwargs: dict[str, Any] = {"timeout": self.REQUEST_TIMEOUT}

        if not proxy_url:
            return kwargs

        # 解析 httpx 版本
        major, minor = _get_httpx_version()

        if (major, minor) >= (0, 28):
            # ✅ 新版 httpx >= 0.28：用 mounts 精细控制
            kwargs["mounts"] = {
                "http://":  httpx.AsyncHTTPTransport(proxy=proxy_url),
                "https://": httpx.AsyncHTTPTransport(proxy=proxy_url),
            }
        else:
            # ✅ 旧版 httpx < 0.28：用 proxies 字典
            kwargs["proxies"] = {
                "http://":  proxy_url,
                "https://": proxy_url,
            }

        return kwargs

    def _build_sync_client_kwargs(self) -> dict[str, Any]:
        """同步客户端版本"""
        proxy_url = settings.tools.proxy_url
        kwargs: dict[str, Any] = {"timeout": self.REQUEST_TIMEOUT}

        if not proxy_url:
            return kwargs

        major, minor = _get_httpx_version()

        if (major, minor) >= (0, 28):
            kwargs["mounts"] = {
                "http://":  httpx.HTTPTransport(proxy=proxy_url),
                "https://": httpx.HTTPTransport(proxy=proxy_url),
            }
        else:
            kwargs["proxies"] = {
                "http://":  proxy_url,
                "https://": proxy_url,
            }

        return kwargs

    def _format_results(self, raw: dict) -> str:
        parts: list[str] = []

        if raw.get("answer"):
            parts.append(f"💡 **AI 摘要**\n{raw['answer']}")

        results = raw.get("results", [])
        if results:
            parts.append("📚 **搜索结果**")
            for i, r in enumerate(results, 1):
                title   = r.get("title", "无标题")
                url     = r.get("url", "")
                content = r.get("content", "").strip()
                score   = r.get("score", 0.0)
                parts.append(
                    f"\n[{i}] {title}（相关度: {score:.2f}）\n"
                    f"🔗 {url}\n"
                    f"{content[:500]}{'...' if len(content) > 500 else ''}"
                )
        else:
            parts.append("⚠️ 未找到相关结果，建议换用其他关键词")

        if raw.get("follow_up_questions"):
            parts.append(
                "\n🤔 **相关问题**\n"
                + "\n".join(f"  • {q}" for q in raw["follow_up_questions"])
            )

        return "\n".join(parts)

    def _check_api_key(self) -> str | None:
        if not settings.tools.tavily_api_key:
            return "TAVILY_API_KEY 未配置，请在 .env 中设置。申请地址：https://tavily.com"
        return None

    # ── 异步主路径 ──────────────────────────────────────────────────────────

    async def _arun(
        self,
        query:        str,
        max_results:  int = 3,
        topic:        str = "general",
        search_depth: str = "basic",
    ) -> str:
        err = self._check_api_key()
        if err:
            return err

        payload     = self._build_payload(query, max_results, topic, search_depth)
        client_kwargs = self._build_client_kwargs()

        logger.info(
            f"[Tavily] 搜索: query={query!r} topic={topic} "
            f"depth={search_depth} "
            f"proxy={bool(settings.tools.proxy_url)}"
        )

        try:
            async with httpx.AsyncClient(**client_kwargs) as client:
                response = await client.post(TAVILY_SEARCH_URL, json=payload)

            if response.status_code == 401:
                return "Tavily API Key 无效，请检查 .env 中的 TAVILY_API_KEY"
            if response.status_code == 429:
                return "Tavily API 调用频率超限，请稍后重试"
            if response.status_code != 200:
                return f"Tavily API 返回异常: HTTP {response.status_code} — {response.text[:200]}"

            result = self._format_results(response.json())
            logger.info(f"[Tavily] 完成: 返回 {len(response.json().get('results', []))} 个结果")
            return result

        except httpx.ProxyError as e:
            logger.error(f"[Tavily] 代理错误: {e}")
            return f"代理连接失败：{e}\n请检查 HTTPS_PROXY 配置（当前：{settings.tools.proxy_url}）"
        except httpx.ConnectError as e:
            logger.error(f"[Tavily] 连接失败: {e}")
            return f"无法连接到 Tavily API，请检查网络或代理设置：{e}"
        except httpx.TimeoutException:
            logger.error("[Tavily] 请求超时")
            return f"Tavily 请求超时（>{self.REQUEST_TIMEOUT}s），请检查网络或代理"
        except Exception as e:
            logger.error(f"[Tavily] 未知错误: {type(e).__name__}: {e}", exc_info=True)
            return f"搜索失败：{type(e).__name__}: {e}"

    # ── 同步兜底 ────────────────────────────────────────────────────────────

    def _run(
        self,
        query:        str,
        max_results:  int = 3,
        topic:        str = "general",
        search_depth: str = "basic",
    ) -> str:
        err = self._check_api_key()
        if err:
            return err

        payload       = self._build_payload(query, max_results, topic, search_depth)
        client_kwargs = self._build_sync_client_kwargs()

        logger.info(f"[Tavily] 同步搜索: query={query!r}")

        try:
            with httpx.Client(**client_kwargs) as client:
                response = client.post(TAVILY_SEARCH_URL, json=payload)

            if response.status_code == 401:
                return "Tavily API Key 无效"
            if response.status_code == 429:
                return "Tavily API 调用频率超限"
            if response.status_code != 200:
                return f"Tavily API 返回异常: HTTP {response.status_code}"

            return self._format_results(response.json())

        except httpx.ProxyError as e:
            return f"代理连接失败：{e}"
        except httpx.ConnectError as e:
            return f"无法连接到 Tavily API：{e}"
        except httpx.TimeoutException:
            return f"请求超时（>{self.REQUEST_TIMEOUT}s）"
        except Exception as e:
            return f"搜索失败：{type(e).__name__}: {e}"


# ── 版本检测工具函数（模块级，只执行一次）──────────────────────────────────

def _get_httpx_version() -> tuple[int, int]:
    """解析 httpx 版本号，返回 (major, minor)"""
    try:
        parts = httpx.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 27)   # 解析失败，保守按旧版处理