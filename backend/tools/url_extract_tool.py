# backend/tools/url_extract_tool.py
from __future__ import annotations
import logging
from typing import ClassVar, Literal, Any

import httpx
from pydantic import BaseModel, Field

from backend.tools.base_tool import BaseAgentTool
from backend.tools.search_tool import _get_httpx_version   # ✅ 复用版本检测
from backend.config import settings

logger = logging.getLogger(__name__)
TAVILY_EXTRACT_URL = "https://api.tavily.com/extract"


class URLExtractInput(BaseModel):
    urls: list[str] = Field(
        description="要抽取内容的 URL 列表（1-5个）",
        min_length=1,
        max_length=5,
    )
    extract_depth: Literal["basic", "advanced"] = Field(
        default="basic",
        description="basic(快速) / advanced(深度，支持复杂页面)",
    )


class URLExtractTool(BaseAgentTool):
    name: str = "url_extract"
    description: str = (
        "从给定 URL 抽取网页正文内容。当用户提供具体网址，"
        "或在搜索结果中想深入阅读某篇文章时使用。"
    )
    args_schema: type[BaseModel] = URLExtractInput

    REQUEST_TIMEOUT: ClassVar[float] = 30.0

    def _build_client_kwargs(self, sync: bool = False) -> dict[str, Any]:
        proxy_url = settings.tools.proxy_url
        kwargs: dict[str, Any] = {"timeout": self.REQUEST_TIMEOUT}

        if not proxy_url:
            return kwargs

        major, minor = _get_httpx_version()

        if (major, minor) >= (0, 28):
            transport_cls = httpx.HTTPTransport if sync else httpx.AsyncHTTPTransport
            kwargs["mounts"] = {
                "http://":  transport_cls(proxy=proxy_url),
                "https://": transport_cls(proxy=proxy_url),
            }
        else:
            kwargs["proxies"] = {"http://": proxy_url, "https://": proxy_url}

        return kwargs

    def _build_payload(self, urls: list[str], extract_depth: str) -> dict[str, Any]:
        return {
            "api_key":       settings.tools.tavily_api_key,
            "urls":          urls,
            "extract_depth": extract_depth,
        }

    def _format_results(self, raw: dict) -> str:
        parts: list[str] = []
        for i, r in enumerate(raw.get("results", []), 1):
            url     = r.get("url", "")
            content = r.get("raw_content", "").strip()
            parts.append(
                f"📄 [{i}] {url}\n"
                f"{content[:2000]}{'...(已截断)' if len(content) > 2000 else ''}"
            )
        failed = raw.get("failed_results", [])
        if failed:
            parts.append(
                "⚠️ 以下 URL 抽取失败：\n"
                + "\n".join(
                    f"  • {f.get('url')}: {f.get('error', '未知错误')}"
                    for f in failed
                )
            )
        return "\n\n---\n\n".join(parts) if parts else "未能抽取任何内容"

    async def _arun(
        self, urls: list[str], extract_depth: str = "basic"
    ) -> str:
        logger.info(f"[Tavily-Extract] urls={urls}")
        try:
            async with httpx.AsyncClient(
                **self._build_client_kwargs(sync=False)
            ) as client:
                response = await client.post(
                    TAVILY_EXTRACT_URL,
                    json=self._build_payload(urls, extract_depth),
                )
            if response.status_code != 200:
                return f"抽取失败，HTTP {response.status_code}: {response.text[:200]}"
            return self._format_results(response.json())
        except httpx.ProxyError as e:
            return f"代理连接失败：{e}"
        except httpx.ConnectError as e:
            return f"无法连接 Tavily：{e}"
        except httpx.TimeoutException:
            return f"请求超时（>{self.REQUEST_TIMEOUT}s）"
        except Exception as e:
            logger.error(f"[Tavily-Extract] 失败: {e}", exc_info=True)
            return f"URL 抽取失败：{type(e).__name__}: {e}"

    def _run(self, urls: list[str], extract_depth: str = "basic") -> str:
        try:
            with httpx.Client(**self._build_client_kwargs(sync=True)) as client:
                response = client.post(
                    TAVILY_EXTRACT_URL,
                    json=self._build_payload(urls, extract_depth),
                )
            if response.status_code != 200:
                return f"抽取失败，HTTP {response.status_code}"
            return self._format_results(response.json())
        except Exception as e:
            return f"URL 抽取失败：{type(e).__name__}: {e}"