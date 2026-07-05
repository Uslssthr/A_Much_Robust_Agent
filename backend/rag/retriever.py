"""
RAG 检索器
独立封装，供 rag_retriever node 和 knowledge_tool 共同调用
"""
from __future__ import annotations

import asyncio
import logging

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from backend.config import settings

logger = logging.getLogger(__name__)


class RAGRetriever:

    def __init__(self):
        cfg = settings.rag
        self.top_k = cfg.top_k
        self.score_threshold = cfg.score_threshold
        self.chroma_dir = cfg.chroma_dir
        self.embeddings = embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            base_url=cfg.base_url,
            api_key=""
        )

    def _get_vectorstore(self, collection: str = "default") -> Chroma:
        return Chroma(
            collection_name=f"kb_{collection}",
            embedding_function=self.embeddings,
            persist_directory=self.chroma_dir,
        )

    async def retrieve(
        self,
        query: str,
        collection: str = "default",
        top_k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """
        异步检索，返回 [(doc, score), ...]
        score 越高越相关（Chroma cosine similarity：0~1）
        """
        k = top_k or self.top_k
        vectorstore = self._get_vectorstore(collection)

        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: vectorstore.similarity_search_with_relevance_scores(
                    query, k=k*2,   # 多取一倍，后续过滤低分结果
                )
            )
            # 过滤低相关文档
            filtered = [
                (doc, score)
                for doc, score in results
                if score >= self.score_threshold
            ]

            # 取前 k 个
            return filtered[:k]

        except Exception as e:
            logger.error(f"[Retriever] 检索失败: {e}")
            return []

    async def retrieve_as_text(
        self,
        query: str,
        collection: str = "default",
    ) -> str:
        """直接返回格式化后的文本（供 Prompt 使用）"""
        results = await self.retrieve(query, collection)
        if not results:
            return "知识库中未找到相关内容"

        parts = []
        for i, (doc, score) in enumerate(results, 1):
            source = doc.metadata.get("source", "未知来源")
            parts.append(
                f"【参考文档 {i}】来源：{source} (相关度：{score:.4f}) \n"
                f"{doc.page_content}\n"
            )
        return "\n\n---\n\n".join(parts)


# 全局实例
rag_retriever = RAGRetriever()
