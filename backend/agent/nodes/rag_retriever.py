"""
RAG 检索节点
流程：查询改写 → 向量检索 → 相关性过滤 → 重排序 → 格式化上下文
"""

from __future__ import annotations

import logging
import time

from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

from backend.agent.state import AgentState, RetrievedDoc
from backend.config import settings
from backend.monitoring.metrics import rag_retrieval_duration, rag_retrieved_docs
from backend.rag.retriever import rag_retriever

logger = logging.getLogger(__name__)

# 查询改写 Prompt
QUERY_REWRITE_PROMPT = """
请将用户问题改写为更适合向量检索的查询语句。
改写要求：
1. 提取核心关键词
2. 去除无关的客套话
3. 保持语义完整
4. 如有多个子问题，拆分为多个查询（每行一个）

用户问题：{question}
改写后的查询（直接输出，不要解释）
"""

class RAGRetrieverNode:

    def __init__(self):
        cfg = settings.rag
        self.top_k = cfg.top_k
        self.score_threshold = cfg.score_threshold

        self.embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=settings.llm.api_key
        )
        self.vectorstore = Chroma(
            collection_name="knowledge_base",
            embedding_function=self.embeddings,
            persist_directory=cfg.chroma_dir,
        )
        self.llm = ChatOpenAI(
            model=settings.llm.model,
            temperature=0,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )

    async def _rewrite_query(self, question: str) -> list[str]:
        """
        查询改写：将用户自然语言转为更适合检索的关键词查询
        同时生成多个查询变体以提升召回率（Multi-Query）
        """
        try:
            response = await self.llm.ainvoke([
                HumanMessage(content=QUERY_REWRITE_PROMPT.format(question=question))
            ])
            queries = [
                q.strip()
                for q in response.content.strip().split('\n')
                if q.strip()
            ]
            # 加入原始问题保底
            if question not in queries:
                queries.insert(0, question)
            return queries[:3]      # 最多3个查询，控制检索成本
        except Exception as e:
            logger.warning(f"[RAG] 查询改写失败：{e}， 使用原始问题")
            return [question]

    async def run(self, state: AgentState) -> dict:
        user_input = state["user_input"]
        collection = state.get("collection", "default")     # 可从 state 传入
        start_time = time.time()

        logger.info(f"[RAGNode] session={state['session_id']} query={user_input[:50]}")

        # 1. 查询改写（Multi-Query）
        queries = await self._rewrite_query(user_input)
        logger.info(f"[RAGNode] rewritten queries: {queries}")

        # 2. 多查询并行检索 + 合并
        import asyncio
        results_list = await asyncio.gather(*[
            rag_retriever.retrieve(q, collection=collection)
            for q in queries
        ])

        # 3. 合并 & 去重
        seen = set()
        combined = []
        for results in results_list:
            for doc, score in results:
                key = doc.page_content[:80]
                if key not in seen:
                    seen.add(key)
                    combined.append((doc, score))

        # 4. 按分数排序，取Top-K
        combined.sort(key=lambda x: x[1], reverse=True)
        final = combined[: settings.rag.top_k]

        # 4.5 上报RAG指标
        elapsed = time.time() - start_time
        rag_retrieval_duration.observe(elapsed)
        rag_retrieved_docs.observe(len(final))

        # 5. 格式化输出
        retrieved_docs: list[RetrievedDoc] = [
            RetrievedDoc(
                content=doc.page_content,
                source=doc.metadata.get("source", "未知来源"),
                score=float(score),
                chunk_id=doc.metadata.get("chunk_id", ""),
                metadata=doc.metadata,
            )
            for doc, score in final
        ]

        # 6. 构建RAG上下文文本
        if final:
            parts = [
                f"【参考文档{i}】来源：{doc.metadata.get('source', '未知来源')}"
                f"（相关度：{score:.2f}） \n {doc.page_content}"
                for i, (doc, score) in enumerate(final, 1)
            ]
            rag_context = "\n\n---\n\n".join(parts)
        else:
            rag_context = "未找到相关结果，建议换用其他关键词"

        elapsed = (time.time() - start_time) * 1000
        logger.info(
            f"[RAGNode] 完成：docs={len(retrieved_docs)}"
            f"elapsed={elapsed:.2f}ms"
        )

        return {
            "retrieved_docs": retrieved_docs,
            "rag_context": rag_context,
        }


rag_retriever_node = RAGRetrieverNode()

async def run(state: AgentState) -> dict:
    return await rag_retriever_node.run(state)


