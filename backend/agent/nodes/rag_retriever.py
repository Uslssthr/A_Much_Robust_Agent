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

    def _deduplicate_docs(self, docs: list[Document]) -> list[Document]:
        """去重：相同 chunk_id 或高度相似内容只保留一个"""
        seen_ids = set()
        seen_content = set()
        result = []
        for doc in docs:
            chunk_id = doc.metadata.get("chunk_id", "")
            # 用前100字符粗略去重
            content_key = doc.page_content[:100].strip()
            if chunk_id in seen_ids and content_key in seen_content:
                continue
            seen_ids.add(chunk_id)
            seen_content.add(content_key)
            result.append(doc)

        return result

    def _filter_by_score(
            self, docs_with_scores: list[tuple[Document, float]]
    ) -> list[tuple[Document, float]]:
        """按相关性分数过滤低质量文档"""
        return [
            (doc, score)
            for doc, score in docs_with_scores
            if score >= self.score_threshold
        ]

    async def _rerank(
            self,
            query: str,
            docs_with_scores: list[tuple[Document, float]],
    ) -> list[tuple[Document, float]]:
        """
        LLM Reranker：对候选文档重新打分排序
        轻量版：让 LLM 对每个文档打相关性分 0-10
        """
        if len(docs_with_scores) <= 1:
            return docs_with_scores

        try:
            doc_texts = "\n---\n".join([
                f"[文档{i+1}] 来源：{doc.metadata.get('source', '未知来源')}\n"
                f"内容：{doc.page_content[:300]}"
                for i, (doc, _) in enumerate(docs_with_scores)
            ])

            rerank_prompt = f"""
            请对以下文档与问题的相关性打分（0-10整数）。
            问题：{query}

            {doc_texts}
            
            按顺序输出每个文档的分数，格式：
            文档1: <分数>
            文档2: <分数>
            """

            response = await self.llm.ainvoke([
                HumanMessage(content=rerank_prompt)
            ])

            # 解析分数
            import re
            scores_raw = re.findall(r'文档\d+:\s*(\d+)', response.content)
            if len(scores_raw) == len(docs_with_scores):
                reranked_scores = [float(s) / 10.0 for s in scores_raw]
                # 重组并排序
                reranked = [
                    (doc, reranked_scores[i])
                    for i, (doc, _) in enumerate(docs_with_scores)
                ]
                return sorted(reranked, key=lambda x:x[1], reverse=True)

        except Exception as e:
            logger.warning(f"[RAG] Rerank失败：{e}， 使用原始排序")

        return docs_with_scores

    def _format_context(self, docs_with_scores: list[tuple[Document, float]]) -> str:
        """将检索文档格式化为 Prompt 可用的上下文文本"""
        if not docs_with_scores:
            return "暂无相关文档"

        parts = []
        for i, (doc, score) in enumerate(docs_with_scores, 1):
            source = doc.metadata.get("source", "未知来源")
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            parts.append(
                f"【参考文档 {i}】\n"
                f"来源：{source}（相关度：{score:.2f}）\n"
                f"内容：{doc.page_content}\n"
            )

        return "\n".join(parts)

    async def run(self, state: AgentState) -> dict:
        user_input = state["user_input"]
        start_time = time.time()

        logger.info(f"[RAG] session={state['session_id']} query={user_input[:50]}")

        # 1. 查询改写（Multi-Query）
        queries = await self._rewrite_query(user_input)
        logger.info(f"[RAG] rewritten queries: {queries}")

        # 2. 多查询并行检索 + 合并
        all_docs_with_scores: list[tuple[Document, float]] = []
        for query in queries:
            try:
                results = self.vectorstore.similarity_search_with_relevance_scores(
                    query,
                    k=self.top_k,
                )
                all_docs_with_scores.extend(results)

            except Exception as e:
                logger.warning(f"[RAG] 检索失败 query={query}: {e}")

        # 3. 去重
        docs = self._deduplicate_docs([d for d, _ in all_docs_with_scores])
        # 重组 (doc, score) 保持去重后的一致性
        seen = {d.page_content[:100] for d in docs}
        deduped = [
            (d, s)
            for d, s in all_docs_with_scores
            if d.page_content[:100] not in seen
        ]
        # 再次去重（可能有重复 key）
        seen_check = set()
        unique_docs = []
        for d, s in deduped:
            key = d.page_content[:100]
            if key not in seen_check:
                seen_check.add(key)
                unique_docs.append((d, s))

        # 4. 相关性过滤
        filtered = self._filter_by_score(unique_docs)

        # 5. Rerank
        reranked = await self._rerank(user_input, filtered)
        # 取 Top-K
        final_docs = reranked[:self.top_k]

        # 6. 格式化
        rag_context = self._format_context(final_docs)

        # 7. 组装 RetrievedDoc 列表
        retrieved_docs: list[RetrievedDoc] = [
            RetrievedDoc(
                content=doc.page_content,
                source=doc.metadata.get("source", "未知来源"),
                score=float(score),
                chunk_id=doc.metadata.get("chunk_id", f"chunk_{i}"),
                metadata=doc.metadata,
            )
            for i, (doc, score) in enumerate(final_docs)
        ]

        elapsed_ms = (time.time() - start_time) *  (1000.0)
        logger.info(
            f"[RAG] 完成检索: retrieved={len(retrieved_docs)} "
            f"elapsed={elapsed_ms:.1f}ms"
        )

        return {
            "retrieved_docs": retrieved_docs,
            "rag_context": rag_context,
        }


rag_retriever_node = RAGRetrieverNode()

async def run(state: AgentState) -> dict:
    return await rag_retriever_node.run(state)


