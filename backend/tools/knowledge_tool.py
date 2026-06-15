"""
知识库查询工具
让 Agent 在 ReAct 推理中主动查询知识库
（区别于 RAG 路由中的自动检索，这里是工具化的按需检索）
"""
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

from backend.config import settings
from backend.tools.base_tool import BaseAgentTool


class KnowledgeQueryInput(BaseModel):
    query: str = Field(description="在知识库中搜索的问题或关键词")
    top_k: int = Field(default=3, description="返回结果数量", ge=1, le=5)


class KnowledgeQueryTool(BaseAgentTool):
    name: str = "knowledge_base_search"
    description: str = (
        "查询内部知识库，获取专业文档、产品手册、规章制度等内部信息。"
        "当问题涉及公司内部知识、专业文档时使用此工具。"
    )
    args_schema: type[BaseModel] = KnowledgeQueryInput

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _get_vectorstore(self) -> Chroma:
        """懒加载向量库（避免循环导入）"""
        embeddings = OpenAIEmbeddings(
            model=settings.rag.embedding_model,
            api_key=settings.llm.api_key,
        )
        return Chroma(
            collection_name="knowledge_base",
            embedding_function=embeddings,
            persist_directory=settings.rag.chroma_dir,
        )

    def _run(self, query: str, top_k: int = 3) -> str:
        try:
            vs = self._get_vectorstore()
            results = vs.similarity_search_with_relevance_scores(query, k=top_k)

            if not results:
                return f"知识库中未找到与 '{query}' 相关的信息。"

            parts = []
            for i, (doc, score) in enumerate(results, 1):
                source = doc.metadata.get("source", "未知")
                parts.append(
                    f"[结果{i}] 相关度:{score:.2f} 来源:{source}\n"
                    f"{doc.page_content}"
                )

            return "\n\n---\n\n".join(parts)

        except Exception as e:
            return f"知识库查询失败：{str(e)}"


