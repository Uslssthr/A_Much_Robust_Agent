"""
RAG 文档导入管线
流程：文件保存 → 格式检测 → 加载 → 分块 → 向量化 → 存储
支持：PDF / Markdown / TXT / DOCX
"""
from __future__ import annotations

import asyncio
import logging
import os
import uuid
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
    Docx2txtLoader,
)
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from multipart import file_path

from backend.config import settings
from backend.db.sqlite_manager import db

logger = logging.getLogger(__name__)

# 支持的文件类型映射
SUPPORTED_TYPES: dict[str, str] = {
    ".pdf": "pdf",
    ".md": "md",
    ".txt": "txt",
    ".docx": "docx",
}

UPLOAD_DIR = os.getenv("UPLOAD_DIR", "./data/uploads")


class IngestionPipeline:
    """文档导入管线"""

    def __init__(self):
        cfg = settings.rag
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
            separators=["\n\n", "\n", "。", "！", "？", "；", " ", ""],
            length_function=len,
        )
        self.embeddings = OpenAIEmbeddings(
            model=cfg.embedding_model,
            api_key=settings.llm.api_key,
            base_url=settings.llm.base_url,
        )
        self.chroma_dir = cfg.chroma_dir
        os.makedirs(self.chroma_dir, exist_ok=True)
        os.makedirs(UPLOAD_DIR, exist_ok=True)

    def _get_vectorstore(self, collection: str = "default") -> Chroma:
        """获取（或创建）向量库"""
        return Chroma(
            collection_name=f"kb_{collection}",
            embedding_function=self.embeddings,
            persist_directory=self.chroma_dir,
        )

    def _detect_file_type(self, filename: str) -> str | None:
        """检测文件类型"""
        suffix = Path(filename).suffix.lower()
        return SUPPORTED_TYPES.get(suffix)

    def _load_documents(self, file_path: str, file_type: str) -> list[Document]:
        """根据文件类型选择加载器"""
        loaders = {
            "pdf": lambda: PyPDFLoader(file_path),
            "md": lambda: UnstructuredMarkdownLoader(file_path),
            "txt": lambda: TextLoader(file_path, encoding="utf-8"),
            "docx": lambda: Docx2txtLoader(file_path),
        }
        loader_fn = loaders.get(file_type)
        if not loader_fn:
            raise ValueError(f"不支持的文件类型：{file_type}")
        return loader_fn().load()

    def _enrich_metadata(
        self,
        chunks: list[Document],
        doc_id: str,
        filename: str,
        file_type: str,
        collection: str = "default",
    ) -> list[Document]:
        """为每个 chunk 补充元数据"""
        enriched = []
        for i, chunk in enumerate(chunks):
            chunk.metadata.update({
                "chunk_id": f"{doc_id}_chunk_{i:04d}",
                "doc_id": doc_id,
                "source": filename,
                "file_type": file_type,
                "collection": collection,
                "chunk_index": i,
                "total_chunks": len(chunks),
            })
            enriched.append(chunk)
        return enriched

    async def ingest_file(
        self,
        file_content: bytes,
        filename: str,
        collection: str = "default",
        uploaded_by: str = "anonymous",
    ) -> dict:
        """
        主入口：导入单个文件
        返回: {doc_id, filename, chunk_count, status}
        """
        # 1. 检测文件类型
        file_type = self._detect_file_type(filename)
        if not file_type:
            return {
                "status": "failed",
                "error": f"不支持的文件类型, 支持类型：{list(SUPPORTED_TYPES.keys())}",
                "doc_id": None,
            }

        doc_id = str(uuid.uuid4())
        file_size = len(file_content)

        # 2. 保存文件到磁盘
        safe_name = f"{doc_id}_{filename}"
        file_path = os.path.join(UPLOAD_DIR, safe_name)

        with open(file_path, "wb") as f:
            f.write(file_content)

        logger.info(
            f"[Ingestion] 开始处理：{filename}"
            f"type={file_type} size={file_size}B doc_id={doc_id}"
        )

        # 3. 记录到 SQLite（状态：processing）
        db.create_kb_doc(
            doc_id=doc_id,
            filename=filename,
            file_type=file_type,
            file_size=file_size,
            collection=collection,
            uploaded_by=uploaded_by,
        )

        # 4. 异步执行实际处理（避免阻塞 API）
        asyncio.create_task(
            self._process_file(
                file_path=file_path,
                file_type=file_type,
                doc_id=doc_id,
                filename=filename,
                collection=collection,
            )
        )

        return {
            "doc_id": doc_id,
            "filename": filename,
            "status": "processing",
            "message": "文件已接收，正在后台处理中...",
        }

    async def _process_file(
        self,
        file_path: str,
        file_type: str,
        doc_id: str,
        filename: str,
        collection: str,
    ):
        """后台处理任务：加载 → 分块 → 向量化 → 入库"""
        try:
            # 加载（在线程池中执行，避免阻塞事件循环）
            loop = asyncio.get_running_loop()
            documents = await loop.run_in_executor(
                None,
                self._load_documents,
                file_path,
                file_type,
            )
            logger.info(f"[Ingestion] 加载完成：{filename} ，{len(documents)} 页/段")

            # 分块
            chunks = self.splitter.split_documents(documents)
            logger.info(f"[Ingestion] 分块完成：{filename} ，{len(chunks)} 个 chunk")

            # 补充元数据
            chunks = self._enrich_metadata(chunks, doc_id, filename, file_type, collection)

            # 向量化并存入 Chroma（批量处理，每批50个）
            vectorstore = self._get_vectorstore(collection)
            batch_size = 50
            total_chunks = len(chunks)

            for i in range(0, total_chunks, batch_size):
                batch = chunks[i: i + batch_size]
                await loop.run_in_executor(
                    None,
                    vectorstore.add_documents,
                    batch,
                )
                logger.info(
                    f"[Ingestion] 向量化进度："
                    f"{min(i + batch_size, total_chunks)}/{total_chunks}"
                )

            # 更新 SQLite 状态
            db.update_kb_doc_status(
                doc_id=doc_id,
                status="active",
                chunk_count=total_chunks,
            )
            logger.info(f"[Ingestion] 完成： doc_id={doc_id} ，{total_chunks} 个 chunk")

        except Exception as e:
            logger.error(f"[Ingestion] 处理文件时出错：doc_id={doc_id} error={e}", exc_info=True)
            db.update_kb_doc_status(
                doc_id=doc_id,
                status="failed",
                error_msg=str(e),
            )

    async def delete_document(self, doc_id: str, collection: str = "default") -> bool:
        """从向量库和 SQLite 中删除文档"""
        try:
            vectorstore = self._get_vectorstore(collection)
            # Chroma 按 metadata 过滤删除
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                None,
                lambda: vectorstore.delete(
                    where={"doc_id": doc_id}
                )
            )
        except Exception as e:
            logger.error(f"[Ingestion] 向量库删除文档时出错：doc_id={doc_id} error={e}", exc_info=True)

        # SQLite 软删除
        return db.soft_delete_kb_doc(doc_id)


# 初始化管道
ingestion_pipline = IngestionPipeline()