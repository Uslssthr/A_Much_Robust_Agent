"""
知识库管理 API
POST   /api/v1/knowledge/upload        → 上传文档
GET    /api/v1/knowledge/docs          → 文档列表
DELETE /api/v1/knowledge/docs/{doc_id} → 删除文档
POST   /api/v1/knowledge/search        → 手动检索（测试用）
GET    /api/v1/knowledge/status/{doc_id} → 查询处理状态
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from pydantic import BaseModel

from backend.db.sqlite_manager import db
from backend.rag.ingestion import ingestion_pipline
from backend.rag.retriever import rag_retriever

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1/knowledge", tags=["Knowledge Base"])

# 文件大小限制：40MB
MAX_FILE_SIZE = 40 * 1024 * 1024

# 上传文档
@router.post("/upload")
async def upload_document(
    file: UploadFile = File(..., description="支持PDF/MD/TXT/DOCX"),
    collection: str = Form(default="default", description="知识库集合"),
    uploaded_by: str = Form(default="anonymous"),
):
    """
    上传文档到知识库
    - 异步处理：接口立即返回，后台向量化
    - 通过 /status/{doc_id} 查询处理进度
    """
    # 读取文件内容
    content = await file.read()

    # 文件大小检查
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大（{len(content)/1024/1024:.2f}MB），最大支持{MAX_FILE_SIZE/1024/1024:.2f}MB",
        )

    if len(collection) == 0:
        raise HTTPException(
            status_code=400,
            detail="文件不能为空",
        )

    # 导入管线处理
    result = await ingestion_pipline.ingest_file(
        file_content=content,
        filename=file.filename or "unknown",
        collection=collection,
        uploaded_by=uploaded_by,
    )

    if result["status"] == "failed":
        raise HTTPException(status_code=400, detail=result.get("error", "未知错误"))

    return result


# 批量上传文档
@router.post("/upload/batch")
async def upload_batch(
    files: list[UploadFile] = File(..., description="支持PDF/MD/TXT/DOCX"),
    collection: str = Form(default="default", description="知识库集合"),
    uploaded_by: str = Form(default="anonymous"),
):
    """批量上传（最多10个文件）"""
    if len(files) > 10:
        raise HTTPException(
            status_code=400,
            detail="最多上传10个文件",
        )

    result = []
    for f in files:
        content = await f.read()
        r = await ingestion_pipline.ingest_file(
            file_content=content,
            filename=f.filename or "unknown",
            collection=collection,
            uploaded_by=uploaded_by,
        )
        result.append(r)

    return {
        "total": len(result),
        "processing": sum(1 for r in result if r["status"] == "processing"),
        "failed": sum(1 for r in result if r["status"] == "failed"),
        "results": result,
    }


# 查询处理状态
@router.get("/status/{doc_id}")
async def get_doc_status(doc_id: str):
    """查询文档处理状态"""
    docs = db.list_kb_docs(status="active")
    all_docs = (
        db.list_kb_docs(status="active")
        + db.list_kb_docs(status="failed")
        + db.list_kb_docs(status="processing")
    )
    target = next((d for d in all_docs if d["doc_id"] == doc_id), None)
    if not target:
        raise HTTPException(status_code=404, detail="文档不存在")
    return target


# 文档列表
@router.get("/docs")
async def list_docs(
    collection: str = Query(default="default"),
    status: str = Query(default="active"),
):
    """获取知识库文档列表"""
    if status not in ["active", "failed", "processing", "deleted"]:
        raise HTTPException(status_code=400, detail="无效的状态值")

    docs = db.list_kb_docs(collection=collection, status=status)
    return {
        "collection": collection,
        "status": status,
        "count": len(docs),
        "documents": docs,
    }


# 删除文档
@router.delete("/docs/{doc_id}")
async def delete_document(
    doc_id: str,
    collection: str = Query(default="default"),
):
    """删除知识库文档（软删除 + 向量库清理）"""
    success = await ingestion_pipline.delete_document(doc_id, collection)
    if not success:
        raise HTTPException(
            status_code=400,
            detail=f"文档{doc_id}不存在或已删除",
        )
    return {"message": f"文档{doc_id}已删除", "doc_id": doc_id}

# 手动检索（测试用）
class SearchRequest(BaseModel):
    query: str = ""
    collection: str = "default"
    top_k: int = 5


@router.post("/search")
async def search_knowledge(req: SearchRequest):
    """手动检索知识库（用于测试检索效果）"""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="查询内容不能为空")

    results = await rag_retriever.retrieve(
        query=req.query,
        collection=req.collection,
        top_k=req.top_k,
    )

    return {
        "query": req.query,
        "collection": req.collection,
        "count": len(results),
        "results": [
            {
                "content": doc.page_content,
                "source": doc.metadata.get("source", "未知"),
                "score": round(score, 4),
                "chunk_id": doc.metadata.get("chunk_id", ""),
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]
    }
