// frontend/src/components/KnowledgeManager.tsx
import { useEffect, useMemo, useRef, useState } from 'react'
import {
  deleteKnowledgeDoc,
  listAllKnowledgeDocs,
  searchKnowledge,
  uploadKnowledge,
  uploadKnowledgeBatch
} from '../api/client'
import type { KnowledgeDoc, KnowledgeSearchResult } from '../types'

interface Props {
  userId: string
  collection: string
  onCollectionChange: (collection: string) => void
}

export default function KnowledgeManager({
  userId,
  collection,
  onCollectionChange
}: Props) {
  const [docs, setDocs] = useState<KnowledgeDoc[]>([])
  const [loading, setLoading] = useState(false)
  const [searching, setSearching] = useState(false)
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<KnowledgeSearchResult[]>([])
  const [uploading, setUploading] = useState(false)
  const fetchingRef = useRef(false)
  const docsRef = useRef<KnowledgeDoc[]>([])

  async function loadDocs() {
    if (fetchingRef.current) return
    fetchingRef.current = true
    setLoading(true)
    try {
      const all = await listAllKnowledgeDocs(collection)
      setDocs(all)
    } catch (err) {
      console.error('[Knowledge] 加载文档列表失败:', err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    void loadDocs()
  }, [collection])

  useEffect(() => {
    const timer = setInterval(() => {
      const hasProcessing = docsRef.current.some((d) => d.status === 'processing')
      if (hasProcessing) {
        void loadDocs()
      }
    }, 3000)

    return () => clearInterval(timer)
  }, [collection])

  async function handleUpload(files: FileList | null) {
    if (!files || files.length === 0) return
    setUploading(true)
    try {
      const arr = Array.from(files)
      if (arr.length === 1) {
        await uploadKnowledge(arr[0], collection, userId)
      } else {
        await uploadKnowledgeBatch(arr, collection, userId)
      }
      await loadDocs()
    } catch (err) {
      alert((err as Error).message)
    } finally {
      setUploading(false)
    }
  }

  async function handleDelete(docId: string) {
    if (!confirm('确认删除该文档？')) return
    try {
      await deleteKnowledgeDoc(docId, collection)
      await loadDocs()
    } catch (err) {
      alert((err as Error).message)
    }
  }

  async function handleSearch() {
    if (!query.trim()) return
    setSearching(true)
    try {
      const res = await searchKnowledge(query.trim(), collection, 3)
      setResults(res)
    } catch (err) {
      alert((err as Error).message)
    } finally {
      setSearching(false)
    }
  }

  const grouped = useMemo(() => {
    return {
      processing: docs.filter((d) => d.status === 'processing'),
      active: docs.filter((d) => d.status === 'active'),
      failed: docs.filter((d) => d.status === 'failed')
    }
  }, [docs])

  return (
    <div className="knowledge-page">
      <div className="page-header">
        <div>
          <h2>知识库管理</h2>
          <p>上传文档、查看状态、测试检索效果。</p>
        </div>

        <div className="collection-switcher">
          <label>Collection</label>
          <input
            value={collection}
            onChange={(e) => onCollectionChange(e.target.value)}
            placeholder="default"
          />
        </div>
      </div>

      <div className="knowledge-grid">
        <section className="card">
          <h3>上传文档</h3>
          <p className="muted">支持 PDF / MD / TXT / DOCX，单文件最大 20MB。</p>

          <label className="upload-box">
            <input
              type="file"
              multiple
              onChange={(e) => void handleUpload(e.target.files)}
              disabled={uploading}
            />
            <span>{uploading ? '上传中...' : '点击选择文件或批量上传'}</span>
          </label>
        </section>

        <section className="card">
          <h3>手动检索测试</h3>
          <div className="search-row">
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="输入一个问题，例如：LangGraph 是什么？"
            />
            <button className="btn btn-primary" onClick={() => void handleSearch()} disabled={searching}>
              {searching ? '检索中...' : '检索'}
            </button>
          </div>

          <div className="search-results">
            {results.length === 0 ? (
              <div className="muted">暂无检索结果</div>
            ) : (
              results.map((item, idx) => (
                <div key={`${item.chunk_id}-${idx}`} className="search-result-card">
                  <div className="search-result-head">
                    <strong>{item.source}</strong>
                    <span>score: {item.score.toFixed(4)}</span>
                  </div>
                  <pre>{item.content}</pre>
                </div>
              ))
            )}
          </div>
        </section>
      </div>

      <section className="card">
        <div className="card-header">
          <h3>文档列表</h3>
          <button className="btn btn-secondary" onClick={() => void loadDocs()} disabled={loading}>
            {loading ? '刷新中...' : '刷新'}
          </button>
        </div>

        <div className="doc-group">
          <h4>处理中</h4>
          {grouped.processing.length === 0 ? (
            <div className="muted">无</div>
          ) : (
            grouped.processing.map((doc) => (
              <DocRow key={doc.doc_id} doc={doc} onDelete={handleDelete} />
            ))
          )}
        </div>

        <div className="doc-group">
          <h4>可用文档</h4>
          {grouped.active.length === 0 ? (
            <div className="muted">无</div>
          ) : (
            grouped.active.map((doc) => (
              <DocRow key={doc.doc_id} doc={doc} onDelete={handleDelete} />
            ))
          )}
        </div>

        <div className="doc-group">
          <h4>失败文档</h4>
          {grouped.failed.length === 0 ? (
            <div className="muted">无</div>
          ) : (
            grouped.failed.map((doc) => (
              <DocRow key={doc.doc_id} doc={doc} onDelete={handleDelete} />
            ))
          )}
        </div>
      </section>
    </div>
  )
}

function DocRow({
  doc,
  onDelete
}: {
  doc: KnowledgeDoc
  onDelete: (docId: string) => void
}) {
  return (
    <div className="doc-row">
      <div className="doc-info">
        <div className="doc-name">{doc.filename}</div>
        <div className="doc-meta-line">
          <span>{doc.file_type}</span>
          <span>{(doc.file_size / 1024).toFixed(1)} KB</span>
          <span>{doc.chunk_count} chunks</span>
        </div>
      </div>

      <div className="doc-actions">
        <span className={`status-badge ${doc.status}`}>{doc.status}</span>
        <button className="btn btn-danger-outline" onClick={() => onDelete(doc.doc_id)}>
          删除
        </button>
      </div>
    </div>
  )
}