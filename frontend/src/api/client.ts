// frontend/src/api/client.ts
import type {
  SessionSummary,
  ChatMessage,
  KnowledgeDoc,
  KnowledgeSearchResult,
  StreamEvent
} from '../types'

const API_BASE = import.meta.env.VITE_API_BASE_URL || ''

function buildUrl(path: string) {
  return `${API_BASE}${path}`
}

async function parseJson<T>(resp: Response): Promise<T> {
  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const data = await resp.json()
      detail = data?.detail?.error || data?.detail || data?.error || detail
    } catch {}
    throw new Error(detail)
  }
  return resp.json() as Promise<T>
}

export async function listSessions(userId: string): Promise<SessionSummary[]> {
  const resp = await fetch(buildUrl(`/api/v1/chat/sessions?user_id=${encodeURIComponent(userId)}`))
  const data = await parseJson<{ sessions: SessionSummary[] }>(resp)
  return data.sessions || []
}

export async function getHistory(sessionId: string): Promise<ChatMessage[]> {
  const resp = await fetch(buildUrl(`/api/v1/chat/history/${sessionId}`))
  const data = await parseJson<{ messages: Array<{ role: string; content: string; created_at?: string }> }>(resp)

  return (data.messages || []).map((m) => ({
    id: crypto.randomUUID(),
    role: normalizeRole(m.role),
    content: m.content,
    createdAt: m.created_at || new Date().toISOString()
  }))
}

export async function deleteSessionApi(sessionId: string): Promise<void> {
  const resp = await fetch(buildUrl(`/api/v1/chat/sessions/${sessionId}`), {
    method: 'DELETE'
  })
  if (!resp.ok) throw new Error(`删除会话失败: ${resp.status}`)
}

function normalizeRole(role: string): 'user' | 'assistant' | 'system' | 'tool' {
  const lower = role.toLowerCase()
  if (lower.includes('user') || lower.includes('human')) return 'user'
  if (lower.includes('assistant') || lower.includes('ai')) return 'assistant'
  if (lower.includes('tool')) return 'tool'
  return 'system'
}

export async function streamChat(
  body: {
    message: string
    session_id?: string
    user_id: string
    collection: string
  },
  handlers: {
    onOpen?: (sessionId?: string) => void
    onEvent: (event: StreamEvent) => void
    signal?: AbortSignal
  }
): Promise<void> {
  const resp = await fetch(buildUrl('/api/v1/chat'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(body),
    signal: handlers.signal
  })

  if (!resp.ok) {
    let detail = `HTTP ${resp.status}`
    try {
      const data = await resp.json()
      detail = data?.detail?.error || data?.detail || data?.error || detail
    } catch {}
    throw new Error(detail)
  }

  handlers.onOpen?.(resp.headers.get('X-Session-ID') || undefined)

  const reader = resp.body?.getReader()
  if (!reader) throw new Error('浏览器不支持流式读取')

  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) break

    buffer += decoder.decode(value, { stream: true })
    const frames = buffer.split('\n\n')
    buffer = frames.pop() || ''

    for (const frame of frames) {
      const lines = frame.split('\n')
      for (const line of lines) {
        if (!line.startsWith('data:')) continue
        const raw = line.slice(5).trim()
        if (!raw) continue
        const event = JSON.parse(raw) as StreamEvent
        handlers.onEvent(event)
      }
    }
  }
}

export async function uploadKnowledge(
  file: File,
  collection: string,
  uploadedBy: string
) {
  const form = new FormData()
  form.append('file', file)
  form.append('collection', collection)
  form.append('uploaded_by', uploadedBy)

  const resp = await fetch(buildUrl('/api/v1/knowledge/upload'), {
    method: 'POST',
    body: form
  })
  return parseJson<{
    doc_id: string
    filename: string
    status: string
    message: string
  }>(resp)
}

export async function uploadKnowledgeBatch(
  files: File[],
  collection: string,
  uploadedBy: string
) {
  const form = new FormData()
  files.forEach((file) => form.append('files', file))
  form.append('collection', collection)
  form.append('uploaded_by', uploadedBy)

  const resp = await fetch(buildUrl('/api/v1/knowledge/upload/batch'), {
    method: 'POST',
    body: form
  })
  return parseJson<{
    total: number
    processing: number
    failed: number
    results: Array<Record<string, unknown>>
  }>(resp)
}

export async function listKnowledgeDocs(
  collection: string,
  status: 'active' | 'processing' | 'failed' | 'deleted'
): Promise<KnowledgeDoc[]> {
  const resp = await fetch(
    buildUrl(`/api/v1/knowledge/docs?collection=${encodeURIComponent(collection)}&status=${status}`)
  )
  const data = await parseJson<{ documents: KnowledgeDoc[] }>(resp)
  return data.documents || []
}

export async function listAllKnowledgeDocs(collection: string): Promise<KnowledgeDoc[]> {
  const results = await Promise.allSettled([
    listKnowledgeDocs(collection, 'active'),
    listKnowledgeDocs(collection, 'processing'),
    listKnowledgeDocs(collection, 'failed'),
  ])

  const docs: KnowledgeDoc[] = []
  const statuses = ['active', 'processing', 'failed']

  results.forEach((result, idx) => {
    if (result.status === 'fulfilled') {
      docs.push(...result.value)
    } else {
      // 只打印警告，不阻断整体流程，不再弹窗骚扰用户
      console.warn(`[Knowledge] 拉取 ${statuses[idx]} 状态文档失败:`, result.reason)
    }
  })

  // processing 排前面，方便用户看到正在处理的
  return docs.sort((a, b) => {
    const order = { processing: 0, active: 1, failed: 2 }
    return (order[a.status as keyof typeof order] ?? 3) -
           (order[b.status as keyof typeof order] ?? 3)
  })
}

export async function deleteKnowledgeDoc(docId: string, collection: string) {
  const resp = await fetch(
    buildUrl(`/api/v1/knowledge/docs/${docId}?collection=${encodeURIComponent(collection)}`),
    { method: 'DELETE' }
  )
  return parseJson<{ message: string; doc_id: string }>(resp)
}

export async function searchKnowledge(
  query: string,
  collection: string,
  topK = 3
): Promise<KnowledgeSearchResult[]> {
  const resp = await fetch(buildUrl('/api/v1/knowledge/search'), {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({
      query,
      collection,
      top_k: topK
    })
  })
  const data = await parseJson<{ results: KnowledgeSearchResult[] }>(resp)
  return data.results || []
}