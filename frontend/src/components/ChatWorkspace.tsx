// frontend/src/components/ChatWorkspace.tsx
import { useEffect, useMemo, useRef } from 'react'
import { getHistory, listSessions, streamChat } from '../api/client'
import { useChatStore } from '../store/chatStore'
import { getOrCreateUserId } from '../utils/user'
import type { ChatMessage, ToolCallView } from '../types'
import MessageList from './MessageList'
import Composer from './Composer'
import EmptyState from './EmptyState'

export default function ChatWorkspace() {
  const userId = getOrCreateUserId()
  const abortRef = useRef<AbortController | null>(null)

  const {
    sessions,
    currentSessionId,
    messagesBySession,
    sending,
    activeCollection,
    setSessions,
    upsertSession,
    setCurrentSession,
    setMessages,
    appendMessage,
    updateMessage,
    setSending,
    setError,
    startLocalSession
  } = useChatStore()

  const messages = useMemo(
    () => (currentSessionId ? messagesBySession[currentSessionId] || [] : []),
    [currentSessionId, messagesBySession]
  )

  useEffect(() => {
    void refreshSessions()
  }, [])

  async function refreshSessions() {
    try {
      const data = await listSessions(userId)
      setSessions(data)
    } catch (err) {
      console.error(err)
    }
  }

  async function loadSessionHistory(sessionId: string) {
    setCurrentSession(sessionId)
    if ((messagesBySession[sessionId] || []).length > 0) return
    try {
      const history = await getHistory(sessionId)
      setMessages(sessionId, history)
    } catch (err) {
      console.error(err)
      setError((err as Error).message)
    }
  }

  async function handleSend(text: string) {
    setError(null)

    let sessionId = currentSessionId
    if (!sessionId) {
      sessionId = startLocalSession(text.slice(0, 20))
    }

    const userMessage: ChatMessage = {
      id: crypto.randomUUID(),
      role: 'user',
      content: text,
      createdAt: new Date().toISOString()
    }

    const assistantMessageId = crypto.randomUUID()
    const assistantDraft: ChatMessage = {
      id: assistantMessageId,
      role: 'assistant',
      content: '',
      createdAt: new Date().toISOString(),
      pending: true,
      toolCalls: []
    }

    appendMessage(sessionId, userMessage)
    appendMessage(sessionId, assistantDraft)
    setSending(true)

    abortRef.current = new AbortController()

    let runtimeSessionId = sessionId

    try {
      await streamChat(
        {
          message: text,
          session_id: sessionId.startsWith('local-') ? undefined : sessionId,
          user_id: userId,
          collection: activeCollection
        },
        {
          signal: abortRef.current.signal,
          onOpen: (realSessionId) => {
            if (!realSessionId) return

            if (runtimeSessionId.startsWith('local-')) {
              upsertSession({
                session_id: runtimeSessionId,
                title: text.slice(0, 20),
                updated_at: new Date().toISOString(),
                isLocal: true
              })

              useChatStore.getState().renameSession(runtimeSessionId, realSessionId)
              runtimeSessionId = realSessionId
            } else {
              runtimeSessionId = realSessionId
            }

            setCurrentSession(runtimeSessionId)
          },
          onEvent: (event) => {
            if (event.type === 'thinking') {
                updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                  ...msg,
                  thinkingStage: event.stage,
                }))
            }

            if (event.type === 'token') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                ...msg,
                content: msg.content + event.content
              }))
            }

            if (event.type === 'tool_start') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => {
                const nextCall: ToolCallView = {
                  id: crypto.randomUUID(),
                  tool: event.tool,
                  args: event.args || {},
                  status: 'running'
                }
                return {
                  ...msg,
                  toolCalls: [...(msg.toolCalls || []), nextCall]
                }
              })
            }

            if (event.type === 'tool_end') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => {
                const toolCalls = [...(msg.toolCalls || [])]
                const idx = [...toolCalls]
                  .reverse()
                  .findIndex((c) => c.tool === event.tool && c.status === 'running')

                if (idx >= 0) {
                  const realIndex = toolCalls.length - 1 - idx
                  toolCalls[realIndex] = {
                    ...toolCalls[realIndex],
                    output: event.output,
                    elapsedMs: event.elapsed_ms,
                    status: 'success'
                  }
                }

                return { ...msg, toolCalls }
              })
            }

            if (event.type === 'final') {
                updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                  ...msg,
                  content: event.content,
                  thinkingStage: undefined,
                }))
            }

            if (event.type === 'meta') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                ...msg,
                meta: {
                  route: event.route,
                  iteration: event.iteration,
                  compressed: event.compressed
                }
              }))
            }

            if (event.type === 'error') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                ...msg,
                pending: false,
                error: true,
                content: msg.content
                  ? `${msg.content}\n\n[错误] ${event.message}`
                  : `[错误] ${event.message}`
              }))
            }

            if (event.type === 'done') {
              updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
                ...msg,
                pending: false
              }))

              upsertSession({
                session_id: event.session_id,
                title: text.slice(0, 20),
                updated_at: new Date().toISOString()
              })

              void refreshSessions()
            }
          }
        }
      )
    } catch (err) {
      const message =
        err instanceof DOMException && err.name === 'AbortError'
          ? '用户已停止生成'
          : (err as Error).message

      updateMessage(runtimeSessionId, assistantMessageId, (msg) => ({
        ...msg,
        pending: false,
        error: true,
        content: msg.content
          ? `${msg.content}\n\n[错误] ${message}`
          : `[错误] ${message}`
      }))
    } finally {
      setSending(false)
      abortRef.current = null
    }
  }

  function handleStop() {
    abortRef.current?.abort()
    setSending(false)
  }

  return (
    <div className="workspace">
      <div className="workspace-header">
        <div>
          <h2>对话工作区</h2>
          <p>支持 ReAct、RAG、工具调用、多轮对话。</p>
        </div>

        <div className="workspace-controls">
          <label>Collection</label>
          <input
            value={activeCollection}
            onChange={(e) => useChatStore.getState().setActiveCollection(e.target.value)}
            placeholder="default"
          />
        </div>
      </div>

      <div className="workspace-body">
        {!currentSessionId ? (
          <EmptyState />
        ) : (
          <MessageList messages={messages} />
        )}
      </div>

      <Composer sending={sending} onSend={handleSend} onStop={handleStop} />

      <div className="workspace-footer">
        <button className="btn btn-secondary" onClick={() => void refreshSessions()}>
          刷新会话列表
        </button>

        {sessions.length > 0 && currentSessionId && (
          <button
            className="btn btn-secondary"
            onClick={() => void loadSessionHistory(currentSessionId)}
          >
            重新加载当前历史
          </button>
        )}
      </div>
    </div>
  )
}