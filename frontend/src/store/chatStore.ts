import { create } from 'zustand'
import type { ViewMode, SessionSummary, ChatMessage } from '../types'

interface ChatState {
  view: ViewMode
  sessions: SessionSummary[]
  currentSessionId: string | null
  messagesBySession: Record<string, ChatMessage[]>
  sending: boolean
  activeCollection: string
  error: string | null

  setView: (view: ViewMode) => void
  setSessions: (sessions: SessionSummary[]) => void
  upsertSession: (session: SessionSummary) => void
  removeSession: (sessionId: string) => void
  renameSession: (oldId: string, newId: string) => void
  setCurrentSession: (sessionId: string | null) => void
  setMessages: (sessionId: string, messages: ChatMessage[]) => void
  appendMessage: (sessionId: string, message: ChatMessage) => void
  updateMessage: (
    sessionId: string,
    messageId: string,
    updater: (msg: ChatMessage) => ChatMessage
  ) => void
  setSending: (sending: boolean) => void
  setActiveCollection: (collection: string) => void
  setError: (error: string | null) => void
  startLocalSession: (title?: string) => string
}

export const useChatStore = create<ChatState>((set, get) => ({
  view: 'chat',
  sessions: [],
  currentSessionId: null,
  messagesBySession: {},
  sending: false,
  activeCollection: 'default',
  error: null,

  setView: (view) => set({ view }),
  setSessions: (sessions) => set({ sessions }),
  setCurrentSession: (sessionId) => set({ currentSessionId: sessionId }),
  setSending: (sending) => set({ sending }),
  setActiveCollection: (activeCollection) => set({ activeCollection }),
  setError: (error) => set({ error }),

  upsertSession: (session) =>
    set((state) => {
      const exists = state.sessions.some((s) => s.session_id === session.session_id)
      if (exists) {
        return {
          sessions: state.sessions.map((s) =>
            s.session_id === session.session_id ? { ...s, ...session } : s
          )
        }
      }
      return {
        sessions: [session, ...state.sessions]
      }
    }),

  removeSession: (sessionId) =>
    set((state) => {
      const nextMessages = { ...state.messagesBySession }
      delete nextMessages[sessionId]

      const nextCurrent =
        state.currentSessionId === sessionId ? null : state.currentSessionId

      return {
        sessions: state.sessions.filter((s) => s.session_id !== sessionId),
        messagesBySession: nextMessages,
        currentSessionId: nextCurrent
      }
    }),

  renameSession: (oldId, newId) =>
    set((state) => {
      if (oldId === newId) return state

      const nextMessages = { ...state.messagesBySession }
      if (nextMessages[oldId]) {
        nextMessages[newId] = nextMessages[oldId]
        delete nextMessages[oldId]
      }

      return {
        sessions: state.sessions.map((s) =>
          s.session_id === oldId
            ? { ...s, session_id: newId, isLocal: false }
            : s
        ),
        messagesBySession: nextMessages,
        currentSessionId:
          state.currentSessionId === oldId ? newId : state.currentSessionId
      }
    }),

  setMessages: (sessionId, messages) =>
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: messages
      }
    })),

  appendMessage: (sessionId, message) =>
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: [...(state.messagesBySession[sessionId] || []), message]
      }
    })),

  updateMessage: (sessionId, messageId, updater) =>
    set((state) => ({
      messagesBySession: {
        ...state.messagesBySession,
        [sessionId]: (state.messagesBySession[sessionId] || []).map((msg) =>
          msg.id === messageId ? updater(msg) : msg
        )
      }
    })),

  startLocalSession: (title) => {
    const localId = `local-${crypto.randomUUID()}`
    const session: SessionSummary = {
      session_id: localId,
      title: title || '新对话',
      summary: '',
      updated_at: new Date().toISOString(),
      created_at: new Date().toISOString(),
      isLocal: true,
      message_count: 0
    }

    set((state) => ({
      sessions: [session, ...state.sessions],
      currentSessionId: localId,
      messagesBySession: {
        ...state.messagesBySession,
        [localId]: []
      }
    }))

    return localId
  }
}))