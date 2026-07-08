// frontend/src/components/Sidebar.tsx
import type { SessionSummary, ViewMode } from '../types'

interface Props {
  userId: string
  view: ViewMode
  sessions: SessionSummary[]
  currentSessionId: string | null
  onSwitchView: (view: ViewMode) => void
  onNewChat: () => void
  onSelectSession: (sessionId: string) => void
  onDeleteSession: (sessionId: string) => void
}

export default function Sidebar({
  userId,
  view,
  sessions,
  currentSessionId,
  onSwitchView,
  onNewChat,
  onSelectSession,
  onDeleteSession
}: Props) {
  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <div className="brand">
          <div className="brand-title">Universal Agent</div>
          <div className="brand-subtitle">{userId.slice(0, 16)}</div>
        </div>
      </div>

      <div className="sidebar-tabs">
        <button
          className={`tab-btn ${view === 'chat' ? 'active' : ''}`}
          onClick={() => onSwitchView('chat')}
        >
          对话
        </button>
        <button
          className={`tab-btn ${view === 'knowledge' ? 'active' : ''}`}
          onClick={() => onSwitchView('knowledge')}
        >
          知识库
        </button>
      </div>

      {view === 'chat' && (
        <>
          <div className="sidebar-actions">
            <button className="btn btn-primary btn-block" onClick={onNewChat}>
              + 新建对话
            </button>
          </div>

          <div className="session-list">
            {sessions.length === 0 && (
              <div className="session-empty">暂无会话</div>
            )}

            {sessions.map((session) => (
              <div
                key={session.session_id}
                className={`session-item ${currentSessionId === session.session_id ? 'active' : ''}`}
                onClick={() => onSelectSession(session.session_id)}
              >
                <div className="session-main">
                  <div className="session-title">
                    {session.title || '未命名会话'}
                    {session.isLocal && <span className="local-badge">local</span>}
                  </div>
                  <div className="session-summary">
                    {session.summary || '点击查看历史消息'}
                  </div>
                </div>

                <button
                  className="session-delete"
                  onClick={(e) => {
                    e.stopPropagation()
                    onDeleteSession(session.session_id)
                  }}
                  title="删除会话"
                >
                  ×
                </button>
              </div>
            ))}
          </div>
        </>
      )}
    </aside>
  )
}