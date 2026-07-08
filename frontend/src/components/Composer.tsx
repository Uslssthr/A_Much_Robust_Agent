// frontend/src/components/Composer.tsx
import { useState } from 'react'

interface Props {
  sending: boolean
  onSend: (text: string) => Promise<void> | void
  onStop: () => void
}

export default function Composer({ sending, onSend, onStop }: Props) {
  const [text, setText] = useState('')

  async function handleSend() {
    const value = text.trim()
    if (!value || sending) return
    setText('')
    await onSend(value)
  }

  return (
    <div className="composer">
      <textarea
        className="composer-input"
        rows={4}
        placeholder="输入你的问题... 支持多轮对话、工具调用、RAG。"
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            void handleSend()
          }
        }}
        disabled={sending}
      />

      <div className="composer-actions">
        <div className="composer-tip">Enter 发送，Shift+Enter 换行</div>

        <div className="composer-buttons">
          {sending ? (
            <button className="btn btn-danger" onClick={onStop}>
              停止生成
            </button>
          ) : (
            <button className="btn btn-primary" onClick={() => void handleSend()}>
              发送
            </button>
          )}
        </div>
      </div>
    </div>
  )
}