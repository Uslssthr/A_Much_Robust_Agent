// frontend/src/components/ToolCallPanel.tsx
import type { ToolCallView } from '../types'

interface Props {
  toolCalls?: ToolCallView[]
}

export default function ToolCallPanel({ toolCalls = [] }: Props) {
  if (!toolCalls.length) return null

  return (
    <div className="tool-panel">
      <div className="tool-panel-title">工具调用</div>
      <div className="tool-list">
        {toolCalls.map((call) => (
          <div key={call.id} className="tool-card">
            <div className="tool-card-header">
              <strong>{call.tool}</strong>
              <span className={`status-badge ${call.status}`}>
                {call.status}
              </span>
            </div>

            <div className="tool-card-section">
              <div className="tool-label">输入</div>
              <pre>{JSON.stringify(call.args, null, 2)}</pre>
            </div>

            {call.output && (
              <div className="tool-card-section">
                <div className="tool-label">输出</div>
                <pre>{call.output}</pre>
              </div>
            )}

            {typeof call.elapsedMs === 'number' && (
              <div className="tool-time">{call.elapsedMs.toFixed(1)} ms</div>
            )}
          </div>
        ))}
      </div>
    </div>
  )
}