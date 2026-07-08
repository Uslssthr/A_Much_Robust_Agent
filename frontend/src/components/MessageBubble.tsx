import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import 'highlight.js/styles/github-dark.css'

import type { ChatMessage } from '../types'
import ToolCallPanel from './ToolCallPanel'

interface Props {
  message: ChatMessage
}

export default function MessageBubble({ message }: Props) {
  const isUser = message.role === 'user'
  const isAssistant = message.role === 'assistant'

  // 实时状态文案：优先显示"正在调用工具xxx"，否则显示通用thinking文案
  const runningTool = message.toolCalls?.find((t) => t.status === 'running')
  const statusText = runningTool
    ? `🔧 正在调用工具: ${runningTool.tool}`
    : message.thinkingStage || 'Agent 正在思考...'

  return (
    <div className={`message-row ${isUser ? 'user' : 'assistant'}`}>
      <div
        className={`message-bubble ${isUser ? 'user' : 'assistant'} ${
          message.error ? 'error' : ''
        }`}
      >
        <div className="message-role">
          {isUser ? '你' : isAssistant ? 'Agent' : message.role}
        </div>

        <div className="message-content">
          {isUser ? (
            // 用户消息保持纯文本展示
            <div className="plain-text">{message.content}</div>
          ) : message.content ? (
            // 助手消息用 Markdown 渲染
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeHighlight]}
            >
              {message.content}
            </ReactMarkdown>
          ) : message.pending ? (
            <div className="thinking-indicator">
              <span className="dot" />
              <span className="dot" />
              <span className="dot" />
              <span className="thinking-text">{statusText}</span>
            </div>
          ) : null}
        </div>

        {message.meta && (
          <div className="message-meta">
            {message.meta.route && <span>route: {message.meta.route}</span>}
            {typeof message.meta.iteration === 'number' && (
              <span>steps: {message.meta.iteration}</span>
            )}
            {message.meta.compressed && <span>context compressed</span>}
          </div>
        )}
      </div>

      <ToolCallPanel toolCalls={message.toolCalls} />
    </div>
  )
}