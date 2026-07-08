export type ViewMode = 'chat' | 'knowledge'
export type MessageRole = 'user' | 'assistant' | 'system' | 'tool'

export interface SessionSummary {
  session_id: string
  title?: string
  summary?: string
  message_count?: number
  token_total?: number
  created_at?: string
  updated_at?: string
  isLocal?: boolean
}

export interface ToolCallView {
  id: string
  tool: string
  args: Record<string, unknown>
  output?: string
  elapsedMs?: number
  status: 'running' | 'success' | 'error'
}

export interface AssistantMeta {
  route?: string
  iteration?: number
  compressed?: boolean
}

export interface ChatMessage {
  id: string
  role: MessageRole
  content: string
  createdAt: string
  pending?: boolean
  error?: boolean
  toolCalls?: ToolCallView[]
  meta?: AssistantMeta
  thinkingStage?: string
}

export interface KnowledgeDoc {
  doc_id: string
  filename: string
  file_type: string
  file_size: number
  chunk_count: number
  status: 'active' | 'processing' | 'failed' | 'deleted'
  created_at?: string
  updated_at?: string
}

export interface KnowledgeSearchResult {
  content: string
  source: string
  score: number
  chunk_id: string
  metadata: Record<string, unknown>
}

export type StreamEvent =
  | { type: 'thinking'; stage: string }
  | { type: 'token'; content: string }
  | { type: 'tool_start'; tool: string; args: Record<string, unknown> }
  | { type: 'tool_end'; tool: string; output: string; elapsed_ms: number }
  | { type: 'meta'; session_id: string; route: string; iteration: number; compressed: boolean }
  | { type: 'final'; content: string }
  | { type: 'error'; message: string }
  | { type: 'done'; session_id: string }