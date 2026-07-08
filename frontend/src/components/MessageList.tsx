import { useEffect, useRef } from 'react'
import type { ChatMessage } from '../types'
import MessageBubble from './MessageBubble'

interface Props {
  messages: ChatMessage[]
}

export default function MessageList({ messages }: Props) {
  const bottomRef = useRef<HTMLDivElement | null>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  return (
    <div className="message-list">
      {messages.map((msg) => (
        <MessageBubble key={msg.id} message={msg} />
      ))}
      <div ref={bottomRef} />
    </div>
  )
}