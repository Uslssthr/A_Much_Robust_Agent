export default function EmptyState() {
  return (
    <div className="empty-state">
      <h2>通用 Agent 控制台</h2>
      <p>你可以开始一个新对话，或者切换到知识库页上传文档。</p>
      <ul>
        <li>支持多轮对话</li>
        <li>支持工具调用可视化</li>
        <li>支持知识库问答</li>
        <li>支持流式输出</li>
      </ul>
    </div>
  )
}