
const USER_ID_KEY = 'agent_demo_user_id'

export function getOrCreateUserId(): string {
  const existing = localStorage.getItem(USER_ID_KEY)
  if (existing) return existing

  const created = `user_${crypto.randomUUID()}`
  localStorage.setItem(USER_ID_KEY, created)
  return created
}