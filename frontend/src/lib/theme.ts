export type Theme = 'system' | 'light' | 'dark'
export const THEME_KEY = 'stockgraph-theme'

export function readTheme(): Theme {
  try {
    const v = localStorage.getItem(THEME_KEY)
    return v === 'light' || v === 'dark' ? v : 'system'
  } catch {
    return 'system'
  }
}

export function applyStoredTheme() {
  const t = readTheme()
  if (t === 'system') document.documentElement.removeAttribute('data-theme')
  else document.documentElement.setAttribute('data-theme', t)
}
