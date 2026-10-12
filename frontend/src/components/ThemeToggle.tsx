import { Monitor, Moon, Sun } from 'lucide-react'
import { useEffect, useState } from 'react'
import { readTheme, THEME_KEY as KEY, type Theme } from '@/lib/theme'
import { cn } from '@/lib/utils'

const OPTIONS: { value: Theme; label: string; icon: typeof Sun }[] = [
  { value: 'light', label: 'Light', icon: Sun },
  { value: 'dark', label: 'Dark', icon: Moon },
  { value: 'system', label: 'Match device', icon: Monitor },
]

export function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(readTheme)
  useEffect(() => {
    try {
      if (theme === 'system') localStorage.removeItem(KEY)
      else localStorage.setItem(KEY, theme)
    } catch {
      /* storage unavailable: the choice lasts for this visit */
    }
    if (theme === 'system') document.documentElement.removeAttribute('data-theme')
    else document.documentElement.setAttribute('data-theme', theme)
  }, [theme])

  return (
    <div role="radiogroup" aria-label="Colour theme" className="bg-muted inline-flex rounded-md p-0.5">
      {OPTIONS.map(({ value, label, icon: Icon }) => (
        <button
          key={value}
          type="button"
          role="radio"
          aria-checked={theme === value}
          aria-label={label}
          title={label}
          onClick={() => setTheme(value)}
          className={cn(
            'text-muted-foreground rounded p-1.5',
            theme === value && 'bg-card text-foreground shadow-[0_0_0_1px_var(--border)]',
          )}
        >
          <Icon className="size-4" aria-hidden />
        </button>
      ))}
    </div>
  )
}
