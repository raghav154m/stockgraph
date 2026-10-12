import { BarChart3, Gauge, Info, LayoutList, Map, Menu, TrendingUp, X } from 'lucide-react'
import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation } from 'react-router-dom'
import { useOverview } from '@/hooks/queries'
import { formatDate } from '@/lib/format'
import { cn } from '@/lib/utils'
import { ThemeToggle } from './ThemeToggle'

const NAV = [
  { to: '/', label: 'Overview', icon: Gauge, end: true },
  { to: '/map', label: 'Map', icon: Map },
  { to: '/recommendations', label: 'Recommendations', icon: LayoutList },
  { to: '/forecast', label: 'Forecasts', icon: TrendingUp },
  { to: '/accuracy', label: 'Model accuracy', icon: BarChart3 },
  { to: '/about', label: 'About the data', icon: Info },
]

function Logo() {
  return (
    <span className="flex items-center gap-2">
      <svg viewBox="0 0 32 32" className="size-7" aria-hidden>
        <rect width="32" height="32" rx="7" fill="var(--primary)" />
        <rect x="6" y="18" width="5" height="8" rx="1.5" fill="var(--primary-foreground)" />
        <rect x="13.5" y="12" width="5" height="14" rx="1.5" fill="var(--primary-foreground)" />
        <rect x="21" y="7" width="5" height="19" rx="1.5" fill="var(--primary-foreground)" />
      </svg>
      <span className="font-display text-lg font-semibold tracking-tight">StockGraph</span>
    </span>
  )
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <ul className="flex flex-col gap-0.5">
      {NAV.map(({ to, label, icon: Icon, end }) => (
        <li key={to}>
          <NavLink
            to={to}
            end={end}
            onClick={onNavigate}
            className={({ isActive }) =>
              cn(
                'flex items-center gap-3 rounded-md px-3 py-2 text-[15px]',
                isActive ? 'bg-accent text-accent-foreground font-medium' : 'text-muted-foreground hover:bg-muted hover:text-foreground',
              )
            }
          >
            <Icon className="size-4 shrink-0" aria-hidden />
            {label}
          </NavLink>
        </li>
      ))}
    </ul>
  )
}

function DatasetNote() {
  const { data } = useOverview()
  if (!data) return null
  return (
    <p className="text-muted-foreground text-xs leading-relaxed">
      {data.name}
      <br />
      Stock as of {formatDate(data.as_of)}
    </p>
  )
}

export function Layout() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => {
    window.scrollTo(0, 0)
  }, [pathname])

  return (
    <div className="min-h-dvh md:grid md:grid-cols-[232px_1fr]">
      <a
        href="#main"
        className="bg-primary text-primary-foreground sr-only z-50 rounded px-3 py-2 focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        Skip to content
      </a>

      {/* desktop sidebar */}
      <aside className="bg-card sticky top-0 hidden h-dvh flex-col justify-between border-r px-3 py-5 md:flex">
        <div className="flex flex-col gap-6">
          <div className="px-3">
            <Logo />
          </div>
          <nav aria-label="Main">
            <NavItems />
          </nav>
        </div>
        <div className="flex flex-col gap-3 px-3">
          <DatasetNote />
          <ThemeToggle />
        </div>
      </aside>

      {/* mobile top bar */}
      <header className="bg-card sticky top-0 z-[1100] flex items-center justify-between border-b px-4 py-3 md:hidden">
        <Logo />
        <button
          type="button"
          className="rounded-md p-2"
          aria-label={open ? 'Close menu' : 'Open menu'}
          aria-expanded={open}
          onClick={() => setOpen((o) => !o)}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </header>
      {open && (
        <div className="bg-card fixed inset-x-0 top-[57px] z-[1100] border-b px-3 py-4 md:hidden">
          <nav aria-label="Main">
            <NavItems onNavigate={() => setOpen(false)} />
          </nav>
          <div className="mt-4 flex items-end justify-between px-3">
            <DatasetNote />
            <ThemeToggle />
          </div>
        </div>
      )}

      <main id="main" className="min-w-0 px-4 py-8 md:px-10 md:py-10">
        <div className="mx-auto max-w-6xl">
          <Outlet />
        </div>
      </main>
    </div>
  )
}
