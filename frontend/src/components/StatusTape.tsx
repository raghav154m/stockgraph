import { useState } from 'react'
import type { Status, StatusCounts } from '@/lib/api'
import { formatPct, formatUnits } from '@/lib/format'
import { STATUS, STATUS_ORDER } from '@/lib/status'
import { cn } from '@/lib/utils'

/**
 * The stock tape: every store x product as one bar, split by status, most urgent on the left.
 * Doubles as the status legend, so it is always labelled (icon, words, count), never colour alone.
 */
export function StatusTape({
  counts,
  onSelect,
  selected,
  size = 'lg',
  columns = 4,
}: {
  counts: StatusCounts
  onSelect?: (s: Status) => void
  selected?: Status | null
  size?: 'lg' | 'sm'
  /** 2 for narrow containers such as a side panel */
  columns?: 2 | 4
}) {
  const total = STATUS_ORDER.reduce((a, s) => a + (counts[s] ?? 0), 0) || 1
  const [hover, setHover] = useState<Status | null>(null)
  const active = hover ?? selected ?? null

  return (
    <figure className="w-full">
      <div
        className={cn('flex w-full gap-[2px] overflow-hidden rounded-md', size === 'lg' ? 'h-10' : 'h-3')}
        role="img"
        aria-label={STATUS_ORDER.map((s) => `${STATUS[s].label}: ${counts[s] ?? 0}`).join(', ')}
      >
        {STATUS_ORDER.map((s) => {
          const n = counts[s] ?? 0
          if (!n) return null
          return (
            <div
              key={s}
              className={cn('h-full transition-opacity', active && active !== s && 'opacity-35')}
              style={{ flexGrow: n, flexBasis: 0, minWidth: 6, background: STATUS[s].areaFill }}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSelect?.(s)}
              title={`${STATUS[s].label}: ${formatUnits(n)} (${formatPct(n / total)})`}
            />
          )
        })}
      </div>
      <figcaption className={cn('grid gap-x-6 gap-y-3', size === 'lg' ? 'mt-4 grid-cols-2 md:grid-cols-4' : columns === 2 ? 'mt-2 grid-cols-2' : 'mt-2 grid-cols-2 sm:grid-cols-4')}>
        {STATUS_ORDER.map((s) => {
          const info = STATUS[s]
          const Icon = info.icon
          const n = counts[s] ?? 0
          const Tag = onSelect ? 'button' : 'div'
          return (
            <Tag
              key={s}
              type={onSelect ? 'button' : undefined}
              onClick={onSelect ? () => onSelect(s) : undefined}
              onMouseEnter={() => setHover(s)}
              onMouseLeave={() => setHover(null)}
              aria-pressed={onSelect ? selected === s : undefined}
              className={cn(
                'flex min-w-0 items-start gap-2 rounded-md text-left',
                onSelect && 'hover:bg-muted -m-1 p-1',
                selected === s && 'bg-muted',
              )}
            >
              <Icon className={cn('mt-0.5 size-4 shrink-0', info.ink)} aria-hidden />
              <span className="min-w-0">
                <span className="flex items-baseline gap-2">
                  <span className={cn('font-medium', size === 'sm' && 'text-sm')}>{info.label}</span>
                  <span className="num text-muted-foreground text-sm">{formatUnits(n)}</span>
                </span>
                {size === 'lg' && <span className="text-muted-foreground block text-sm leading-snug">{info.meaning}</span>}
              </span>
            </Tag>
          )
        })}
      </figcaption>
    </figure>
  )
}
