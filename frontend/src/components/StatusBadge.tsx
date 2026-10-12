import type { Status } from '@/lib/api'
import { STATUS } from '@/lib/status'
import { cn } from '@/lib/utils'

/** Status always appears as icon + words, never colour alone. */
export function StatusBadge({ status, className }: { status: Status; className?: string }) {
  const s = STATUS[status]
  const Icon = s.icon
  return (
    <span className={cn('inline-flex items-center gap-1.5 text-sm font-medium whitespace-nowrap', s.ink, className)}>
      <Icon className="size-4 shrink-0" aria-hidden />
      {s.label}
    </span>
  )
}
