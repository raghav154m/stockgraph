import { ArrowRight, SignalHigh, SignalLow, SignalMedium, Truck, ArrowLeftRight } from 'lucide-react'
import { Link } from 'react-router-dom'
import type { Recommendation } from '@/lib/api'
import { daysBetween, dueText, formatDate, formatINR, formatUnits } from '@/lib/format'
import { cn } from '@/lib/utils'

const PRIORITY = {
  high: { label: 'High priority', icon: SignalHigh, cls: 'text-foreground font-medium' },
  medium: { label: 'Medium priority', icon: SignalMedium, cls: 'text-muted-foreground' },
  low: { label: 'Low priority', icon: SignalLow, cls: 'text-muted-foreground' },
} as const

export function RecommendationRow({ rec, asOf, compact = false }: { rec: Recommendation; asOf: string; compact?: boolean }) {
  const p = PRIORITY[rec.priority]
  const PIcon = p.icon
  const isTransfer = rec.type === 'transfer'
  const TypeIcon = isTransfer ? ArrowLeftRight : Truck
  const due = daysBetween(asOf, rec.deadline)

  return (
    <article className={cn('grid gap-x-6 gap-y-2 py-4', !compact && 'md:grid-cols-[1fr_auto]')}>
      <div className="min-w-0">
        <div className="text-muted-foreground mb-1 flex flex-wrap items-center gap-x-4 gap-y-1 text-sm">
          <span className={cn('inline-flex items-center gap-1.5', p.cls)}>
            <PIcon className="size-4" aria-hidden />
            {p.label}
          </span>
          <span className="inline-flex items-center gap-1.5">
            <TypeIcon className="size-4" aria-hidden />
            {isTransfer ? 'Transfer between stores' : rec.from_location === 'SUPPLIER' ? 'Order from supplier' : 'Order from warehouse'}
          </span>
          <span className={cn(due <= 0 && 'text-stockout-ink font-medium')} title={formatDate(rec.deadline)}>
            {dueText(rec.deadline, asOf)}
          </span>
        </div>
        <h3 className="font-sans text-base font-medium tracking-normal">
          {isTransfer ? 'Move' : 'Order'} <span className="num">{formatUnits(rec.quantity)}</span> units of{' '}
          {rec.product_name}
        </h3>
        <p className="mt-0.5 flex flex-wrap items-center gap-1.5 text-sm">
          <span>{rec.from_name}</span>
          <ArrowRight className="text-muted-foreground size-3.5" aria-label="to" />
          <span>{rec.to_name}</span>
        </p>
        {!compact && <p className="text-muted-foreground mt-2 max-w-[70ch] text-sm">{rec.reason}</p>}
      </div>
      <div className={cn('flex items-baseline gap-4 text-sm', !compact && 'md:flex-col md:items-end md:gap-1')}>
        <span className="num font-medium" title={isTransfer ? 'Value of stock moved' : 'Cost of the order'}>
          {formatINR(rec.value_inr)}
        </span>
        <Link
          to={`/forecast?location=${encodeURIComponent(rec.to_location)}&product=${encodeURIComponent(rec.product_id)}`}
          className="text-primary underline-offset-4 hover:underline"
        >
          See forecast
        </Link>
      </div>
    </article>
  )
}
