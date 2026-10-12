import { Link, useNavigate } from 'react-router-dom'
import { Button } from '@/components/ui/button'
import { ErrorState, LoadingBlock } from '@/components/States'
import { RecommendationRow } from '@/components/RecommendationRow'
import { StatusTape } from '@/components/StatusTape'
import { useOverview, useProducts, useRecommendations } from '@/hooks/queries'
import { formatDate, formatINRShort, formatPct, formatUnits } from '@/lib/format'
import { MODEL_LABEL, STATUS } from '@/lib/status'

export function OverviewPage() {
  const overview = useOverview()
  const navigate = useNavigate()

  if (overview.isError) return <ErrorState error={overview.error} onRetry={() => overview.refetch()} />
  if (!overview.data) return <LoadingBlock className="h-96" />

  const o = overview.data
  const s = o.summary
  const c = s.status_counts
  const short = c.stockout_risk + c.reorder_due
  const wape = s.wape[s.forecast_model]

  return (
    <div className="flex flex-col gap-12">
      <section aria-labelledby="headline">
        <p className="text-muted-foreground mb-3 text-sm">
          {o.stores} stores and {o.warehouses} warehouses, stock counted on {formatDate(o.as_of)}
        </p>
        <h1 id="headline" className="max-w-4xl text-3xl leading-tight font-semibold text-balance md:text-5xl">
          <span className="num">{formatUnits(short)}</span> of {formatUnits(s.store_items)} store items need stock soon.{' '}
          <span className="text-muted-foreground">
            <span className="num">{formatUnits(c.stockout_risk)}</span> could run out before a delivery arrives.
          </span>
        </h1>
        <div className="mt-8">
          <StatusTape counts={c} onSelect={(st) => navigate(`/map?status=${st}`)} />
        </div>
        <p className="text-muted-foreground mt-4 text-sm">
          A store item is one product family at one store. Select a status to see where it is on the map.
        </p>
      </section>

      <div className="grid gap-12 lg:grid-cols-[1fr_320px]">
        <TopActions asOf={o.as_of} total={s.recommendations.transfer + s.recommendations.reorder} high={s.high_priority} />

        <aside className="flex flex-col gap-8">
          <Figure
            value={formatINRShort(s.overstock_value_inr)}
            label={`tied up in overstock at ${formatUnits(c.overstock)} store items`}
            hint="Transfers move some of it to stores that are running short."
          />
          <Figure
            value={formatINRShort(s.reorder_value_inr)}
            label="of new orders recommended"
            hint={`${s.recommendations.reorder} orders from warehouses and suppliers, ${s.recommendations.transfer} transfers between stores.`}
          />
          {wape !== undefined && (
            <Figure
              value={formatPct(wape)}
              label="average forecast error"
              hint={
                <>
                  Forecasts by the {MODEL_LABEL[s.forecast_model]} model, checked on the last 28 days of sales.{' '}
                  <Link to="/accuracy" className="text-primary underline-offset-4 hover:underline">
                    How accurate is it?
                  </Link>
                </>
              }
            />
          )}
        </aside>
      </div>

      <ProductsAtRisk />
    </div>
  )
}

function Figure({ value, label, hint }: { value: string; label: string; hint?: React.ReactNode }) {
  return (
    <div className="border-t pt-4">
      <p className="num font-display text-3xl font-semibold">{value}</p>
      <p className="mt-1 font-medium">{label}</p>
      {hint && <p className="text-muted-foreground mt-1 text-sm">{hint}</p>}
    </div>
  )
}

function TopActions({ asOf, total, high }: { asOf: string; total: number; high: number }) {
  const recs = useRecommendations({ limit: 5 })
  return (
    <section aria-labelledby="todo">
      <div className="flex items-baseline justify-between gap-4 border-b pb-3">
        <h2 id="todo" className="text-2xl font-semibold">
          Do these first
        </h2>
        <span className="text-muted-foreground text-sm">
          {high} high priority of {total}
        </span>
      </div>
      {recs.isError ? (
        <div className="mt-4">
          <ErrorState error={recs.error} onRetry={() => recs.refetch()} />
        </div>
      ) : !recs.data ? (
        <LoadingBlock className="mt-4 h-72" />
      ) : (
        <div className="divide-y">
          {recs.data.items.map((r) => (
            <RecommendationRow key={r.rec_id} rec={r} asOf={asOf} />
          ))}
        </div>
      )}
      <Button asChild variant="outline" className="mt-4">
        <Link to="/recommendations">See all {total} recommendations</Link>
      </Button>
    </section>
  )
}

function ProductsAtRisk() {
  const products = useProducts()
  if (products.isError) return <ErrorState error={products.error} onRetry={() => products.refetch()} />
  if (!products.data) return <LoadingBlock className="h-48" />
  const top = [...products.data]
    .filter((p) => p.stores_at_risk > 0)
    .sort((a, b) => b.stores_at_risk - a.stores_at_risk || a.name.localeCompare(b.name))
    .slice(0, 6)
  const Icon = STATUS.stockout_risk.icon

  return (
    <section aria-labelledby="products-risk">
      <div className="border-b pb-3">
        <h2 id="products-risk" className="text-2xl font-semibold">
          Products running out in the most stores
        </h2>
      </div>
      {top.length === 0 ? (
        <p className="text-muted-foreground mt-4">No product is at risk of running out at any store.</p>
      ) : (
        <ul className="mt-2 grid gap-x-10 sm:grid-cols-2 lg:grid-cols-3">
          {top.map((p) => (
            <li key={p.product_id} className="flex items-center justify-between gap-4 border-b py-3">
              <span className="min-w-0">
                <span className="block truncate font-medium">{p.name}</span>
                <span className="text-muted-foreground text-sm">{p.category}</span>
              </span>
              <span className="text-stockout-ink inline-flex shrink-0 items-center gap-1.5 text-sm font-medium">
                <Icon className="size-4" aria-hidden />
                <span className="num">{p.stores_at_risk}</span> {p.stores_at_risk === 1 ? 'store' : 'stores'}
              </span>
            </li>
          ))}
        </ul>
      )}
    </section>
  )
}
