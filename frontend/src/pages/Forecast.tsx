import { useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
  type TooltipContentProps,
} from 'recharts'
import type { NameType, ValueType } from 'recharts/types/component/DefaultTooltipContent'
import { PageHeader } from '@/components/PageHeader'
import { ErrorState, LoadingBlock } from '@/components/States'
import { StatusBadge } from '@/components/StatusBadge'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { useForecast, useLocations, useProducts } from '@/hooks/queries'
import type { Forecast, InventoryItem, ModelName } from '@/lib/api'
import { formatDate, formatDateShort, formatDays, formatPct, formatUnits, formatUnits1 } from '@/lib/format'
import { MODEL_COLOR, MODEL_LABEL, MODEL_SHORT, STATUS } from '@/lib/status'
import { cn } from '@/lib/utils'

const DEFAULT_LOCATION = 'ST-01'
const DEFAULT_PRODUCT = 'grocery-i'
const ALL_STORES = 'ALL'
const COMPARE_ORDER: ModelName[] = ['ensemble', 'baseline', 'stgnn']

const axisProps = {
  stroke: 'var(--chart-grid)',
  tick: { fill: 'var(--chart-axis)', fontSize: 12 },
  tickLine: false,
} as const

export function ForecastPage() {
  const [params, setParams] = useSearchParams()
  const location = params.get('location') ?? DEFAULT_LOCATION
  const product = params.get('product') ?? DEFAULT_PRODUCT
  const compare = (params.get('compare') ?? '').split(',').filter(Boolean) as ModelName[]

  const set = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v === null || v === '') next.delete(k)
      else next.set(k, v)
    }
    setParams(next, { replace: true })
  }

  const locations = useLocations()
  const products = useProducts()
  const forecast = useForecast(location, product)

  const stores = (locations.data ?? []).filter((l) => l.type === 'store').sort((a, b) => a.name.localeCompare(b.name))
  const locationName =
    location === ALL_STORES ? 'all stores' : (locations.data?.find((l) => l.location_id === location)?.name ?? location)
  const productName = forecast.data?.product_name ?? products.data?.find((p) => p.product_id === product)?.name ?? product

  return (
    <>
      <PageHeader title="Forecasts">
        Daily sales for one product at one store, and what the model expects over the next 28 days.
      </PageHeader>

      <div className="mb-8 flex flex-col gap-3 sm:flex-row">
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="text-muted-foreground">Store</span>
          <Select value={location} onValueChange={(v) => set({ location: v })}>
            <SelectTrigger className="w-full sm:w-72">
              <SelectValue placeholder="Choose a store" />
            </SelectTrigger>
            <SelectContent className="max-h-80">
              <SelectItem value={ALL_STORES}>All stores (whole network)</SelectItem>
              <SelectGroup>
                <SelectLabel>Stores</SelectLabel>
                {stores.map((l) => (
                  <SelectItem key={l.location_id} value={l.location_id}>
                    {l.name}
                  </SelectItem>
                ))}
              </SelectGroup>
            </SelectContent>
          </Select>
        </label>
        <label className="flex flex-col gap-1.5 text-sm">
          <span className="text-muted-foreground">Product</span>
          <Select value={product} onValueChange={(v) => set({ product: v })}>
            <SelectTrigger className="w-full sm:w-64">
              <SelectValue placeholder="Choose a product" />
            </SelectTrigger>
            <SelectContent className="max-h-80">
              {(products.data ?? [])
                .slice()
                .sort((a, b) => a.name.localeCompare(b.name))
                .map((p) => (
                  <SelectItem key={p.product_id} value={p.product_id}>
                    {p.name}
                  </SelectItem>
                ))}
            </SelectContent>
          </Select>
        </label>
      </div>

      {forecast.isError ? (
        <ErrorState error={forecast.error} onRetry={() => forecast.refetch()} />
      ) : !forecast.data ? (
        <LoadingBlock className="h-[420px]" />
      ) : (
        <div className={cn('flex flex-col gap-12', forecast.isPlaceholderData && 'opacity-60')}>
          <section aria-labelledby="fc-title">
            <h2 id="fc-title" className="text-2xl font-semibold">
              {productName} at {locationName}
            </h2>
            <ForecastSummary data={forecast.data} />
            <ModelToggles
              recommended={forecast.data.recommendation_model}
              available={COMPARE_ORDER.filter((m) => forecast.data!.forecasts[m]?.length)}
              compare={compare}
              onChange={(next) => set({ compare: next.join(',') })}
            />
            <ForecastChart data={forecast.data} compare={compare} />
          </section>

          {forecast.data.inventory && <StockPanel item={forecast.data.inventory} />}

          <BacktestSection data={forecast.data} />
        </div>
      )}
    </>
  )
}

function ForecastSummary({ data }: { data: Forecast }) {
  const fc = data.forecasts[data.recommendation_model] ?? []
  if (!fc.length) return <p className="text-muted-foreground mt-2">No forecast is available for this product here.</p>
  const p50 = fc.reduce((a, d) => a + d.p50, 0)
  const recent = data.history.slice(-28)
  const recentTotal = recent.reduce((a, d) => a + d.units_sold, 0)
  const change = recentTotal > 0 ? p50 / recentTotal - 1 : null
  return (
    <p className="text-muted-foreground mt-2 max-w-[70ch]">
      Expected to sell about <span className="text-foreground num font-medium">{formatUnits(p50)} units</span> from{' '}
      {formatDate(fc[0].date)} to {formatDate(fc[fc.length - 1].date)}
      {change !== null && (
        <>
          , {Math.abs(change) < 0.02 ? 'about the same as' : change > 0 ? `${formatPct(change, 0)} more than` : `${formatPct(-change, 0)} less than`}{' '}
          the last 28 days
        </>
      )}
      . The shaded band is where daily sales should land 8 days in 10.
    </p>
  )
}

function ModelToggles({
  recommended,
  available,
  compare,
  onChange,
}: {
  recommended: ModelName
  available: ModelName[]
  compare: ModelName[]
  onChange: (next: ModelName[]) => void
}) {
  const others = available.filter((m) => m !== recommended)
  return (
    <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-sm">
      <LegendItem kind="line" color="var(--foreground)" label="Actual sales" />
      <LegendItem kind="line" color={MODEL_COLOR[recommended]} label={`Forecast (${MODEL_SHORT[recommended]}, used for recommendations)`} />
      <LegendItem kind="band" color="var(--chart-band)" label="80% range" />
      {others.length > 0 && <span className="text-muted-foreground">Compare with:</span>}
      {others.map((m) => {
        const on = compare.includes(m)
        return (
          <label key={m} className="inline-flex cursor-pointer items-center gap-2">
            <input
              type="checkbox"
              className="accent-primary size-4"
              checked={on}
              onChange={() => onChange(on ? compare.filter((x) => x !== m) : [...compare, m])}
            />
            <Swatch kind="dash" color={MODEL_COLOR[m]} />
            {MODEL_LABEL[m]}
          </label>
        )
      })}
    </div>
  )
}

function Swatch({ kind, color }: { kind: 'line' | 'band' | 'dash'; color: string }) {
  if (kind === 'band') return <span className="inline-block h-3 w-5 rounded-sm opacity-25" style={{ background: color }} aria-hidden />
  return (
    <svg width="20" height="8" aria-hidden className="shrink-0">
      <line x1="0" y1="4" x2="20" y2="4" stroke={color} strokeWidth="2" strokeDasharray={kind === 'dash' ? '4 3' : undefined} />
    </svg>
  )
}

function LegendItem({ kind, color, label }: { kind: 'line' | 'band'; color: string; label: string }) {
  return (
    <span className="inline-flex items-center gap-2">
      <Swatch kind={kind} color={color} />
      {label}
    </span>
  )
}

type Row = {
  date: string
  actual?: number
  promo?: boolean
  p50?: number
  band?: [number, number]
} & Partial<Record<ModelName, number>>

function ForecastChart({ data, compare }: { data: Forecast; compare: ModelName[] }) {
  const rec = data.recommendation_model
  const rows = useMemo(() => {
    const byDate = new Map<string, Row>()
    const row = (d: string) => {
      let r = byDate.get(d)
      if (!r) byDate.set(d, (r = { date: d }))
      return r
    }
    for (const h of data.history) {
      const r = row(h.date)
      r.actual = h.units_sold
      r.promo = h.on_promotion > 0
    }
    for (const f of data.forecasts[rec] ?? []) {
      const r = row(f.date)
      r.p50 = f.p50
      r.band = [f.p10, f.p90]
    }
    for (const m of compare) for (const f of data.forecasts[m] ?? []) row(f.date)[m] = f.p50
    // join the actual line to the forecast so there is no gap at "today"
    const last = data.history[data.history.length - 1]
    if (last) {
      const r = row(last.date)
      r.p50 = last.units_sold
      r.band = [last.units_sold, last.units_sold]
      for (const m of compare) r[m] = last.units_sold
    }
    return [...byDate.values()].sort((a, b) => a.date.localeCompare(b.date))
  }, [data, compare, rec])

  return (
    <figure className="mt-4">
      <div className="h-[340px] w-full md:h-[400px]">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 16, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            <XAxis dataKey="date" {...axisProps} tickFormatter={formatDateShort} minTickGap={36} />
            <YAxis
              {...axisProps}
              axisLine={false}
              width={48}
              tickFormatter={(v: number) => formatUnits(v)}
              label={{ value: 'units a day', angle: -90, position: 'insideLeft', fill: 'var(--chart-axis)', fontSize: 12, dy: 40 }}
            />
            <ReferenceLine
              x={data.as_of}
              stroke="var(--muted-foreground)"
              label={{ value: 'Today', position: 'insideTopLeft', fill: 'var(--muted-foreground)', fontSize: 12 }}
            />
            <Area
              dataKey="band"
              stroke="none"
              fill="var(--chart-band)"
              fillOpacity={0.16}
              isAnimationActive={false}
              activeDot={false}
            />
            <Line dataKey="actual" stroke="var(--foreground)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {compare.map((m) => (
              <Line
                key={m}
                dataKey={m}
                stroke={MODEL_COLOR[m]}
                strokeWidth={2}
                strokeDasharray="5 4"
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <Line dataKey="p50" stroke={MODEL_COLOR[rec]} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Tooltip content={(p) => <ForecastTooltip {...p} asOf={data.as_of} rec={rec} />} cursor={{ stroke: 'var(--muted-foreground)', strokeWidth: 1 }} />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
      <figcaption className="sr-only">
        Line chart of daily sales for the last {data.history.length} days, then the 28-day forecast with an 80% range.
      </figcaption>
    </figure>
  )
}

function ForecastTooltip({
  active,
  payload,
  label,
  asOf,
  rec,
}: TooltipContentProps<ValueType, NameType> & { asOf: string; rec: ModelName }) {
  if (!active || !payload?.length) return null
  const r = payload[0].payload as Row
  const future = String(label) > asOf
  return (
    <div className="bg-popover text-popover-foreground rounded-md border px-3 py-2 text-sm shadow-sm">
      <p className="font-medium">{formatDate(String(label))}</p>
      {!future && r.actual !== undefined && (
        <p className="num">
          Sold {formatUnits1(r.actual)}
          {r.promo && <span className="text-muted-foreground"> (on promotion)</span>}
        </p>
      )}
      {future && r.p50 !== undefined && (
        <>
          <p className="num">
            {MODEL_SHORT[rec]}: {formatUnits1(r.p50)}
          </p>
          {r.band && (
            <p className="text-muted-foreground num">
              Likely {formatUnits1(r.band[0])} to {formatUnits1(r.band[1])}
            </p>
          )}
          {(['ensemble', 'baseline', 'stgnn'] as ModelName[])
            .filter((m) => m !== rec && r[m] !== undefined)
            .map((m) => (
              <p key={m} className="num">
                {MODEL_SHORT[m]}: {formatUnits1(r[m]!)}
              </p>
            ))}
        </>
      )}
    </div>
  )
}

function StockPanel({ item }: { item: InventoryItem }) {
  const s = STATUS[item.status]
  const stats: { label: string; value: string; hint: string }[] = [
    { label: 'On hand', value: formatUnits(item.on_hand), hint: 'units in the store now' },
    { label: 'On order', value: formatUnits(item.on_order), hint: 'units already on the way' },
    {
      label: 'Lasts about',
      value: item.days_of_cover === null ? 'n/a' : formatDays(item.days_of_cover),
      hint: `at ${formatUnits1(item.avg_daily_forecast)} units a day`,
    },
    { label: 'Reorder point', value: formatUnits1(item.reorder_point), hint: 'order when stock falls below this' },
    { label: 'Safety stock', value: formatUnits1(item.safety_stock), hint: 'cushion for a busier-than-expected week' },
    { label: 'Order up to', value: formatUnits1(item.target_level), hint: `delivery takes ${formatUnits(item.lead_time_days)} ${item.lead_time_days === 1 ? 'day' : 'days'}` },
  ]
  return (
    <section aria-labelledby="stock-title">
      <div className="flex flex-wrap items-baseline justify-between gap-3 border-b pb-3">
        <h2 id="stock-title" className="text-2xl font-semibold">
          Stock at this store
        </h2>
        <StatusBadge status={item.status} />
      </div>
      <p className="text-muted-foreground mt-3 text-sm">{s.meaning}.</p>
      <dl className="mt-4 grid grid-cols-2 gap-x-8 gap-y-5 md:grid-cols-3 lg:grid-cols-6">
        {stats.map((x) => (
          <div key={x.label}>
            <dt className="text-muted-foreground text-sm">{x.label}</dt>
            <dd className="num font-display mt-0.5 text-2xl font-semibold">{x.value}</dd>
            <dd className="text-muted-foreground text-xs">{x.hint}</dd>
          </div>
        ))}
      </dl>
    </section>
  )
}

function BacktestSection({ data }: { data: Forecast }) {
  const rec = data.recommendation_model
  const rows = data.backtest
  const scored = (['ensemble', 'baseline', 'stgnn', 'naive'] as ModelName[]).filter((m) =>
    rows.some((r) => r[m] !== null),
  )
  const actualSum = rows.reduce((a, r) => a + (r.actual ?? 0), 0)
  const wape = (m: ModelName) =>
    actualSum > 0 ? rows.reduce((a, r) => a + Math.abs((r[m] ?? 0) - (r.actual ?? 0)), 0) / actualSum : null

  if (!rows.length) return null
  const shown: ModelName[] = [rec, ...(scored.includes('naive') && rec !== 'naive' ? (['naive'] as ModelName[]) : [])]

  return (
    <section aria-labelledby="bt-title">
      <h2 id="bt-title" className="border-b pb-3 text-2xl font-semibold">
        How the forecast did on the last 28 days
      </h2>
      <p className="text-muted-foreground mt-3 max-w-[70ch] text-sm">
        Before forecasting the future, each model predicted the 28 days up to {formatDate(data.as_of)} without seeing them.
        Here is that prediction against what actually sold. Error is the total miss as a share of what sold.
      </p>
      <div className="mt-4 flex flex-wrap gap-x-5 gap-y-2 text-sm">
        <LegendItem kind="line" color="var(--foreground)" label="Actual sales" />
        {shown.map((m) => (
          <span key={m} className="inline-flex items-center gap-2">
            <Swatch kind={m === rec ? 'line' : 'dash'} color={MODEL_COLOR[m]} />
            {MODEL_SHORT[m]}
            {wape(m) !== null && <span className="text-muted-foreground num">error {formatPct(wape(m)!)}</span>}
          </span>
        ))}
      </div>
      <div className="mt-2 h-[260px] w-full">
        <ResponsiveContainer width="100%" height="100%">
          <ComposedChart data={rows} margin={{ top: 12, right: 12, bottom: 0, left: 0 }}>
            <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
            <XAxis dataKey="date" {...axisProps} tickFormatter={formatDateShort} minTickGap={36} />
            <YAxis {...axisProps} axisLine={false} width={48} tickFormatter={(v: number) => formatUnits(v)} />
            <Line dataKey="actual" name="Actual" stroke="var(--foreground)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
            {shown.map((m) => (
              <Line
                key={m}
                dataKey={m}
                name={MODEL_SHORT[m]}
                stroke={MODEL_COLOR[m]}
                strokeWidth={2}
                strokeDasharray={m === rec ? undefined : '5 4'}
                dot={false}
                isAnimationActive={false}
              />
            ))}
            <Tooltip
              cursor={{ stroke: 'var(--muted-foreground)', strokeWidth: 1 }}
              content={({ active, payload, label }) =>
                active && payload?.length ? (
                  <div className="bg-popover text-popover-foreground rounded-md border px-3 py-2 text-sm shadow-sm">
                    <p className="font-medium">{formatDate(String(label))}</p>
                    {payload.map((p) => (
                      <p key={String(p.dataKey)} className="num">
                        {p.name}: {p.value === null || p.value === undefined ? 'n/a' : formatUnits1(Number(p.value))}
                      </p>
                    ))}
                  </div>
                ) : null
              }
            />
          </ComposedChart>
        </ResponsiveContainer>
      </div>
    </section>
  )
}
