import { ArrowDown, ArrowUp } from 'lucide-react'
import { useMemo, useState } from 'react'
import { Link } from 'react-router-dom'
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PageHeader } from '@/components/PageHeader'
import { ErrorState, LoadingBlock } from '@/components/States'
import { Button } from '@/components/ui/button'
import { useMetrics } from '@/hooks/queries'
import type { Metrics, ModelName } from '@/lib/api'
import { formatDate, formatPct } from '@/lib/format'
import { MODEL_LABEL, MODEL_SHORT } from '@/lib/status'
import { cn } from '@/lib/utils'

const MODEL_ORDER: ModelName[] = ['ensemble', 'baseline', 'stgnn', 'naive']

const MODEL_BLURB: Record<ModelName, string> = {
  ensemble: 'Half ST-GNN, half gradient boosting. Drives the recommendations.',
  baseline: 'Learns from each store’s recent sales, weekday, promotions and holidays.',
  stgnn: 'A graph neural network that also learns from nearby and similar stores.',
  naive: 'No model at all, just recent sales repeated. The bar any model has to clear.',
}

export function AccuracyPage() {
  const metrics = useMetrics()
  return (
    <>
      <PageHeader title="Model accuracy">
        Every model forecast the last 28 days without seeing them, then was scored against what actually sold. Error
        is the total miss as a share of total sales, so lower is better.
      </PageHeader>
      {metrics.isError ? (
        <ErrorState error={metrics.error} onRetry={() => metrics.refetch()} />
      ) : !metrics.data ? (
        <LoadingBlock className="h-96" />
      ) : (
        <div className="flex flex-col gap-14">
          <Overall m={metrics.data} />
          <Rolling m={metrics.data} />
          <ByStore m={metrics.data} />
          <Training m={metrics.data} />
        </div>
      )}
    </>
  )
}

function Overall({ m }: { m: Metrics }) {
  const rec = m.overall.find((r) => r.model === m.forecast_model)
  const naive = m.overall.find((r) => r.model === 'naive')
  const max = Math.max(...m.overall.map((r) => r.wape), 0.0001)

  return (
    <section aria-labelledby="overall">
      <h2 id="overall" className="sr-only">
        Overall accuracy
      </h2>
      {rec && (
        <p className="font-display max-w-3xl text-2xl leading-snug font-medium md:text-3xl">
          The forecast behind the recommendations missed by{' '}
          <span className="num">{formatPct(rec.wape)}</span> of what sold
          {naive && (
            <>
              . Repeating recent weeks would have missed by <span className="num">{formatPct(naive.wape)}</span>
            </>
          )}
          .
        </p>
      )}

      <ul className="mt-8 flex flex-col gap-5" aria-label="Error by model, lower is better">
        {m.overall.map((r) => {
          const isRec = r.model === m.forecast_model
          return (
            <li key={r.model} className="grid gap-x-6 gap-y-1 md:grid-cols-[260px_1fr]">
              <div>
                <p className={cn('font-medium', !isRec && 'text-muted-foreground')}>
                  {MODEL_LABEL[r.model]}
                  {isRec && <span className="text-primary ml-2 text-sm font-medium">Used</span>}
                </p>
                <p className="text-muted-foreground text-sm">{MODEL_BLURB[r.model]}</p>
              </div>
              <div className="flex items-center gap-3">
                <div className="bg-muted h-6 flex-1 rounded-sm">
                  <div
                    className="h-full rounded-r-[4px]"
                    style={{
                      width: `${(r.wape / max) * 100}%`,
                      background: isRec ? 'var(--chart-1)' : 'var(--muted-foreground)',
                      opacity: isRec ? 1 : 0.45,
                    }}
                  />
                </div>
                <span className={cn('num w-14 text-right font-medium', !isRec && 'text-muted-foreground')}>
                  {formatPct(r.wape)}
                </span>
              </div>
            </li>
          )
        })}
      </ul>

      <BiasNote m={m} />
    </section>
  )
}

function BiasNote({ m }: { m: Metrics }) {
  const rec = m.overall.find((r) => r.model === m.forecast_model)
  if (!rec || Math.abs(rec.bias) < 0.005) return null
  return (
    <p className="text-muted-foreground mt-6 max-w-[70ch] text-sm">
      On balance the {MODEL_SHORT[rec.model].toLowerCase()} forecast ran{' '}
      <span className="text-foreground num font-medium">
        {formatPct(Math.abs(rec.bias))} {rec.bias < 0 ? 'low' : 'high'}
      </span>
      . Safety stock is sized from each store’s own forecast errors, so this is already allowed for in the reorder
      points.
    </p>
  )
}

function Rolling({ m }: { m: Metrics }) {
  const holdouts = [...new Set(m.rolling.map((r) => r.holdout_start))].sort((a, b) =>
    a === 'ALL' ? 1 : b === 'ALL' ? -1 : a.localeCompare(b),
  )
  const models = MODEL_ORDER.filter((x) => m.rolling.some((r) => r.model === x))
  if (!holdouts.length) return null
  const cell = (h: string, x: ModelName) => m.rolling.find((r) => r.holdout_start === h && r.model === x)?.wape
  const periods = holdouts.filter((h) => h !== 'ALL').length

  return (
    <section aria-labelledby="rolling">
      <h2 id="rolling" className="border-b pb-3 text-2xl font-semibold">
        Checked over {periods === 1 ? 'one 28-day period' : `${periods} back-to-back 28-day periods`}
      </h2>
      <p className="text-muted-foreground mt-3 max-w-[70ch] text-sm">
        One good month can be luck, so the models were also scored on earlier periods. Each column is a model; the
        lowest error in each row is marked.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[520px] text-sm">
          <thead>
            <tr className="text-muted-foreground border-b text-left">
              <th className="py-2 pr-4 font-medium">28 days from</th>
              {models.map((x) => (
                <th key={x} className="py-2 pr-4 text-right font-medium">
                  {MODEL_SHORT[x]}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {holdouts.map((h) => {
              const vals = models.map((x) => cell(h, x)).filter((v): v is number => v !== undefined)
              const best = Math.min(...vals)
              return (
                <tr key={h} className={cn('border-b', h === 'ALL' && 'font-medium')}>
                  <td className="py-2 pr-4">{h === 'ALL' ? 'All periods together' : formatDate(h)}</td>
                  {models.map((x) => {
                    const v = cell(h, x)
                    const isBest = v !== undefined && v === best
                    return (
                      <td key={x} className="num py-2 pr-4 text-right">
                        {v === undefined ? '–' : formatPct(v)}
                        {isBest && <span className="text-primary ml-1" aria-label="lowest">●</span>}
                      </td>
                    )
                  })}
                </tr>
              )
            })}
          </tbody>
        </table>
      </div>
    </section>
  )
}

type SortKey = 'name' | ModelName

function ByStore({ m }: { m: Metrics }) {
  const rec = m.forecast_model
  const models = MODEL_ORDER.filter((x) => m.by_store.some((r) => r.wape[x] !== undefined))
  const [sort, setSort] = useState<{ key: SortKey; desc: boolean }>({ key: rec, desc: true })
  const [all, setAll] = useState(false)

  const rows = useMemo(() => {
    const r = [...m.by_store]
    r.sort((a, b) => {
      const d =
        sort.key === 'name' ? a.name.localeCompare(b.name) : (a.wape[sort.key] ?? Infinity) - (b.wape[sort.key] ?? Infinity)
      return sort.desc ? -d : d
    })
    return r
  }, [m.by_store, sort])

  if (!rows.length) return null
  const visible = all ? rows : rows.slice(0, 10)
  const beatsNaive = m.by_store.filter((r) => (r.wape[rec] ?? Infinity) < (r.wape.naive ?? Infinity)).length

  const header = (key: SortKey, label: string, right = true) => {
    const active = sort.key === key
    const Icon = sort.desc ? ArrowDown : ArrowUp
    return (
      <th key={key} className={cn('py-2 pr-4 font-medium', right && 'text-right')} aria-sort={active ? (sort.desc ? 'descending' : 'ascending') : 'none'}>
        <button
          type="button"
          className={cn('inline-flex items-center gap-1', active && 'text-foreground')}
          onClick={() => setSort({ key, desc: active ? !sort.desc : key !== 'name' })}
        >
          {label}
          {active && <Icon className="size-3.5" aria-hidden />}
        </button>
      </th>
    )
  }

  return (
    <section aria-labelledby="by-store">
      <h2 id="by-store" className="border-b pb-3 text-2xl font-semibold">
        Store by store
      </h2>
      <p className="text-muted-foreground mt-3 max-w-[70ch] text-sm">
        The {MODEL_SHORT[rec].toLowerCase()} forecast beat repeating recent weeks at{' '}
        <span className="text-foreground num font-medium">
          {beatsNaive} of {m.by_store.length}
        </span>{' '}
        stores. Smaller stores sell fewer units, so a few units off shows up as a larger error there. Hardest to forecast
        first.
      </p>
      <div className="mt-4 overflow-x-auto">
        <table className="w-full min-w-[560px] text-sm">
          <thead>
            <tr className="text-muted-foreground border-b text-left">
              {header('name', 'Store', false)}
              {models.map((x) => header(x, MODEL_SHORT[x]))}
            </tr>
          </thead>
          <tbody>
            {visible.map((r) => (
              <tr key={r.location_id} className="border-b">
                <td className="py-2 pr-4">
                  <Link to={`/forecast?location=${encodeURIComponent(r.location_id)}`} className="hover:underline">
                    {r.name}
                  </Link>
                </td>
                {models.map((x) => (
                  <td key={x} className={cn('num py-2 pr-4 text-right', x === rec ? 'font-medium' : 'text-muted-foreground')}>
                    {r.wape[x] === undefined ? '–' : formatPct(r.wape[x]!)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {rows.length > 10 && (
        <Button variant="outline" size="sm" className="mt-4" onClick={() => setAll((a) => !a)}>
          {all ? 'Show the 10 hardest stores' : `Show all ${rows.length} stores`}
        </Button>
      )}
    </section>
  )
}

function Training({ m }: { m: Metrics }) {
  const t = m.training
  if (!t) return null
  const history = t.history ?? []
  return (
    <section aria-labelledby="training">
      <h2 id="training" className="border-b pb-3 text-2xl font-semibold">
        How the ST-GNN was trained
      </h2>
      <div className="text-muted-foreground mt-3 max-w-[70ch] space-y-2 text-sm">
        <p>
          {t.best_epoch
            ? `The version from training pass ${t.best_epoch} was kept because it scored best on held-out days.`
            : 'The ST-GNN was trained on the store network’s sales history.'}
          {t.ensemble_weight_stgnn !== undefined &&
            ` Recommendations use a blend that gives the ST-GNN ${formatPct(t.ensemble_weight_stgnn, 0)} of the weight.`}
        </p>
        {t.note && <p>{t.note}</p>}
      </div>
      {history.length > 1 && (
        <figure className="mt-4">
          <p className="mb-2 text-sm font-medium">Error on held-out days after each training pass</p>
          <div className="h-[220px] w-full max-w-2xl">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history} margin={{ top: 8, right: 12, bottom: 0, left: 0 }}>
                <CartesianGrid vertical={false} stroke="var(--chart-grid)" />
                <XAxis dataKey="epoch" stroke="var(--chart-grid)" tick={{ fill: 'var(--chart-axis)', fontSize: 12 }} tickLine={false} />
                <YAxis
                  stroke="var(--chart-grid)"
                  tick={{ fill: 'var(--chart-axis)', fontSize: 12 }}
                  tickLine={false}
                  axisLine={false}
                  width={48}
                  tickFormatter={(v: number) => formatPct(v, 0)}
                  domain={['auto', 'auto']}
                />
                <Line dataKey="val_wape" stroke="var(--chart-3)" strokeWidth={2} dot={{ r: 3, fill: 'var(--chart-3)' }} isAnimationActive={false} />
                <Tooltip
                  cursor={{ stroke: 'var(--muted-foreground)', strokeWidth: 1 }}
                  content={({ active, payload, label }) =>
                    active && payload?.length ? (
                      <div className="bg-popover rounded-md border px-3 py-2 text-sm shadow-sm">
                        <p className="font-medium">Pass {label}</p>
                        <p className="num">Error {formatPct(Number(payload[0].value))}</p>
                      </div>
                    ) : null
                  }
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </figure>
      )}
    </section>
  )
}
