import { PageHeader } from '@/components/PageHeader'
import { useOverview } from '@/hooks/queries'
import { formatDate, formatPct, formatUnits } from '@/lib/format'
import { MODEL_LABEL, STATUS, STATUS_ORDER } from '@/lib/status'
import type { ModelName } from '@/lib/api'

// Shown even if the API is down: the disclosure must always be on this page.
const FALLBACK_DISCLOSURE =
  'Demo data is adapted from the Favorita grocery sales dataset (Corporación Favorita, Ecuador). Store locations are relabelled to Telangana, prices are shown in ₹, and the warehouse and inventory layer is simulated. Sales patterns, including holidays, come from the original data.'

const REPO = 'https://github.com/raghav154m/stockgraph'

export function AboutPage() {
  const { data } = useOverview()
  const disclosure = data?.disclosure ?? FALLBACK_DISCLOSURE
  const wape = data?.summary.wape ?? {}
  const models = (['baseline', 'ensemble', 'stgnn', 'naive'] as ModelName[]).filter((m) => wape[m] !== undefined)

  return (
    <article className="max-w-3xl">
      <PageHeader title="About the data">
        StockGraph forecasts demand across a network of stores and warehouses and turns the forecast into transfers and
        orders. It is a portfolio project, built to show the whole path from raw sales to a decision.
      </PageHeader>

      <section aria-labelledby="disclosure" className="border-overstock bg-card rounded-lg border-l-4 p-5">
        <h2 id="disclosure" className="text-xl font-semibold">
          What’s real and what isn’t
        </h2>
        <p className="mt-2">{disclosure}</p>
        <ul className="text-muted-foreground mt-3 list-disc space-y-1 pl-5 text-sm">
          <li>
            <span className="text-foreground">Real:</span> daily sales by store and product family, promotions, holidays and
            the store network’s shape (which stores share a city).
          </li>
          <li>
            <span className="text-foreground">Relabelled:</span> each original city is mapped to a Telangana city, largest to
            largest, so stores that shared a city still do. Names and map positions are approximate.
          </li>
          <li>
            <span className="text-foreground">Simulated:</span> the three warehouses, stock on hand, stock on order, unit
            costs (set in ₹, as the original data has no prices) and delivery times. No real stock records were used.
          </li>
        </ul>
        {data && (
          <p className="text-muted-foreground mt-3 text-sm">
            This copy covers {data.stores} stores, {data.warehouses} warehouses and {data.products} product families
            {data.meta.days ? `, ${formatUnits(Number(data.meta.days))} days of sales` : ''}, up to {formatDate(data.as_of)}.
          </p>
        )}
      </section>

      <section aria-labelledby="how" className="mt-12">
        <h2 id="how" className="border-b pb-3 text-2xl font-semibold">
          How a recommendation is made
        </h2>
        <ol className="mt-4 space-y-5">
          <Step n={1} title="Forecast demand">
            Each store’s next 28 days of sales are forecast for every product family. A graph neural network (ST-GNN)
            treats stores as a connected network, so a store learns from nearby and similar stores as well as its own
            history. Its forecast is blended 50/50 with a gradient-boosted model, and every forecast comes with a range.
          </Step>
          <Step n={2} title="Check the stock">
            The forecast and its range set a safety stock, a reorder point and an order-up-to level for each store item.
            Comparing those with what’s on hand gives each item a status:
            <ul className="mt-3 space-y-2">
              {STATUS_ORDER.map((s) => {
                const Icon = STATUS[s].icon
                return (
                  <li key={s} className="flex gap-2">
                    <Icon className={`mt-0.5 size-4 shrink-0 ${STATUS[s].ink}`} aria-hidden />
                    <span>
                      <span className="text-foreground font-medium">{STATUS[s].label}:</span> {STATUS[s].meaning.toLowerCase()}.
                    </span>
                  </li>
                )
              })}
            </ul>
          </Step>
          <Step n={3} title="Move before you buy">
            A store that’s running short first takes spare stock from the nearest overstocked store. Whatever transfers
            can’t cover becomes an order from its warehouse, and warehouses order from suppliers. Each recommendation
            gives a deadline, a value in ₹ and a plain reason.
          </Step>
        </ol>
      </section>

      {models.length > 0 && (
        <section aria-labelledby="models" className="mt-12">
          <h2 id="models" className="border-b pb-3 text-2xl font-semibold">
            The models in this copy
          </h2>
          <table className="mt-2 w-full text-sm">
            <thead>
              <tr className="text-muted-foreground border-b text-left">
                <th className="py-2 pr-4 font-medium">Model</th>
                <th className="py-2 text-right font-medium">Error on the last 28 days</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m) => (
                <tr key={m} className="border-b">
                  <td className="py-2 pr-4">
                    {MODEL_LABEL[m]}
                    {m === data?.forecast_model && <span className="text-primary ml-2 font-medium">Used</span>}
                  </td>
                  <td className="num py-2 text-right">{formatPct(wape[m]!)}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="text-muted-foreground mt-3 text-sm">
            Error is the total miss as a share of what sold (WAPE). On the full dataset, scored over four back-to-back
            28-day periods, gradient boosting reaches 13.9%, the blend 14.0%, the ST-GNN 14.8% and the naive benchmark
            16.2%.
          </p>
        </section>
      )}

      <section aria-labelledby="built" className="mt-12">
        <h2 id="built" className="border-b pb-3 text-2xl font-semibold">
          How it’s built
        </h2>
        <p className="mt-4">
          Python, pandas, scikit-learn and PyTorch for the models; FastAPI and SQLite for the API; React, TypeScript,
          Tailwind, shadcn/ui, Recharts and Leaflet for this dashboard. Map tiles from OpenStreetMap.
        </p>
        <p className="mt-3">
          Sales data: Corporación Favorita, from Kaggle’s{' '}
          <a
            className="text-primary underline-offset-4 hover:underline"
            href="https://www.kaggle.com/competitions/store-sales-time-series-forecasting"
            target="_blank"
            rel="noreferrer"
          >
            Store Sales – Time Series Forecasting
          </a>{' '}
          competition. The raw data isn’t redistributed.
        </p>
        <p className="mt-3">
          <a className="text-primary underline-offset-4 hover:underline" href={REPO} target="_blank" rel="noreferrer">
            Source code on GitHub
          </a>
        </p>
      </section>
    </article>
  )
}

function Step({ n, title, children }: { n: number; title: string; children: React.ReactNode }) {
  return (
    <li className="grid grid-cols-[2rem_1fr] gap-x-3">
      <span className="bg-accent text-accent-foreground num flex size-8 items-center justify-center rounded-full text-sm font-medium" aria-hidden>
        {n}
      </span>
      <div>
        <h3 className="font-sans text-base font-medium tracking-normal">{title}</h3>
        <div className="text-muted-foreground mt-1">{children}</div>
      </div>
    </li>
  )
}
