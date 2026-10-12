import { useSearchParams } from 'react-router-dom'
import { PageHeader } from '@/components/PageHeader'
import { RecommendationRow } from '@/components/RecommendationRow'
import { ErrorState, LoadingBlock } from '@/components/States'
import { Button } from '@/components/ui/button'
import { Select, SelectContent, SelectGroup, SelectItem, SelectLabel, SelectTrigger, SelectValue } from '@/components/ui/select'
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { useLocations, useOverview, useProducts, useRecommendations } from '@/hooks/queries'
import { formatINR, formatUnits } from '@/lib/format'

const PAGE_SIZE = 20
const ALL = 'all'

export function RecommendationsPage() {
  const [params, setParams] = useSearchParams()
  const type = params.get('type') ?? ALL
  const priority = params.get('priority') ?? ALL
  const location = params.get('location') ?? ALL
  const product = params.get('product') ?? ALL
  const page = Math.max(1, Number(params.get('page') ?? 1) || 1)

  const set = (key: string, value: string) => {
    const next = new URLSearchParams(params)
    if (value === ALL || (key === 'page' && value === '1')) next.delete(key)
    else next.set(key, value)
    if (key !== 'page') next.delete('page')
    setParams(next, { replace: key !== 'page' })
  }

  const overview = useOverview()
  const locations = useLocations()
  const products = useProducts()
  const recs = useRecommendations({
    type: type === ALL ? undefined : type,
    priority: priority === ALL ? undefined : priority,
    location_id: location === ALL ? undefined : location,
    product_id: product === ALL ? undefined : product,
    limit: PAGE_SIZE,
    offset: (page - 1) * PAGE_SIZE,
  })

  const filtered = type !== ALL || priority !== ALL || location !== ALL || product !== ALL
  const pages = recs.data ? Math.max(1, Math.ceil(recs.data.total / PAGE_SIZE)) : 1
  const asOf = overview.data?.as_of

  const stores = (locations.data ?? []).filter((l) => l.type === 'store').sort((a, b) => a.name.localeCompare(b.name))
  const warehouses = (locations.data ?? []).filter((l) => l.type === 'warehouse')

  return (
    <>
      <PageHeader title="Recommendations">
        Transfers move spare stock from a nearby store before anything new is ordered. Orders top up what transfers
        can’t cover. Most urgent first.
      </PageHeader>

      <div className="flex flex-col gap-3 md:flex-row md:flex-wrap md:items-center">
        <Tabs value={type} onValueChange={(v) => set('type', v)}>
          <TabsList>
            <TabsTrigger value={ALL}>All</TabsTrigger>
            <TabsTrigger value="transfer">Transfers</TabsTrigger>
            <TabsTrigger value="reorder">Orders</TabsTrigger>
          </TabsList>
        </Tabs>

        <Select value={priority} onValueChange={(v) => set('priority', v)}>
          <SelectTrigger className="w-full md:w-44" aria-label="Priority">
            <SelectValue />
          </SelectTrigger>
          <SelectContent>
            <SelectItem value={ALL}>Any priority</SelectItem>
            <SelectItem value="high">High priority</SelectItem>
            <SelectItem value="medium">Medium priority</SelectItem>
            <SelectItem value="low">Low priority</SelectItem>
          </SelectContent>
        </Select>

        <Select value={location} onValueChange={(v) => set('location', v)}>
          <SelectTrigger className="w-full md:w-64" aria-label="Store or warehouse">
            <SelectValue placeholder="Any location" />
          </SelectTrigger>
          <SelectContent className="max-h-80">
            <SelectItem value={ALL}>Any location</SelectItem>
            <SelectGroup>
              <SelectLabel>Warehouses</SelectLabel>
              {warehouses.map((l) => (
                <SelectItem key={l.location_id} value={l.location_id}>
                  {l.name}
                </SelectItem>
              ))}
            </SelectGroup>
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

        <Select value={product} onValueChange={(v) => set('product', v)}>
          <SelectTrigger className="w-full md:w-56" aria-label="Product">
            <SelectValue placeholder="Any product" />
          </SelectTrigger>
          <SelectContent className="max-h-80">
            <SelectItem value={ALL}>Any product</SelectItem>
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

        {filtered && (
          <Button variant="ghost" size="sm" onClick={() => setParams(new URLSearchParams(), { replace: true })}>
            Clear filters
          </Button>
        )}
      </div>

      {recs.isError ? (
        <div className="mt-6">
          <ErrorState error={recs.error} onRetry={() => recs.refetch()} />
        </div>
      ) : !recs.data || !asOf ? (
        <LoadingBlock className="mt-6 h-96" />
      ) : (
        <>
          <p className="text-muted-foreground mt-6 border-b pb-3 text-sm" aria-live="polite">
            <span className="text-foreground num font-medium">{formatUnits(recs.data.total)}</span>{' '}
            {recs.data.total === 1 ? 'recommendation' : 'recommendations'} worth{' '}
            <span className="text-foreground num font-medium">{formatINR(recs.data.total_value_inr)}</span>
          </p>
          {recs.data.items.length === 0 ? (
            <div className="py-10">
              <p className="font-medium">Nothing matches these filters.</p>
              <p className="text-muted-foreground mt-1 text-sm">Try a different priority or location, or clear the filters.</p>
            </div>
          ) : (
            <div className={recs.isPlaceholderData ? 'divide-y opacity-60' : 'divide-y'}>
              {recs.data.items.map((r) => (
                <RecommendationRow key={r.rec_id} rec={r} asOf={asOf} />
              ))}
            </div>
          )}
          {pages > 1 && (
            <nav className="mt-4 flex items-center justify-between border-t pt-4" aria-label="Pages">
              <Button variant="outline" size="sm" disabled={page <= 1} onClick={() => set('page', String(page - 1))}>
                Previous
              </Button>
              <span className="text-muted-foreground num text-sm">
                Page {page} of {pages}
              </span>
              <Button variant="outline" size="sm" disabled={page >= pages} onClick={() => set('page', String(page + 1))}>
                Next
              </Button>
            </nav>
          )}
        </>
      )}
    </>
  )
}
