import L from 'leaflet'
import { Building2, Store } from 'lucide-react'
import { useEffect, useMemo } from 'react'
import { CircleMarker, MapContainer, Marker, Polyline, TileLayer, Tooltip, useMap } from 'react-leaflet'
import { Link, useSearchParams } from 'react-router-dom'
import { PageHeader } from '@/components/PageHeader'
import { ErrorState, LoadingBlock } from '@/components/States'
import { StatusBadge } from '@/components/StatusBadge'
import { StatusTape } from '@/components/StatusTape'
import { useResolvedTheme } from '@/lib/theme'
import { Button } from '@/components/ui/button'
import { useLocation as useLocationDetail, useLocations } from '@/hooks/queries'
import type { MapLocation, Status, StatusCounts } from '@/lib/api'
import { formatDays, formatUnits } from '@/lib/format'
import { STATUS, STATUS_ORDER } from '@/lib/status'
import { cn } from '@/lib/utils'

const TILES = {
  light: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
  dark: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
}
const ATTRIBUTION =
  '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors &copy; <a href="https://carto.com/attributions">CARTO</a>'

const isStatus = (v: string | null): v is Status => !!v && (STATUS_ORDER as string[]).includes(v)

/** A store's colour on the map: its most urgent status, or the filtered status when a filter is on. */
function markerStatus(loc: MapLocation, filter: Status | null): Status | null {
  if (!filter) return loc.worst_status
  return loc.status_counts[filter] > 0 ? filter : null
}

function warehouseIcon(selected: boolean) {
  return L.divIcon({
    className: '',
    html: `<div style="width:18px;height:18px;border-radius:3px;background:var(--foreground);border:${selected ? 3 : 2}px solid var(--card);box-shadow:0 0 0 ${selected ? 2 : 1}px var(--foreground)"></div>`,
    iconSize: [18, 18],
    iconAnchor: [9, 9],
  })
}

function FitBounds({ locations }: { locations: MapLocation[] }) {
  const map = useMap()
  useEffect(() => {
    if (!locations.length) return
    map.fitBounds(L.latLngBounds(locations.map((l) => [l.latitude, l.longitude] as [number, number])), {
      padding: [24, 24],
    })
  }, [locations, map])
  return null
}

export default function MapPage() {
  const locations = useLocations()
  const theme = useResolvedTheme()
  const [params, setParams] = useSearchParams()
  const filter = isStatus(params.get('status')) ? (params.get('status') as Status) : null
  const selectedId = params.get('location')
  const showLinks = params.get('links') === '1'

  const update = (patch: Record<string, string | null>) => {
    const next = new URLSearchParams(params)
    for (const [k, v] of Object.entries(patch)) {
      if (v === null) next.delete(k)
      else next.set(k, v)
    }
    setParams(next, { replace: true })
  }

  const data = useMemo(() => locations.data ?? [], [locations.data])
  const stores = data.filter((l) => l.type === 'store')
  const warehouses = data.filter((l) => l.type === 'warehouse')
  const byId = useMemo(() => Object.fromEntries(data.map((l) => [l.location_id, l])), [data])

  const storeCounts = useMemo(() => {
    const c = Object.fromEntries(STATUS_ORDER.map((s) => [s, 0])) as StatusCounts
    for (const s of stores) for (const k of STATUS_ORDER) c[k] += s.status_counts[k]
    return c
  }, [stores])

  // urgent stores drawn last so they sit on top
  const drawOrder = [...stores].sort(
    (a, b) => STATUS_ORDER.indexOf(b.worst_status) - STATUS_ORDER.indexOf(a.worst_status),
  )

  const matching = filter ? stores.filter((s) => s.status_counts[filter] > 0).length : stores.length

  return (
    <>
      <PageHeader
        title="Map"
        actions={
          <Button variant="outline" size="sm" onClick={() => update({ links: showLinks ? null : '1' })} aria-pressed={showLinks}>
            {showLinks ? 'Hide supply routes' : 'Show supply routes'}
          </Button>
        }
      >
        Each dot is a store, coloured by its most urgent stock problem. Squares are warehouses.
      </PageHeader>

      {locations.isError ? (
        <ErrorState error={locations.error} onRetry={() => locations.refetch()} />
      ) : !locations.data ? (
        <LoadingBlock className="h-[60vh]" />
      ) : (
        <>
          <div className="mb-4">
            <StatusTape
              counts={storeCounts}
              size="sm"
              selected={filter}
              onSelect={(s) => update({ status: filter === s ? null : s })}
            />
            <p className="text-muted-foreground mt-3 text-sm" aria-live="polite">
              {filter
                ? `${matching} of ${stores.length} stores have at least one item with ${STATUS[filter].label.toLowerCase()}. `
                : `Store items across ${stores.length} stores. Choose a status to show only the stores that have it. `}
              {filter && (
                <button type="button" className="text-primary underline-offset-4 hover:underline" onClick={() => update({ status: null })}>
                  Show all stores
                </button>
              )}
            </p>
          </div>

          <div className="grid gap-6 lg:grid-cols-[1fr_360px]">
            <div className="h-[60vh] min-h-[380px] overflow-hidden rounded-lg border lg:h-[640px]">
              <MapContainer center={[17.9, 79.0]} zoom={7} scrollWheelZoom className="h-full w-full">
                <TileLayer key={theme} url={TILES[theme]} attribution={ATTRIBUTION} subdomains="abcd" maxZoom={19} />
                <FitBounds locations={data} />
                {showLinks &&
                  stores.map((s) => {
                    const wh = byId[s.supplied_by]
                    if (!wh) return null
                    return (
                      <Polyline
                        key={`l-${s.location_id}`}
                        positions={[
                          [s.latitude, s.longitude],
                          [wh.latitude, wh.longitude],
                        ]}
                        pathOptions={{ color: 'var(--muted-foreground)', weight: 1, opacity: 0.45 }}
                      />
                    )
                  })}
                {drawOrder.map((s) => {
                  const st = markerStatus(s, filter)
                  const selected = s.location_id === selectedId
                  return (
                    <CircleMarker
                      key={s.location_id}
                      center={[s.latitude, s.longitude]}
                      radius={selected ? 10 : 7}
                      pathOptions={{
                        color: selected ? 'var(--foreground)' : 'var(--card)',
                        weight: selected ? 3 : 2,
                        fillColor: st ? STATUS[st].fill : 'var(--muted-foreground)',
                        fillOpacity: st ? 0.95 : 0.25,
                        opacity: st ? 1 : 0.5,
                      }}
                      eventHandlers={{ click: () => update({ location: s.location_id }) }}
                    >
                      <Tooltip direction="top" offset={[0, -6]}>
                        <span className="font-medium">{s.name}</span>
                        <br />
                        {STATUS[s.worst_status].label}
                        {s.high_priority_recommendations > 0 && `, ${s.high_priority_recommendations} urgent`}
                      </Tooltip>
                    </CircleMarker>
                  )
                })}
                {warehouses.map((w) => (
                  <Marker
                    key={w.location_id}
                    position={[w.latitude, w.longitude]}
                    icon={warehouseIcon(w.location_id === selectedId)}
                    eventHandlers={{ click: () => update({ location: w.location_id }) }}
                    zIndexOffset={1000}
                  >
                    <Tooltip direction="top" offset={[0, -8]}>
                      <span className="font-medium">{w.name}</span>
                      <br />
                      Warehouse
                    </Tooltip>
                  </Marker>
                ))}
              </MapContainer>
            </div>

            <LocationPanel
              id={selectedId}
              summary={selectedId ? byId[selectedId] : undefined}
              warehouseName={(id) => byId[id]?.name ?? id}
              onClose={() => update({ location: null })}
            />
          </div>

          <LocationList
            locations={[...stores, ...warehouses]}
            filter={filter}
            selectedId={selectedId}
            onSelect={(id) => {
              update({ location: id })
              window.scrollTo({ top: 0, behavior: 'smooth' })
            }}
          />
        </>
      )}
    </>
  )
}

function LocationPanel({
  id,
  summary,
  warehouseName,
  onClose,
}: {
  id: string | null
  summary?: MapLocation
  warehouseName: (id: string) => string
  onClose: () => void
}) {
  const detail = useLocationDetail(id)

  if (!id || !summary) {
    return (
      <aside className="bg-card flex flex-col justify-center rounded-lg border p-6 text-sm">
        <p className="font-medium">Select a store or warehouse</p>
        <p className="text-muted-foreground mt-1">
          Click a dot or a square on the map, or choose one from the list below, to see which products need attention there.
        </p>
      </aside>
    )
  }

  const isStore = summary.type === 'store'
  const Icon = isStore ? Store : Building2
  const problems = detail.data?.inventory.filter((i) => i.status !== 'healthy') ?? []

  return (
    <aside className="bg-card flex max-h-[640px] flex-col overflow-hidden rounded-lg border" aria-label={`${summary.name} details`}>
      <div className="border-b p-5">
        <div className="flex items-start justify-between gap-3">
          <div>
            <p className="text-muted-foreground flex items-center gap-1.5 text-sm">
              <Icon className="size-4" aria-hidden />
              {isStore ? `Store in ${summary.city}` : `Warehouse in ${summary.city}`}
            </p>
            <h2 className="mt-1 text-xl font-semibold">{summary.name}</h2>
            <p className="text-muted-foreground mt-1 text-sm">
              {isStore ? `Supplied by ${warehouseName(summary.supplied_by)}` : 'Supplied directly by suppliers'}
            </p>
          </div>
          <Button variant="ghost" size="sm" onClick={onClose}>
            Close
          </Button>
        </div>
        <div className="mt-4">
          <StatusTape counts={summary.status_counts} size="sm" columns={2} />
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        {detail.isError ? (
          <ErrorState error={detail.error} onRetry={() => detail.refetch()} />
        ) : !detail.data ? (
          <LoadingBlock className="h-48" />
        ) : problems.length === 0 ? (
          <p className="text-muted-foreground text-sm">Every product here has healthy stock.</p>
        ) : (
          <>
            <h3 className="font-sans text-sm font-medium tracking-normal">
              {problems.length} {problems.length === 1 ? 'product needs' : 'products need'} attention
            </h3>
            <ul className="mt-2 divide-y">
              {problems.map((i) => (
                <li key={i.product_id} className="py-3">
                  <div className="flex items-start justify-between gap-3">
                    <span className="font-medium">{i.product_name}</span>
                    <StatusBadge status={i.status} className="text-xs" />
                  </div>
                  <p className="text-muted-foreground num mt-1 text-sm">
                    {formatUnits(i.on_hand)} on hand
                    {i.on_order > 0 && `, ${formatUnits(i.on_order)} on order`}
                    {i.days_of_cover !== null && `, lasts about ${formatDays(i.days_of_cover)}`}
                  </p>
                  {isStore && (
                    <Link
                      to={`/forecast?location=${encodeURIComponent(i.location_id)}&product=${encodeURIComponent(i.product_id)}`}
                      className="text-primary mt-1 inline-block text-sm underline-offset-4 hover:underline"
                    >
                      See forecast
                    </Link>
                  )}
                </li>
              ))}
            </ul>
          </>
        )}
      </div>
    </aside>
  )
}

function LocationList({
  locations,
  filter,
  selectedId,
  onSelect,
}: {
  locations: MapLocation[]
  filter: Status | null
  selectedId: string | null
  onSelect: (id: string) => void
}) {
  const rows = locations
    .filter((l) => !filter || l.status_counts[filter] > 0)
    .sort(
      (a, b) =>
        STATUS_ORDER.indexOf(a.worst_status) - STATUS_ORDER.indexOf(b.worst_status) ||
        b.status_counts.stockout_risk - a.status_counts.stockout_risk ||
        a.name.localeCompare(b.name),
    )

  return (
    <section className="mt-10" aria-labelledby="loc-list">
      <h2 id="loc-list" className="border-b pb-3 text-2xl font-semibold">
        {filter ? `Stores with ${STATUS[filter].label.toLowerCase()}` : 'All locations'}
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-muted-foreground border-b text-left">
              <th className="py-2 pr-4 font-medium">Location</th>
              <th className="py-2 pr-4 font-medium">Most urgent</th>
              {STATUS_ORDER.slice(0, 3).map((s) => (
                <th key={s} className="py-2 pr-4 text-right font-medium">
                  {STATUS[s].label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((l) => (
              <tr key={l.location_id} className={cn('border-b', l.location_id === selectedId && 'bg-accent')}>
                <td className="py-2 pr-4">
                  <button type="button" className="text-left font-medium hover:underline" onClick={() => onSelect(l.location_id)}>
                    {l.name}
                  </button>
                  <span className="text-muted-foreground block text-xs">
                    {l.type === 'store' ? l.city : `Warehouse, ${l.city}`}
                  </span>
                </td>
                <td className="py-2 pr-4">
                  <StatusBadge status={l.worst_status} />
                </td>
                {STATUS_ORDER.slice(0, 3).map((s) => (
                  <td key={s} className={cn('num py-2 pr-4 text-right', !l.status_counts[s] && 'text-muted-foreground')}>
                    {l.status_counts[s] || '–'}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  )
}
