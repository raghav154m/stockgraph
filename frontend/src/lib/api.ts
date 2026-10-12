/**
 * Typed client for the StockGraph read API (backend/api.py).
 * In development Vite proxies /api to the FastAPI server; in production set VITE_API_URL.
 */

export const API_BASE = (import.meta.env.VITE_API_URL ?? '').replace(/\/$/, '')
export const DATASET = 'demo'

export type Status = 'stockout_risk' | 'reorder_due' | 'overstock' | 'healthy'
export type ModelName = 'baseline' | 'stgnn' | 'ensemble' | 'naive'
export type StatusCounts = Record<Status, number>

export interface Overview {
  id: string
  name: string
  source: string
  as_of: string
  forecast_model: ModelName
  loaded_at: string
  stores: number
  warehouses: number
  products: number
  summary: {
    as_of: string
    forecast_model: ModelName
    service_level: number
    wape: Partial<Record<ModelName, number>>
    store_items: number
    status_counts: StatusCounts
    overstock_value_inr: number
    recommendations: { transfer: number; reorder: number }
    high_priority: number
    reorder_value_inr: number
  }
  meta: { start?: string; days?: number; series?: number; currency?: string; [k: string]: unknown }
  disclosure: string | null
}

export interface Location {
  location_id: string
  name: string
  type: 'store' | 'warehouse'
  city: string
  region: string
  latitude: number
  longitude: number
  supplied_by: string
}

export interface MapLocation extends Location {
  status_counts: StatusCounts
  worst_status: Status
  high_priority_recommendations: number
}

export interface InventoryItem {
  location_id: string
  product_id: string
  supplied_by: string
  on_hand: number
  on_order: number
  avg_daily_forecast: number
  days_of_cover: number | null
  lead_time_days: number
  safety_stock: number
  reorder_point: number
  target_level: number
  status: Status
}

export interface Accuracy {
  model: ModelName
  mae: number
  wape: number
  bias: number
}

export interface LocationDetail extends Location {
  inventory: (InventoryItem & { product_name: string })[]
  accuracy: Accuracy[]
}

export interface Product {
  product_id: string
  name: string
  category: string
  unit_cost: number
  lead_time_days: number
  store_on_hand: number
  stores_at_risk: number
  stores_overstocked: number
}

export interface Recommendation {
  rec_id: string
  type: 'transfer' | 'reorder'
  product_id: string
  product_name: string
  from_location: string
  from_name: string
  to_location: string
  to_name: string
  quantity: number
  deadline: string
  priority: 'high' | 'medium' | 'low'
  value_inr: number
  reason: string
}

export interface RecommendationPage {
  total: number
  total_value_inr: number
  items: Recommendation[]
}

export interface ForecastPoint {
  date: string
  p50: number
  p10: number
  p90: number
}

export interface BacktestPoint {
  date: string
  actual: number | null
  baseline: number | null
  naive: number | null
  stgnn: number | null
  ensemble: number | null
}

export interface Forecast {
  location_id: string
  product_id: string
  product_name: string
  as_of: string
  recommendation_model: ModelName
  history: { date: string; units_sold: number; on_promotion: number }[]
  forecasts: Partial<Record<ModelName, ForecastPoint[]>>
  backtest: BacktestPoint[]
  inventory: InventoryItem | null
}

export interface Metrics {
  forecast_model: ModelName
  overall: Accuracy[]
  by_store: { location_id: string; name: string; wape: Partial<Record<ModelName, number>> }[]
  rolling: { holdout_start: string; model: ModelName; mae: number; wape: number; bias: number }[]
  training: {
    settings?: Record<string, unknown>
    best_epoch?: number
    history?: { epoch: number; loss: number; val_wape: number }[]
    rolling_wape?: Partial<Record<ModelName, number>>
    holdout_wape?: Partial<Record<ModelName, number>>
    best_model?: ModelName
    recommendation_model?: ModelName
    selection_basis?: string
    ensemble_weight_stgnn?: number
    note?: string
  } | null
}

export class ApiError extends Error {
  status: number
  constructor(status: number, message: string) {
    super(message)
    this.status = status
  }
}

type Params = Record<string, string | number | undefined | null>

async function get<T>(path: string, params: Params = {}): Promise<T> {
  const qs = new URLSearchParams()
  for (const [k, v] of Object.entries(params)) {
    if (v !== undefined && v !== null && v !== '') qs.set(k, String(v))
  }
  const url = `${API_BASE}/api${path}${qs.size ? `?${qs}` : ''}`
  let res: Response
  try {
    res = await fetch(url)
  } catch {
    throw new ApiError(0, 'The StockGraph server could not be reached. Check that the API is running.')
  }
  if (!res.ok) {
    let detail = res.statusText
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* not JSON */
    }
    throw new ApiError(res.status, detail)
  }
  return res.json() as Promise<T>
}

const ds = (path = '') => `/datasets/${DATASET}${path}`

export const api = {
  overview: () => get<Overview>(ds()),
  locations: (type?: 'store' | 'warehouse') => get<MapLocation[]>(ds('/locations'), { type }),
  location: (id: string) => get<LocationDetail>(ds(`/locations/${encodeURIComponent(id)}`)),
  products: () => get<Product[]>(ds('/products')),
  recommendations: (p: {
    type?: string
    priority?: string
    location_id?: string
    product_id?: string
    limit?: number
    offset?: number
  }) => get<RecommendationPage>(ds('/recommendations'), p),
  forecast: (location_id: string, product_id: string, history_days = 56) =>
    get<Forecast>(ds('/forecast'), { location_id, product_id, history_days }),
  metrics: () => get<Metrics>(ds('/metrics')),
}
