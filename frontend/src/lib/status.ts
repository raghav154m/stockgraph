import { AlertOctagon, Archive, CheckCircle2, Clock, type LucideIcon } from 'lucide-react'
import type { ModelName, Status } from './api'

export interface StatusInfo {
  key: Status
  label: string
  /** one-line meaning for a manager */
  meaning: string
  icon: LucideIcon
  /** CSS colour for fills */
  fill: string
  /** quieter fill for large areas; only healthy differs, so problems carry the colour */
  areaFill: string
  /** Tailwind classes for text-safe ink */
  ink: string
  bg: string
}

export const STATUS: Record<Status, StatusInfo> = {
  stockout_risk: {
    key: 'stockout_risk',
    label: 'Stockout risk',
    meaning: 'Will run out before new stock can arrive',
    icon: AlertOctagon,
    fill: 'var(--status-stockout)',
    areaFill: 'var(--status-stockout)',
    ink: 'text-stockout-ink',
    bg: 'bg-stockout',
  },
  reorder_due: {
    key: 'reorder_due',
    label: 'Reorder due',
    meaning: 'Below the reorder point; order now to stay covered',
    icon: Clock,
    fill: 'var(--status-reorder)',
    areaFill: 'var(--status-reorder)',
    ink: 'text-reorder-ink',
    bg: 'bg-reorder',
  },
  overstock: {
    key: 'overstock',
    label: 'Overstock',
    meaning: 'More than 5 weeks of stock; money sitting idle',
    icon: Archive,
    fill: 'var(--status-overstock)',
    areaFill: 'var(--status-overstock)',
    ink: 'text-overstock-ink',
    bg: 'bg-overstock',
  },
  healthy: {
    key: 'healthy',
    label: 'Healthy',
    meaning: 'Enough stock, not too much',
    icon: CheckCircle2,
    fill: 'var(--status-healthy)',
    areaFill: 'var(--status-healthy-soft)',
    ink: 'text-healthy-ink',
    bg: 'bg-healthy',
  },
}

/** Most urgent first, the order the API sorts by. */
export const STATUS_ORDER: Status[] = ['stockout_risk', 'reorder_due', 'overstock', 'healthy']

export const MODEL_LABEL: Record<ModelName, string> = {
  ensemble: 'ST-GNN + gradient boosting',
  baseline: 'Gradient boosting',
  stgnn: 'ST-GNN',
  naive: 'Same weekday, last 4 weeks',
}

export const MODEL_SHORT: Record<ModelName, string> = {
  ensemble: 'Blend',
  baseline: 'Gradient boosting',
  stgnn: 'ST-GNN',
  naive: 'Naive',
}

/** Fixed series colour per model: colour follows the model, never its rank. */
export const MODEL_COLOR: Record<ModelName, string> = {
  ensemble: 'var(--chart-1)',
  baseline: 'var(--chart-2)',
  stgnn: 'var(--chart-3)',
  naive: 'var(--chart-4)',
}
