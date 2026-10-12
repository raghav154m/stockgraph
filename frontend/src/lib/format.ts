/** Formatting for a reader in India: ₹ with lakh/crore grouping, day-month dates. */

const inr = new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 })
const num = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 0 })
const num1 = new Intl.NumberFormat('en-IN', { maximumFractionDigits: 1 })

export const formatINR = (v: number) => inr.format(v)

/** ₹4.47 lakh, ₹10.06 crore: how an Indian manager says large sums. */
export function formatINRShort(v: number): string {
  const a = Math.abs(v)
  if (a >= 1e7) return `₹${num1.format(v / 1e7)} crore`
  if (a >= 1e5) return `₹${num1.format(v / 1e5)} lakh`
  return inr.format(v)
}

export const formatUnits = (v: number) => num.format(v)
export const formatUnits1 = (v: number) => num1.format(v)
export const formatPct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`

function parseDate(iso: string) {
  const [y, m, d] = iso.split('-').map(Number)
  return new Date(y, m - 1, d)
}

export const formatDate = (iso: string) =>
  parseDate(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short', year: 'numeric' })
export const formatDateShort = (iso: string) =>
  parseDate(iso).toLocaleDateString('en-IN', { day: 'numeric', month: 'short' })

export function daysBetween(fromIso: string, toIso: string) {
  return Math.round((parseDate(toIso).getTime() - parseDate(fromIso).getTime()) / 86_400_000)
}

export function addDays(iso: string, days: number) {
  const d = parseDate(iso)
  d.setDate(d.getDate() + days)
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`
}

/** 'Due today', 'Due tomorrow', 'Due in 3 days', relative to the data's as-of date. */
export function dueText(deadline: string, asOf: string) {
  const d = daysBetween(asOf, deadline)
  if (d <= 0) return 'Due today'
  if (d === 1) return 'Due tomorrow'
  return `Due in ${d} days`
}

/** '1 day', '36.6 days' */
export const formatDays = (v: number) => {
  const s = num1.format(v)
  return `${s} ${s === '1' ? 'day' : 'days'}`
}
