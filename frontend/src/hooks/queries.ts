import { keepPreviousData, useQuery } from '@tanstack/react-query'
import { api } from '@/lib/api'

// Results only change when the pipeline is re-run, so cached data stays good for the session.
const stable = { staleTime: 5 * 60_000 }

export const useOverview = () => useQuery({ queryKey: ['overview'], queryFn: api.overview, ...stable })
export const useLocations = () => useQuery({ queryKey: ['locations'], queryFn: () => api.locations(), ...stable })
export const useLocation = (id: string | null) =>
  useQuery({ queryKey: ['location', id], queryFn: () => api.location(id!), enabled: !!id, ...stable })
export const useProducts = () => useQuery({ queryKey: ['products'], queryFn: api.products, ...stable })
export const useMetrics = () => useQuery({ queryKey: ['metrics'], queryFn: api.metrics, ...stable })

export const useRecommendations = (p: Parameters<typeof api.recommendations>[0]) =>
  useQuery({
    queryKey: ['recommendations', p],
    queryFn: () => api.recommendations(p),
    placeholderData: keepPreviousData,
    ...stable,
  })

export const useForecast = (location: string, product: string) =>
  useQuery({
    queryKey: ['forecast', location, product],
    queryFn: () => api.forecast(location, product),
    placeholderData: keepPreviousData,
    ...stable,
  })
