import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { lazy, Suspense } from 'react'
import { BrowserRouter, Route, Routes } from 'react-router-dom'
import { Layout } from '@/components/Layout'
import { LoadingBlock } from '@/components/States'
import { OverviewPage } from '@/pages/Overview'
import { RecommendationsPage } from '@/pages/Recommendations'
import { AboutPage } from '@/pages/About'
import { NotFoundPage } from '@/pages/NotFound'

// Leaflet and Recharts are the heaviest dependencies; load them only on the pages that use them.
const MapPage = lazy(() => import('@/pages/MapPage'))
const ForecastPage = lazy(() => import('@/pages/Forecast').then((m) => ({ default: m.ForecastPage })))
const AccuracyPage = lazy(() => import('@/pages/Accuracy').then((m) => ({ default: m.AccuracyPage })))

const page = (el: React.ReactNode) => <Suspense fallback={<LoadingBlock className="h-[60vh]" />}>{el}</Suspense>

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: 1, refetchOnWindowFocus: false } },
})

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          <Route element={<Layout />}>
            <Route index element={<OverviewPage />} />
            <Route
              path="map"
              element={page(<MapPage />)}
            />
            <Route path="recommendations" element={<RecommendationsPage />} />
            <Route path="forecast" element={page(<ForecastPage />)} />
            <Route path="accuracy" element={page(<AccuracyPage />)} />
            <Route path="about" element={<AboutPage />} />
            <Route path="*" element={<NotFoundPage />} />
          </Route>
        </Routes>
      </BrowserRouter>
    </QueryClientProvider>
  )
}
