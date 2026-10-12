import { AlertTriangle } from 'lucide-react'
import { ApiError } from '@/lib/api'
import { Button } from '@/components/ui/button'
import { Skeleton } from '@/components/ui/skeleton'

export function ErrorState({ error, onRetry }: { error: unknown; onRetry?: () => void }) {
  const message =
    error instanceof ApiError
      ? error.status === 0
        ? error.message
        : `The server answered with an error (${error.status}): ${error.message}`
      : 'Something went wrong while loading this page.'
  return (
    <div role="alert" className="bg-card flex flex-col items-start gap-3 rounded-lg border p-5">
      <div className="flex items-center gap-2 font-medium">
        <AlertTriangle className="text-destructive size-5" aria-hidden />
        Couldn’t load this data
      </div>
      <p className="text-muted-foreground max-w-prose text-sm">{message}</p>
      {onRetry && (
        <Button variant="outline" size="sm" onClick={onRetry}>
          Try again
        </Button>
      )}
    </div>
  )
}

export function LoadingBlock({ className = 'h-40' }: { className?: string }) {
  return <Skeleton className={`w-full ${className}`} aria-label="Loading" />
}
