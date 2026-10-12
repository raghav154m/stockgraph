import { Link } from 'react-router-dom'
import { Button } from '@/components/ui/button'

export function NotFoundPage() {
  return (
    <div className="py-16">
      <h1 className="text-3xl font-semibold">This page doesn’t exist</h1>
      <p className="text-muted-foreground mt-2">The link may be out of date.</p>
      <Button asChild className="mt-6">
        <Link to="/">Go to the overview</Link>
      </Button>
    </div>
  )
}
