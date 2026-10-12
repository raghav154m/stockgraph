import path from 'node:path'
import tailwindcss from '@tailwindcss/vite'
import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// In development /api is proxied to the FastAPI server, so no CORS setup is needed.
// In production set VITE_API_URL to the deployed backend (see src/lib/api.ts).
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { '@': path.resolve(import.meta.dirname, './src') },
  },
  server: {
    proxy: {
      '/api': { target: process.env.STOCKGRAPH_API ?? 'http://127.0.0.1:8000', changeOrigin: true },
    },
  },
})
