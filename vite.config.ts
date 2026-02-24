import { defineConfig } from 'vite'

export default defineConfig({
  base: process.env.GH_PAGES_BASE || '/',
  build: { target: 'es2020' },
})
