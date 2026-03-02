import { defineConfig } from 'vite'

export default defineConfig({
  // Relative base avoids hard-coding a deployment subpath.
  // This keeps asset URLs valid on both root domains and project subpaths.
  base: './',
  build: { target: 'es2020' },
})
