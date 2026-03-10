# AGENTS.md

## Default Completion Workflow

For every completed code/edit task:

1. Run the relevant validation commands for the change.
2. Commit the finished changes with a clear, descriptive commit message.
3. Push the current branch to the configured remote.

Assume this workflow by default unless the user explicitly asks not to commit/push.
If push fails, report the exact error and required next step.

## Repo Workflow And Command Notes

Use these commands/workflows for this repository unless a task explicitly requires something else:

- Install dependencies:
  - `npm ci` for clean/reproducible installs (CI uses this).
  - `npm install` for local setup when lockfile-strict reproducibility is not required.
- Runtime preparation:
  - `npm run prepare:opencv` copies `opencv.js` into `public/opencv/`.
  - `npm run prepare:maxflow` builds/refreshes maxflow wasm artifacts in `public/wasm/maxflow/`.
  - `npm run dev` and `npm run build` already run both preparation steps via `predev`/`prebuild`.
  - `npm run preview` serves the built `dist/` output for local verification.
- Validation/test commands:
  - `npm run typecheck`
  - `npm run build`
  - `npm test` (alias of `npm run test:e2e`)
  - `npm run test:e2e`
    - `pretest:e2e` runs `npm run prepare:maxflow` first.
  - `npm run verify` (runs typecheck + build + e2e)
- Test fixture generation:
  - `npm run generate:test-images` regenerates license-safe synthetic fixtures under `public/test_images/`.

E2E workflow details:

- `scripts/test-e2e.sh` starts Vite on `127.0.0.1:${PORT:-5176}` and writes server logs to `${SERVER_LOG:-/tmp/brenizer-vite.log}`.
- E2E behavior can be narrowed/tuned with `test.js` env vars such as `APP_URL`, `SCENARIO_ID`, `SEAM_TIER`, `TURBO_MODE`, `EXPECT_COI`, `MAXFLOW_SELF_TEST`, `SEAM_BENCHMARK`, `SEAM_BENCHMARK_ASSERT`, `SEAM_BENCHMARK_TIERS`, `SEAM_BENCHMARK_SCENARIOS`, and `EXPORT_SMOKE`.

CI (GitHub Pages) workflow notes:

- `.github/workflows/deploy-pages.yml` triggers on pushes to `main` and via manual `workflow_dispatch`.
- `.github/workflows/deploy-pages.yml` uses Node 20 + Python 3.12, then runs `npm ci`, `npm run typecheck`.
- Before `npm run build`, CI bootstraps Emscripten (`emsdk install/activate 5.0.2`) and sources `emsdk_env.sh` so `prepare:maxflow` can compile via `em++`.
