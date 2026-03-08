/**
 * test.js - Headless E2E suite for Brenizer stitching quality.
 *
 * Runs multiple composite scenarios and verifies:
 *  1) pipeline completes without errors
 *  2) enough images are blended
 *  3) composite output is "good enough" via numeric canvas quality metrics
 */
import puppeteer from 'puppeteer';

const APP_URL = process.env.APP_URL || 'http://localhost:5176';

const LANDSCAPE_TILES = [
  'lnd_test/tile30_r1_c1.png', 'lnd_test/tile30_r1_c2.png', 'lnd_test/tile30_r1_c3.png',
  'lnd_test/tile30_r2_c1.png', 'lnd_test/tile30_r2_c2.png', 'lnd_test/tile30_r2_c3.png',
  'lnd_test/tile30_r3_c1.png', 'lnd_test/tile30_r3_c2.png', 'lnd_test/tile30_r3_c3.png',
];

const LANDSCAPE_SNAKE = [
  'lnd_test/tile30_r1_c1.png', 'lnd_test/tile30_r1_c2.png', 'lnd_test/tile30_r1_c3.png',
  'lnd_test/tile30_r2_c3.png', 'lnd_test/tile30_r2_c2.png', 'lnd_test/tile30_r2_c1.png',
  'lnd_test/tile30_r3_c1.png', 'lnd_test/tile30_r3_c2.png', 'lnd_test/tile30_r3_c3.png',
];

const AXM_SEQUENCE = [
  'AXM_1754.jpeg', 'AXM_1755.jpeg', 'AXM_1756.jpeg',
  'AXM_1757.jpeg', 'AXM_1758.jpeg', 'AXM_1759.jpeg',
  'AXM_1760.jpeg', 'AXM_1761.jpeg', 'AXM_1762.jpeg',
];

const AXM_CENTER_OUT = [
  'AXM_1758.jpeg', 'AXM_1757.jpeg', 'AXM_1759.jpeg',
  'AXM_1756.jpeg', 'AXM_1760.jpeg', 'AXM_1755.jpeg',
  'AXM_1761.jpeg', 'AXM_1754.jpeg', 'AXM_1762.jpeg',
];

const SYNTH_ORCHARD_ROW_MAJOR = [
  'synth_orchard_3x3/tile_r1_c1.png', 'synth_orchard_3x3/tile_r1_c2.png', 'synth_orchard_3x3/tile_r1_c3.png',
  'synth_orchard_3x3/tile_r2_c1.png', 'synth_orchard_3x3/tile_r2_c2.png', 'synth_orchard_3x3/tile_r2_c3.png',
  'synth_orchard_3x3/tile_r3_c1.png', 'synth_orchard_3x3/tile_r3_c2.png', 'synth_orchard_3x3/tile_r3_c3.png',
];

const SYNTH_MARKET_SNAKE = [
  'synth_market_4x3/tile_r1_c1.png', 'synth_market_4x3/tile_r1_c2.png', 'synth_market_4x3/tile_r1_c3.png', 'synth_market_4x3/tile_r1_c4.png',
  'synth_market_4x3/tile_r2_c4.png', 'synth_market_4x3/tile_r2_c3.png', 'synth_market_4x3/tile_r2_c2.png', 'synth_market_4x3/tile_r2_c1.png',
  'synth_market_4x3/tile_r3_c1.png', 'synth_market_4x3/tile_r3_c2.png', 'synth_market_4x3/tile_r3_c3.png', 'synth_market_4x3/tile_r3_c4.png',
];

const SCENARIOS = [
  {
    id: 'landscape-row-major',
    files: LANDSCAPE_TILES,
    runOptimizeFirst: true,
    sameCameraSettings: true,
    thresholds: {
      minBlended: 8,
      coverageMin: 0.10,
      bboxCoverageMin: 0.12,
      largestComponentMin: 0.90,
      luminanceVarianceMin: 0.0025,
      edgeEnergyMin: 0.0075,
    },
  },
  {
    id: 'landscape-snake-order',
    files: LANDSCAPE_SNAKE,
    runOptimizeFirst: false,
    sameCameraSettings: false,
    thresholds: {
      minBlended: 8,
      coverageMin: 0.10,
      bboxCoverageMin: 0.12,
      largestComponentMin: 0.88,
      luminanceVarianceMin: 0.0025,
      edgeEnergyMin: 0.0075,
    },
  },
  {
    id: 'axm-sequential',
    files: AXM_SEQUENCE,
    runOptimizeFirst: true,
    sameCameraSettings: true,
    thresholds: {
      minBlended: 6,
      coverageMin: 0.07,
      bboxCoverageMin: 0.09,
      largestComponentMin: 0.80,
      luminanceVarianceMin: 0.0018,
      edgeEnergyMin: 0.0060,
    },
  },
  {
    id: 'axm-center-out',
    files: AXM_CENTER_OUT,
    runOptimizeFirst: false,
    sameCameraSettings: true,
    thresholds: {
      minBlended: 6,
      coverageMin: 0.07,
      bboxCoverageMin: 0.09,
      largestComponentMin: 0.78,
      luminanceVarianceMin: 0.0018,
      edgeEnergyMin: 0.0060,
    },
  },
  {
    id: 'synth-orchard-row-major',
    files: SYNTH_ORCHARD_ROW_MAJOR,
    runOptimizeFirst: false,
    sameCameraSettings: false,
    thresholds: {
      minBlended: 9,
      coverageMin: 0.75,
      bboxCoverageMin: 0.80,
      largestComponentMin: 0.98,
      luminanceVarianceMin: 0.0060,
      edgeEnergyMin: 0.0060,
    },
  },
  {
    id: 'synth-market-snake',
    files: SYNTH_MARKET_SNAKE,
    runOptimizeFirst: false,
    sameCameraSettings: false,
    thresholds: {
      minBlended: 12,
      coverageMin: 0.70,
      bboxCoverageMin: 0.78,
      largestComponentMin: 0.98,
      luminanceVarianceMin: 0.0060,
      edgeEnergyMin: 0.0060,
    },
  },
];

const TIMEOUTS = {
  pageLoadMs: 60_000,
  buttonWaitSec: 150,
  optimizeSec: 360,
  pipelineSec: 720,
  exportSec: 360,
  protocolMs: 1_800_000,
};

function parseBooleanish(raw) {
  if (raw == null || raw === '') return null;
  if (/^(1|true|yes|on)$/i.test(raw)) return true;
  if (/^(0|false|no|off)$/i.test(raw)) return false;
  return null;
}

const SEAM_TIER_OVERRIDE = (process.env.SEAM_TIER || '').trim() || null;
const TURBO_MODE_OVERRIDE = parseBooleanish(process.env.TURBO_MODE);
const EXPECT_COI = /^(1|true|yes)$/i.test(process.env.EXPECT_COI || '');
const MAXFLOW_SELF_TEST = parseBooleanish(process.env.MAXFLOW_SELF_TEST) ?? true;
const RUN_SEAM_BENCHMARKS = /^(1|true|yes)$/i.test(process.env.SEAM_BENCHMARK || '');
const BENCHMARK_ASSERT = /^(1|true|yes)$/i.test(process.env.SEAM_BENCHMARK_ASSERT || '');
const EXPORT_SMOKE = /^(1|true|yes)$/i.test(process.env.EXPORT_SMOKE || '');
const SCENARIO_ID_FILTER = (process.env.SCENARIO_ID || '').trim();
const BENCHMARK_TIER_MATRIX = (process.env.SEAM_BENCHMARK_TIERS || 'legacyCpu,webglGrid')
  .split(',')
  .map(v => v.trim())
  .filter(Boolean);
const BENCHMARK_SCENARIO_IDS = new Set(
  (process.env.SEAM_BENCHMARK_SCENARIOS || 'synth-orchard-row-major,landscape-row-major,axm-center-out')
    .split(',')
    .map(v => v.trim())
    .filter(Boolean),
);

function formatMetrics(m) {
  return [
    `blended=${m.blendedCount}`,
    `coverage=${m.coverage.toFixed(3)}`,
    `bbox=${m.bboxCoverage.toFixed(3)}`,
    `largestCC=${m.largestComponentRatio.toFixed(3)}`,
    `var=${m.luminanceVariance.toFixed(4)}`,
    `edge=${m.edgeEnergy.toFixed(4)}`,
  ].join(', ');
}

function withRuntimeOptions(baseUrl, options = {}) {
  const url = new URL(baseUrl);
  if (options.seamTier) {
    url.searchParams.set('seamTier', options.seamTier);
  }
  if (typeof options.turboMode === 'boolean') {
    url.searchParams.set('turboMode', options.turboMode ? '1' : '0');
  }
  return url.toString();
}

function median(values) {
  if (!values.length) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) / 2
    : sorted[mid];
}

function formatCountMap(entries) {
  return Array.from(entries.entries())
    .sort((a, b) => a[0].localeCompare(b[0]))
    .map(([key, count]) => `${key}:${count}`)
    .join(', ');
}

function summarizeSeamJobs(seamJobs) {
  if (!Array.isArray(seamJobs) || seamJobs.length === 0) return 'seams=0';
  const tierCounts = new Map();
  const builderCounts = new Map();
  const solverCounts = new Map();
  const readbackBytes = [];
  const totalMs = [];
  const solverMs = [];
  for (const job of seamJobs) {
    tierCounts.set(job.tier, (tierCounts.get(job.tier) || 0) + 1);
    builderCounts.set(job.builderBackend, (builderCounts.get(job.builderBackend) || 0) + 1);
    solverCounts.set(job.solverBackend, (solverCounts.get(job.solverBackend) || 0) + 1);
    readbackBytes.push(Number(job.readbackBytes) || 0);
    totalMs.push(Number(job.totalMs) || 0);
    solverMs.push(Number(job.solverMs) || 0);
  }
  return [
    `seams=${seamJobs.length}`,
    `tiers=${formatCountMap(tierCounts)}`,
    `builders=${formatCountMap(builderCounts)}`,
    `solvers=${formatCountMap(solverCounts)}`,
    `medianReadback=${Math.round(median(readbackBytes)).toLocaleString()}B`,
    `medianSolver=${median(solverMs).toFixed(1)}ms`,
    `medianTotal=${median(totalMs).toFixed(1)}ms`,
  ].join(', ');
}

function extractBenchmarkStats(seamJobs) {
  const readbackBytes = seamJobs.map(job => Number(job.readbackBytes) || 0);
  const totalMs = seamJobs.map(job => Number(job.totalMs) || 0);
  const solverMs = seamJobs.map(job => Number(job.solverMs) || 0);
  return {
    count: seamJobs.length,
    medianReadbackBytes: median(readbackBytes),
    medianTotalMs: median(totalMs),
    medianSolverMs: median(solverMs),
  };
}

function isTransientNavigationError(err) {
  if (!(err instanceof Error)) return false;
  return /Execution context was destroyed|Cannot find context with specified id|Target closed/i.test(err.message);
}

let maxflowProbeDone = false;
let seamPostProbeDone = false;

async function waitForRuntimeBootstrap(page, scenarioLabel) {
  const maxAttempts = 4;
  for (let attempt = 1; attempt <= maxAttempts; attempt++) {
    try {
      await page.waitForFunction(
        () => {
          const runtime = window.__brenizerRuntime;
          const stitchBtn = document.getElementById('btn-stitch');
          const status = document.getElementById('status-bar')?.textContent || '';
          return !!runtime?.caps && !!stitchBtn && !/^Error:/i.test(status);
        },
        { timeout: TIMEOUTS.pageLoadMs },
      );
      return;
    } catch (err) {
      if (!isTransientNavigationError(err) || attempt === maxAttempts) {
        try {
          const snapshot = await page.evaluate(() => ({
            hasStitchButton: !!document.getElementById('btn-stitch'),
            hasRuntimeCaps: !!window.__brenizerRuntime?.caps,
            status: document.getElementById('status-bar')?.textContent || '',
            readyState: document.readyState,
          }));
          console.log(`[${scenarioLabel}] Bootstrap snapshot: ${JSON.stringify(snapshot)}`);
          if (snapshot.hasStitchButton && snapshot.hasRuntimeCaps && !/^Error:/i.test(snapshot.status)) {
            return;
          }
        } catch {
          // Ignore evaluation failures and rethrow the original error below.
        }
        throw err;
      }
      console.log(`[${scenarioLabel}] Waiting through bootstrap reload (${attempt}/${maxAttempts})...`);
      await page.waitForNavigation({
        waitUntil: 'networkidle0',
        timeout: TIMEOUTS.pageLoadMs,
      }).catch(() => undefined);
    }
  }
}

async function runMaxflowSelfTest(page, scenarioLabel) {
  const probe = await page.evaluate(async () => {
    function buildCase(id, gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hardConstraints) {
      return {
        id,
        gridW,
        gridH,
        dataCosts,
        edgeWeightsH,
        edgeWeightsV,
        hardConstraints,
      };
    }

    function makeWorker(baseUrl) {
      return new Worker(new URL('workers/seam-worker.js', baseUrl).toString());
    }

    function waitForMessage(worker, predicate, timeoutMs) {
      return new Promise((resolve, reject) => {
        const timer = setTimeout(() => {
          cleanup();
          reject(new Error('Timed out waiting for worker message'));
        }, timeoutMs);

        function onMessage(ev) {
          const msg = ev.data;
          if (!predicate(msg)) return;
          cleanup();
          resolve(msg);
        }

        function onError(err) {
          cleanup();
          reject(new Error(err.message || 'Worker crashed'));
        }

        function cleanup() {
          clearTimeout(timer);
          worker.removeEventListener('message', onMessage);
          worker.removeEventListener('error', onError);
        }

        worker.addEventListener('message', onMessage);
        worker.addEventListener('error', onError);
      });
    }

    async function initWorker(worker, baseUrl) {
      worker.postMessage({
        type: 'init',
        baseUrl,
        maxflowPath: new URL('wasm/maxflow/maxflow-simd.js', baseUrl).toString(),
        wasmPathSimd: new URL('wasm/maxflow/maxflow-simd.js', baseUrl).toString(),
        wasmPathThreads: new URL('wasm/maxflow/maxflow-threads.js', baseUrl).toString(),
        wasmWorkerPathThreads: new URL('wasm/maxflow/maxflow-threads.worker.js', baseUrl).toString(),
      });
      return waitForMessage(worker, (msg) => msg.type === 'progress' && msg.stage === 'seam-init', 20000);
    }

    async function solve(worker, graph, forceLegacy) {
      const jobId = `${graph.id}:${forceLegacy ? 'legacy' : 'native'}`;
      const dataCostsBuffer = new Float32Array(graph.dataCosts).buffer;
      const edgeWeightsHBuffer = new Float32Array(graph.edgeWeightsH).buffer;
      const edgeWeightsVBuffer = new Float32Array(graph.edgeWeightsV).buffer;
      const hardConstraintsBuffer = new Uint8Array(graph.hardConstraints).buffer;
      worker.postMessage({
        type: 'solve',
        jobId,
        gridW: graph.gridW,
        gridH: graph.gridH,
        dataCostsBuffer,
        edgeWeightsHBuffer,
        edgeWeightsVBuffer,
        hardConstraintsBuffer,
        params: forceLegacy ? { forceLegacy: true } : {},
      }, [dataCostsBuffer, edgeWeightsHBuffer, edgeWeightsVBuffer, hardConstraintsBuffer]);

      const result = await waitForMessage(
        worker,
        (msg) => msg.jobId === jobId && (msg.type === 'result' || msg.type === 'error'),
        20000,
      );
      if (result.type === 'error') {
        throw new Error(result.message || 'Worker solve failed');
      }
      return {
        backendId: result.backendId,
        labels: Array.from(new Uint8Array(result.labelsBuffer)),
      };
    }

    const graphs = [
      buildCase(
        'striped-4x3',
        4,
        3,
        [
          0.2, 2.0, 0.2, 2.0, 2.0, 0.2, 2.0, 0.2,
          0.2, 2.0, 0.2, 2.0, 2.0, 0.2, 2.0, 0.2,
          0.2, 2.0, 0.2, 2.0, 2.0, 0.2, 2.0, 0.2,
        ],
        [1.3, 1.1, 1.2, 1.3, 1.1, 1.2, 1.3, 1.1, 1.2],
        [0.9, 0.8, 0.9, 0.9, 0.8, 0.9, 0.9, 0.8],
        new Array(12).fill(0),
      ),
      buildCase(
        'hard-constraints-3x3',
        3,
        3,
        [
          0.7, 0.8, 0.9, 0.6, 1.4, 0.2,
          0.6, 0.8, 1.0, 0.5, 1.5, 0.2,
          0.7, 0.9, 1.1, 0.4, 1.6, 0.2,
        ],
        [0.4, 1.0, 0.5, 0.9, 0.5, 1.1],
        [1.2, 0.6, 1.0, 1.1, 0.7, 0.9],
        [1, 0, 2, 1, 0, 2, 1, 0, 2],
      ),
    ];

    const baseUrl = new URL('./', window.location.href).toString();
    const worker = makeWorker(baseUrl);
    try {
      const init = await initWorker(worker, baseUrl);
      const backendId = init.backendId || 'unknown';
      const expectedBackendPrefix = self.crossOriginIsolated ? 'wasm-threads' : 'wasm-simd';
      const cases = [];

      for (const graph of graphs) {
        const compiled = await solve(worker, graph, false);
        const legacy = await solve(worker, graph, true);
        const matches = compiled.labels.length === legacy.labels.length
          && compiled.labels.every((value, idx) => value === legacy.labels[idx]);
        cases.push({
          id: graph.id,
          matches,
          compiledBackend: compiled.backendId,
          compiledLabels: compiled.labels,
          legacyLabels: legacy.labels,
        });
      }

      return {
        ok: backendId.startsWith(expectedBackendPrefix) && cases.every((testCase) => testCase.matches),
        backendId,
        expectedBackendPrefix,
        initInfo: init.info || '',
        cases,
      };
    } finally {
      worker.terminate();
    }
  });

  console.log(
    `[${scenarioLabel}] maxflow self-test backend=${probe.backendId} expected=${probe.expectedBackendPrefix} ` +
    `info=${probe.initInfo || 'n/a'} ` +
    `cases=${probe.cases.map((testCase) => `${testCase.id}:${testCase.matches ? 'ok' : 'mismatch'}`).join(',')}`,
  );
  if (!probe.ok) {
    throw new Error(
      `Maxflow self-test failed: backend=${probe.backendId}, expected=${probe.expectedBackendPrefix}, ` +
      `info=${probe.initInfo || 'n/a'}, ` +
      `cases=${probe.cases.map((testCase) => `${testCase.id}:${testCase.matches ? 'ok' : 'mismatch'}`).join(', ')}`,
    );
  }
}

async function runSeamPostprocessSelfTest(page, scenarioLabel) {
  const probe = await page.evaluate(async () => {
    const mod = await import(new URL('/src/seamPostprocess.ts', window.location.href).toString());
    return mod.runSeamPostprocessSyntheticSelfTest();
  });

  console.log(
    `[${scenarioLabel}] seam-post self-test before=${probe.beforeStep.toFixed(2)} after=${probe.afterStep.toFixed(2)} ` +
    `edgeBefore=${probe.edgeBefore.toFixed(2)} edgeAfter=${probe.edgeAfter.toFixed(2)} retention=${probe.edgeRetention.toFixed(3)} ` +
    `seams=${probe.seamCount}`,
  );

  if (!(probe.afterStep < probe.beforeStep * 0.55)) {
    throw new Error(
      `Seam post-process self-test failed to reduce seam step enough: before=${probe.beforeStep}, after=${probe.afterStep}`,
    );
  }
  if (!(probe.edgeRetention > 0.72)) {
    throw new Error(
      `Seam post-process self-test over-smoothed crossing edge: retention=${probe.edgeRetention}`,
    );
  }
  if (!(probe.seamCount >= 1)) {
    throw new Error('Seam post-process self-test found no seam candidates');
  }
}

async function runScenario(browser, scenario, options = {}) {
  const seamTier = options.seamTier ?? SEAM_TIER_OVERRIDE;
  const turboMode = typeof options.turboMode === 'boolean' ? options.turboMode : TURBO_MODE_OVERRIDE;
  const scenarioUrl = withRuntimeOptions(APP_URL, { seamTier, turboMode });
  const suffix = [
    seamTier ? `seam=${seamTier}` : '',
    typeof turboMode === 'boolean' ? `turbo=${turboMode ? 'on' : 'off'}` : '',
  ].filter(Boolean).join(',');
  const scenarioLabel = suffix ? `${scenario.id}@${suffix}` : scenario.id;
  const page = await browser.newPage();
  page.on('console', msg => console.log(`[${scenarioLabel}] PAGE:`, msg.text()));
  page.on('pageerror', err => console.error(`[${scenarioLabel}] PAGE ERROR:`, err));

  try {
    await page.goto(scenarioUrl, {
      waitUntil: 'networkidle0',
      timeout: TIMEOUTS.pageLoadMs,
    });
    if (turboMode) {
      await page.waitForNavigation({
        waitUntil: 'networkidle0',
        timeout: 10000,
      }).catch(() => undefined);
    }
    await waitForRuntimeBootstrap(page, scenarioLabel);
    if (!seamPostProbeDone) {
      await runSeamPostprocessSelfTest(page, scenarioLabel);
      seamPostProbeDone = true;
    }
    if (MAXFLOW_SELF_TEST && !maxflowProbeDone) {
      await runMaxflowSelfTest(page, scenarioLabel);
      maxflowProbeDone = true;
    }

    const result = await page.evaluate(async (sc, timeouts) => {
      function sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
      }

      function readStatus() {
        const bar = document.getElementById('status-bar');
        const msg = bar?.querySelector('.status-msg');
        const status = (msg || bar)?.textContent || '';
        return status.trim();
      }

      async function waitForEnabledControl(id, timeoutSec) {
        for (let i = 0; i < timeoutSec; i++) {
          const control = document.getElementById(id);
          if (control && !control.disabled) return true;
          await sleep(1000);
        }
        return false;
      }

      function inferMime(path) {
        const lower = path.toLowerCase();
        if (lower.endsWith('.png')) return 'image/png';
        if (lower.endsWith('.jpg') || lower.endsWith('.jpeg')) return 'image/jpeg';
        if (lower.endsWith('.heic')) return 'image/heic';
        if (lower.endsWith('.dng')) return 'image/x-adobe-dng';
        return 'application/octet-stream';
      }

      function largestConnectedComponentRatio(mask, width, height, occupiedCount) {
        if (occupiedCount <= 0) return 0;
        const visited = new Uint8Array(mask.length);
        const queue = new Int32Array(mask.length);
        let largest = 0;
        let qHead = 0;
        let qTail = 0;
        for (let i = 0; i < mask.length; i++) {
          if (!mask[i] || visited[i]) continue;
          visited[i] = 1;
          queue[0] = i;
          qHead = 0;
          qTail = 1;
          let size = 0;
          while (qHead < qTail) {
            const cur = queue[qHead++];
            size++;
            const x = cur % width;
            const y = (cur / width) | 0;

            if (x > 0) {
              const n = cur - 1;
              if (mask[n] && !visited[n]) {
                visited[n] = 1;
                queue[qTail++] = n;
              }
            }
            if (x + 1 < width) {
              const n = cur + 1;
              if (mask[n] && !visited[n]) {
                visited[n] = 1;
                queue[qTail++] = n;
              }
            }
            if (y > 0) {
              const n = cur - width;
              if (mask[n] && !visited[n]) {
                visited[n] = 1;
                queue[qTail++] = n;
              }
            }
            if (y + 1 < height) {
              const n = cur + width;
              if (mask[n] && !visited[n]) {
                visited[n] = 1;
                queue[qTail++] = n;
              }
            }
          }
          if (size > largest) largest = size;
        }
        return largest / occupiedCount;
      }

      function analyzeCanvasQuality(canvas) {
        const w = canvas.width | 0;
        const h = canvas.height | 0;
        if (w <= 0 || h <= 0) {
          return {
            ok: false,
            reason: `invalid canvas size ${w}x${h}`,
          };
        }

        const temp = document.createElement('canvas');
        temp.width = w;
        temp.height = h;
        const ctx = temp.getContext('2d', { willReadFrequently: true });
        if (!ctx) {
          return {
            ok: false,
            reason: '2D context unavailable for quality analysis',
          };
        }
        ctx.drawImage(canvas, 0, 0);
        const data = ctx.getImageData(0, 0, w, h).data;

        const step = Math.max(1, Math.floor(Math.max(w, h) / 320));
        const sw = Math.max(1, Math.floor(w / step));
        const sh = Math.max(1, Math.floor(h / step));
        const n = sw * sh;
        const mask = new Uint8Array(n);
        const lum = new Float32Array(n);

        let alphaOccupied = 0;
        let sampled = 0;
        for (let y = 0; y < sh; y++) {
          const py = Math.min(h - 1, y * step);
          for (let x = 0; x < sw; x++) {
            const px = Math.min(w - 1, x * step);
            const idx = (py * w + px) * 4;
            if (data[idx + 3] > 8) alphaOccupied++;
            sampled++;
          }
        }

        const alphaRatio = sampled > 0 ? alphaOccupied / sampled : 0;
        const useAlphaMask = alphaRatio > 0.01 && alphaRatio < 0.995;
        let occupied = 0;
        let minX = sw, minY = sh, maxX = -1, maxY = -1;

        for (let y = 0; y < sh; y++) {
          const py = Math.min(h - 1, y * step);
          for (let x = 0; x < sw; x++) {
            const px = Math.min(w - 1, x * step);
            const idx = (py * w + px) * 4;
            const r = data[idx];
            const g = data[idx + 1];
            const b = data[idx + 2];
            const a = data[idx + 3];
            const l = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255;
            const i = y * sw + x;
            lum[i] = l;

            const nonBlack = (r + g + b) > 20;
            const isOccupied = useAlphaMask ? a > 8 : (l > 0.04 && nonBlack);
            if (isOccupied) {
              mask[i] = 1;
              occupied++;
              if (x < minX) minX = x;
              if (x > maxX) maxX = x;
              if (y < minY) minY = y;
              if (y > maxY) maxY = y;
            }
          }
        }

        const coverage = occupied / n;
        const bboxCoverage = occupied > 0
          ? ((maxX - minX + 1) * (maxY - minY + 1)) / n
          : 0;

        let mean = 0;
        let count = 0;
        for (let i = 0; i < n; i++) {
          if (!mask[i]) continue;
          mean += lum[i];
          count++;
        }
        mean = count > 0 ? mean / count : 0;

        let varSum = 0;
        for (let i = 0; i < n; i++) {
          if (!mask[i]) continue;
          const d = lum[i] - mean;
          varSum += d * d;
        }
        const luminanceVariance = count > 1 ? varSum / (count - 1) : 0;

        let edgeSum = 0;
        let edgeCount = 0;
        for (let y = 0; y < sh - 1; y++) {
          for (let x = 0; x < sw - 1; x++) {
            const i = y * sw + x;
            const right = i + 1;
            const down = i + sw;
            if (mask[i] || mask[right]) {
              edgeSum += Math.abs(lum[right] - lum[i]);
              edgeCount++;
            }
            if (mask[i] || mask[down]) {
              edgeSum += Math.abs(lum[down] - lum[i]);
              edgeCount++;
            }
          }
        }
        const edgeEnergy = edgeCount > 0 ? edgeSum / edgeCount : 0;
        const largestComponentRatio = largestConnectedComponentRatio(mask, sw, sh, occupied);

        return {
          ok: true,
          width: w,
          height: h,
          sampleStep: step,
          sampledWidth: sw,
          sampledHeight: sh,
          coverage,
          bboxCoverage,
          largestComponentRatio,
          luminanceVariance,
          edgeEnergy,
        };
      }

      function parseBlendedCount(status) {
        const m = status.match(/composite complete\s*[-\u2014]\s*(\d+)\s+images blended/i);
        if (!m) return null;
        return Number(m[1]);
      }

      window.addEventListener('unhandledrejection', e => {
        console.error('UNHANDLED REJECTION:', e.reason?.message || e.reason);
      });

      const fetchedFiles = [];
      try {
        for (const name of sc.files) {
          const resp = await fetch('/test_images/' + name);
          if (!resp.ok) {
            return { ok: false, phase: 'fetch', reason: `HTTP ${resp.status} for ${name}` };
          }
          const blob = await resp.blob();
          const fileName = name.split('/').pop() || name;
          fetchedFiles.push(new File([blob], fileName, { type: inferMime(name) }));
        }
      } catch (err) {
        return { ok: false, phase: 'fetch', reason: String(err?.message || err) };
      }

      const input = document.getElementById('file-input');
      if (!input) return { ok: false, phase: 'upload', reason: 'missing #file-input' };
      const dt = new DataTransfer();
      for (const f of fetchedFiles) dt.items.add(f);
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));

      const alignStep = document.getElementById('workflow-step-align');
      const cameraStep = document.getElementById('workflow-step-camera');
      const optimizeBtn = document.getElementById('btn-optimize');
      const previewBtn = document.getElementById('btn-stitch');
      if (!alignStep || !cameraStep || !optimizeBtn || !previewBtn) {
        return { ok: false, phase: 'workflow', reason: 'missing workflow controls' };
      }

      alignStep.value = 'alignmentOnly';
      alignStep.dispatchEvent(new Event('change', { bubbles: true }));
      await sleep(100);

      const cameraChoiceReady = await waitForEnabledControl(
        'workflow-step-camera',
        timeouts.buttonWaitSec,
      );
      if (!cameraChoiceReady) {
        return { ok: false, phase: 'workflow', reason: 'camera settings dropdown stayed disabled after alignment choice' };
      }

      cameraStep.value = sc.sameCameraSettings === false ? 'mixed' : 'same';
      cameraStep.dispatchEvent(new Event('change', { bubbles: true }));
      await sleep(100);

      const optimizeReady = await waitForEnabledControl('btn-optimize', timeouts.buttonWaitSec);
      if (!optimizeReady) {
        return { ok: false, phase: 'optimize-ready', reason: 'optimize button stayed disabled after workflow setup' };
      }

      optimizeBtn.click();
      let optimizeStatus = '';
      for (let i = 0; i < timeouts.optimizeSec; i++) {
        await sleep(1000);
        optimizeStatus = readStatus();
        if (optimizeStatus.toLowerCase().includes('optimization complete')) break;
        if (optimizeStatus.toLowerCase().includes('optimization error') ||
            optimizeStatus.toLowerCase().includes('first-pass optimization error') ||
            optimizeStatus.toLowerCase().includes('pipeline error')) {
          return { ok: false, phase: 'optimize', reason: optimizeStatus || 'optimization failed' };
        }
        if (i === timeouts.optimizeSec - 1) {
          return { ok: false, phase: 'optimize', reason: `optimization timeout: ${optimizeStatus}` };
        }
      }

      const stitchReady = await waitForEnabledControl('btn-stitch', timeouts.buttonWaitSec);
      if (!stitchReady) {
        return { ok: false, phase: 'ready', reason: 'stitch button stayed disabled after optimization' };
      }

      previewBtn.click();

      let finalStatus = '';
      let lastStatus = '';
      for (let i = 0; i < timeouts.pipelineSec; i++) {
        await sleep(1000);
        const status = readStatus();
        if (status !== lastStatus) {
          console.log(`Status: ${status}`);
          lastStatus = status;
        }
        const s = status.toLowerCase();
        if (s.includes('composite complete') || s.includes('pipeline complete')) {
          finalStatus = status;
          break;
        }
        if (s.includes('pipeline error') || s.includes('failed') || s.includes('no matching pairs')) {
          return { ok: false, phase: 'pipeline', reason: status };
        }
        if (i === timeouts.pipelineSec - 1) {
          return { ok: false, phase: 'pipeline', reason: `timeout: ${status}` };
        }
      }

      const blendedCount = parseBlendedCount(finalStatus);
      const canvas = document.getElementById('preview-canvas');
      if (!canvas) {
        return { ok: false, phase: 'quality', reason: 'missing #preview-canvas', status: finalStatus };
      }
      const quality = analyzeCanvasQuality(canvas);
      if (!quality.ok) {
        return { ok: false, phase: 'quality', reason: quality.reason, status: finalStatus };
      }

      const previewSeamPost = Array.isArray(window.__brenizerPerf?.seamPost?.preview)
        ? window.__brenizerPerf.seamPost.preview
        : null;
      if (!previewSeamPost) {
        return {
          ok: false,
          phase: 'seam-post-preview',
          reason: 'preview seam smoothing did not record a stage result',
          status: finalStatus,
        };
      }

      let exportStatus = '';
      let exportSeamPost = null;
      if (timeouts.exportSmoke) {
        const exportStep = document.getElementById('workflow-step-export');
        if (!exportStep) {
          return {
            ok: false,
            phase: 'export',
            reason: 'missing #workflow-step-export',
            status: finalStatus,
          };
        }

        const exportReady = await waitForEnabledControl('workflow-step-export', timeouts.buttonWaitSec);
        if (!exportReady) {
          return {
            ok: false,
            phase: 'export',
            reason: 'export dropdown stayed disabled after preview',
            status: finalStatus,
          };
        }

        exportStep.value = 'fullres';
        exportStep.dispatchEvent(new Event('change', { bubbles: true }));

        let lastExportStatus = finalStatus;
        for (let i = 0; i < timeouts.exportSec; i++) {
          await sleep(1000);
          const status = readStatus();
          if (status !== lastExportStatus) {
            console.log(`Export Status: ${status}`);
            lastExportStatus = status;
          }
          const s = status.toLowerCase();
          if (s.includes('exported ')) {
            exportStatus = status;
            break;
          }
          if (s.includes('export error') || s.includes('pipeline error') || s.includes('failed')) {
            return {
              ok: false,
              phase: 'export',
              reason: status,
              status,
            };
          }
          if (i === timeouts.exportSec - 1) {
            return {
              ok: false,
              phase: 'export',
              reason: `timeout: ${status}`,
              status,
            };
          }
        }

        exportSeamPost = Array.isArray(window.__brenizerPerf?.seamPost?.export)
          ? window.__brenizerPerf.seamPost.export
          : null;
        if (!exportSeamPost) {
          return {
            ok: false,
            phase: 'seam-post-export',
            reason: 'export seam smoothing did not record a stage result',
            status: exportStatus || finalStatus,
          };
        }
      }

      const t = sc.thresholds;
      const failures = [];
      if (blendedCount !== null && blendedCount < t.minBlended) {
        failures.push(`blended ${blendedCount} < ${t.minBlended}`);
      }
      if (quality.coverage < t.coverageMin) {
        failures.push(`coverage ${quality.coverage.toFixed(3)} < ${t.coverageMin}`);
      }
      if (quality.bboxCoverage < t.bboxCoverageMin) {
        failures.push(`bboxCoverage ${quality.bboxCoverage.toFixed(3)} < ${t.bboxCoverageMin}`);
      }
      if (quality.largestComponentRatio < t.largestComponentMin) {
        failures.push(`largestCC ${quality.largestComponentRatio.toFixed(3)} < ${t.largestComponentMin}`);
      }
      if (quality.luminanceVariance < t.luminanceVarianceMin) {
        failures.push(`luminanceVariance ${quality.luminanceVariance.toFixed(4)} < ${t.luminanceVarianceMin}`);
      }
      if (quality.edgeEnergy < t.edgeEnergyMin) {
        failures.push(`edgeEnergy ${quality.edgeEnergy.toFixed(4)} < ${t.edgeEnergyMin}`);
      }

      const runtime = {
        crossOriginIsolated: self.crossOriginIsolated === true,
        turboModeEnabled: window.__brenizerRuntime?.turboModeEnabled ?? null,
        seamTier: window.__brenizerRuntime?.caps?.seamAccelerationTier ?? null,
        crossOriginIsolationMode: window.__brenizerRuntime?.caps?.crossOriginIsolationMode ?? 'none',
      };

      return {
        ok: failures.length === 0,
        phase: failures.length === 0 ? 'done' : 'quality',
        reason: failures.join('; '),
        status: finalStatus,
        runtime,
        seamJobs: Array.isArray(window.__brenizerPerf?.seamJobs)
          ? window.__brenizerPerf.seamJobs
          : [],
        seamPost: {
          previewCount: previewSeamPost.length,
          exportCount: exportSeamPost?.length ?? null,
          exportStatus: exportStatus || null,
        },
        metrics: {
          blendedCount: blendedCount ?? -1,
          coverage: quality.coverage,
          bboxCoverage: quality.bboxCoverage,
          largestComponentRatio: quality.largestComponentRatio,
          luminanceVariance: quality.luminanceVariance,
          edgeEnergy: quality.edgeEnergy,
        },
      };
    }, scenario, { ...TIMEOUTS, exportSmoke: EXPORT_SMOKE });

    return result;
  } finally {
    await page.close();
  }
}

async function main() {
  const selectedScenarios = SCENARIO_ID_FILTER
    ? SCENARIOS.filter(s => s.id === SCENARIO_ID_FILTER)
    : SCENARIOS;

  console.log('--- Brenizer E2E suite: multi-scenario composite quality checks ---');
  console.log(`Server: ${APP_URL}`);
  if (SEAM_TIER_OVERRIDE) console.log(`Forced seam tier: ${SEAM_TIER_OVERRIDE}`);
  if (typeof TURBO_MODE_OVERRIDE === 'boolean') console.log(`Forced turbo mode: ${TURBO_MODE_OVERRIDE ? 'on' : 'off'}`);
  if (EXPORT_SMOKE) console.log('Full-resolution export smoke: enabled');
  console.log(`Scenarios: ${selectedScenarios.map(s => s.id).join(', ')}`);
  console.log();

  if (selectedScenarios.length === 0) {
    console.error(`No scenarios matched SCENARIO_ID=${SCENARIO_ID_FILTER}`);
    process.exit(1);
  }

  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      '--disable-setuid-sandbox',
      '--use-gl=angle',
      '--use-angle=swiftshader',
      '--enable-unsafe-swiftshader',
    ],
    protocolTimeout: TIMEOUTS.protocolMs,
  });

  const failures = [];
  try {
    for (const scenario of selectedScenarios) {
      console.log(`Running scenario: ${scenario.id} (${scenario.files.length} images)`);
      const result = await runScenario(browser, scenario);
      if (EXPECT_COI && !result.runtime?.crossOriginIsolated) {
        result.ok = false;
        result.phase = 'runtime';
        result.reason = 'crossOriginIsolated was false';
      }
      if (!result.ok) {
        failures.push({ scenario: scenario.id, result });
        console.log(`  FAIL [${result.phase}] ${result.reason || 'unknown reason'}`);
        if (result.status) console.log(`  status: ${result.status}`);
        if (result.runtime) {
          console.log(
            `  runtime: coi=${result.runtime.crossOriginIsolated} mode=${result.runtime.crossOriginIsolationMode} ` +
            `turbo=${result.runtime.turboModeEnabled} seamTier=${result.runtime.seamTier}`,
          );
        }
        console.log(`  seam: ${summarizeSeamJobs(result.seamJobs || [])}`);
        if (result.seamPost) {
          console.log(
            `  seam-post: preview=${result.seamPost.previewCount}` +
            (result.seamPost.exportCount === null ? '' : ` export=${result.seamPost.exportCount}`),
          );
        }
      } else {
        console.log(`  PASS ${formatMetrics(result.metrics)}`);
        console.log(`  status: ${result.status}`);
        if (result.runtime) {
          console.log(
            `  runtime: coi=${result.runtime.crossOriginIsolated} mode=${result.runtime.crossOriginIsolationMode} ` +
            `turbo=${result.runtime.turboModeEnabled} seamTier=${result.runtime.seamTier}`,
          );
        }
        console.log(`  seam: ${summarizeSeamJobs(result.seamJobs || [])}`);
        if (result.seamPost) {
          console.log(
            `  seam-post: preview=${result.seamPost.previewCount}` +
            (result.seamPost.exportCount === null ? '' : ` export=${result.seamPost.exportCount}`),
          );
        }
      }
      console.log();
    }

    if (RUN_SEAM_BENCHMARKS) {
      console.log('--- Seam benchmark matrix ---');
      const benchmarkScenarios = selectedScenarios.filter(s => BENCHMARK_SCENARIO_IDS.has(s.id));
      const benchmarkResults = [];

      for (const scenario of benchmarkScenarios) {
        for (const seamTier of BENCHMARK_TIER_MATRIX) {
          console.log(`Benchmark scenario: ${scenario.id} @ ${seamTier}`);
          const result = await runScenario(browser, scenario, { seamTier, turboMode: TURBO_MODE_OVERRIDE });
          if (!result.ok) {
            failures.push({ scenario: `${scenario.id}@${seamTier}`, result });
            console.log(`  FAIL [${result.phase}] ${result.reason || 'unknown reason'}`);
            if (result.status) console.log(`  status: ${result.status}`);
            console.log(`  seam: ${summarizeSeamJobs(result.seamJobs || [])}`);
            console.log();
            continue;
          }

          const stats = extractBenchmarkStats(result.seamJobs || []);
          benchmarkResults.push({ scenario: scenario.id, seamTier, stats });
          console.log(`  PASS ${formatMetrics(result.metrics)}`);
          console.log(`  seam: ${summarizeSeamJobs(result.seamJobs || [])}`);
          console.log();
        }
      }

      for (const scenario of benchmarkScenarios) {
        const runs = benchmarkResults.filter(r => r.scenario === scenario.id);
        if (runs.length === 0) continue;
        console.log(`Benchmark summary: ${scenario.id}`);
        for (const run of runs) {
          console.log(
            `  ${run.seamTier}: seams=${run.stats.count}, medianReadback=${Math.round(run.stats.medianReadbackBytes).toLocaleString()}B, ` +
            `medianSolver=${run.stats.medianSolverMs.toFixed(1)}ms, medianTotal=${run.stats.medianTotalMs.toFixed(1)}ms`,
          );
        }

        if (BENCHMARK_ASSERT) {
          const legacy = runs.find(r => r.seamTier === 'legacyCpu');
          for (const run of runs) {
            if (!legacy || run.seamTier === 'legacyCpu' || run.stats.count === 0 || legacy.stats.count === 0) continue;
            const readbackRatio = legacy.stats.medianReadbackBytes > 0
              ? run.stats.medianReadbackBytes / legacy.stats.medianReadbackBytes
              : 0;
            if (readbackRatio > 0.05) {
              failures.push({
                scenario: `${scenario.id}@${run.seamTier}`,
                result: {
                  phase: 'benchmark',
                  reason: `median readback ratio ${readbackRatio.toFixed(3)} exceeds 0.05`,
                  status: '',
                },
              });
              console.log(
                `  FAIL benchmark readback: ${run.seamTier} medianReadback ${Math.round(run.stats.medianReadbackBytes).toLocaleString()}B ` +
                `vs legacy ${Math.round(legacy.stats.medianReadbackBytes).toLocaleString()}B`,
              );
            }
          }
        }
        console.log();
      }
    }
  } finally {
    await browser.close();
  }

  console.log('==========================================');
  if (failures.length > 0) {
    console.log(`E2E FAILED: ${failures.length}/${selectedScenarios.length} scenario(s) failed`);
    for (const f of failures) {
      console.log(`- ${f.scenario}: [${f.result.phase}] ${f.result.reason}`);
      if (f.result.status) console.log(`  status: ${f.result.status}`);
    }
    process.exit(1);
  }

  console.log(`E2E PASSED: ${selectedScenarios.length}/${selectedScenarios.length} scenarios`);
  process.exit(0);
}

main().catch(err => {
  console.error('Test runner error:', err);
  process.exit(1);
});
