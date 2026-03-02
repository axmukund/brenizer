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
  protocolMs: 1_800_000,
};

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

async function runScenario(browser, scenario) {
  const page = await browser.newPage();
  page.on('console', msg => console.log(`[${scenario.id}] PAGE:`, msg.text()));
  page.on('pageerror', err => console.error(`[${scenario.id}] PAGE ERROR:`, err));

  try {
    await page.goto(APP_URL, {
      waitUntil: 'networkidle0',
      timeout: TIMEOUTS.pageLoadMs,
    });

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

      async function waitForEnabledButton(id, timeoutSec) {
        for (let i = 0; i < timeoutSec; i++) {
          const btn = document.getElementById(id);
          if (btn && !btn.disabled) return true;
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

      const stitchReady = await waitForEnabledButton('btn-stitch', timeouts.buttonWaitSec);
      if (!stitchReady) {
        return { ok: false, phase: 'ready', reason: 'stitch button stayed disabled' };
      }

      if (sc.runOptimizeFirst) {
        const optimizeReady = await waitForEnabledButton('btn-optimize', timeouts.buttonWaitSec);
        if (!optimizeReady) {
          return { ok: false, phase: 'optimize-ready', reason: 'optimize button stayed disabled' };
        }

        const optimizeBtn = document.getElementById('btn-optimize');
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
      }

      const stitchBtn = document.getElementById('btn-stitch');
      stitchBtn.click();

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

      return {
        ok: failures.length === 0,
        phase: failures.length === 0 ? 'done' : 'quality',
        reason: failures.join('; '),
        status: finalStatus,
        metrics: {
          blendedCount: blendedCount ?? -1,
          coverage: quality.coverage,
          bboxCoverage: quality.bboxCoverage,
          largestComponentRatio: quality.largestComponentRatio,
          luminanceVariance: quality.luminanceVariance,
          edgeEnergy: quality.edgeEnergy,
        },
      };
    }, scenario, TIMEOUTS);

    return result;
  } finally {
    await page.close();
  }
}

async function main() {
  console.log('--- Brenizer E2E suite: multi-scenario composite quality checks ---');
  console.log(`Server: ${APP_URL}`);
  console.log(`Scenarios: ${SCENARIOS.map(s => s.id).join(', ')}`);
  console.log();

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
    for (const scenario of SCENARIOS) {
      console.log(`Running scenario: ${scenario.id} (${scenario.files.length} images)`);
      const result = await runScenario(browser, scenario);
      if (!result.ok) {
        failures.push({ scenario: scenario.id, result });
        console.log(`  FAIL [${result.phase}] ${result.reason || 'unknown reason'}`);
        if (result.status) console.log(`  status: ${result.status}`);
      } else {
        console.log(`  PASS ${formatMetrics(result.metrics)}`);
        console.log(`  status: ${result.status}`);
      }
      console.log();
    }
  } finally {
    await browser.close();
  }

  console.log('==========================================');
  if (failures.length > 0) {
    console.log(`E2E FAILED: ${failures.length}/${SCENARIOS.length} scenario(s) failed`);
    for (const f of failures) {
      console.log(`- ${f.scenario}: [${f.result.phase}] ${f.result.reason}`);
      if (f.result.status) console.log(`  status: ${f.result.status}`);
    }
    process.exit(1);
  }

  console.log(`E2E PASSED: ${SCENARIOS.length}/${SCENARIOS.length} scenarios`);
  process.exit(0);
}

main().catch(err => {
  console.error('Test runner error:', err);
  process.exit(1);
});
