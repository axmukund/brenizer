/**
 * test.js — Headless E2E test for the Brenizer image stitching pipeline.
 *
 * Uses Puppeteer to:
 *  1. Launch a headless Chrome with software-rendered WebGL (SwiftShader)
 *  2. Upload 9 overlapping tiles (3×3 grid, 30% overlap) to the stitcher
 *  3. Click "Stitch Preview" and wait for the pipeline to finish
 *  4. Verify the composite was produced successfully
 *
 * Tile naming convention: tile30_r{row}_c{col}.png
 *   - "30" = 30% overlap between adjacent tiles
 *   - row 1–3, col 1–3, each tile 256×192 px (PNG, 8-bit RGB)
 *   - The ground-truth panorama is 1024×768 JPEG
 *
 * Usage:
 *   1. Start the Vite dev server:  npx vite --port 5176
 *   2. Run the test:               node test.js
 *
 * Exit codes:
 *   0 = success (pipeline completed)
 *   1 = failure (pipeline error, timeout, or upload failure)
 */
import puppeteer from 'puppeteer';

/** URL of the Vite dev server serving the stitcher app */
const APP_URL = 'http://localhost:5176';

/**
 * Landscape test tiles: 3×3 grid of 256×192 PNG crops with 30% overlap.
 * Row-major order (top-left → bottom-right) — matches the natural reading
 * order and is consistent with how images[] gets iterated for MST.
 */
const TILE_NAMES = [
  'lnd_test/tile30_r1_c1.png', 'lnd_test/tile30_r1_c2.png', 'lnd_test/tile30_r1_c3.png',
  'lnd_test/tile30_r2_c1.png', 'lnd_test/tile30_r2_c2.png', 'lnd_test/tile30_r2_c3.png',
  'lnd_test/tile30_r3_c1.png', 'lnd_test/tile30_r3_c2.png', 'lnd_test/tile30_r3_c3.png',
];

/**
 * Maximum seconds to wait at each major stage.
 * Headless SwiftShader is slower than GPU, so timeouts are generous.
 */
const TIMEOUTS = {
  pageLoad:    60_000,   // ms — wait for full page load (WASM + OpenCV)
  stitchBtn:   120,      // seconds — wait for Stitch button to become enabled
  pipeline:    600,      // seconds — full pipeline (features → stitch → composite)
  protocol:    900_000,  // ms — Puppeteer CDP protocol timeout (entire run)
};

async function runTest() {
  console.log('--- Brenizer E2E test: 3×3 landscape tiles ---');
  console.log(`  Tiles: ${TILE_NAMES.length} images`);
  console.log(`  Server: ${APP_URL}`);
  console.log();

  // ── Launch browser ────────────────────────────────────────────────
  const browser = await puppeteer.launch({
    headless: true,
    args: [
      '--no-sandbox',
      // Force software WebGL via SwiftShader so test works without a GPU
      '--use-gl=angle',
      '--use-angle=swiftshader',
    ],
    protocolTimeout: TIMEOUTS.protocol,
  });
  const page = await browser.newPage();

  // Forward console/error from the page to Node stdout for debugging
  page.on('console', msg => console.log('PAGE:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err));

  // ── Navigate to app ───────────────────────────────────────────────
  console.log('Loading app…');
  await page.goto(APP_URL, {
    waitUntil: 'networkidle0',
    timeout: TIMEOUTS.pageLoad,
  });

  // ── Upload tiles and run pipeline inside page context ─────────────
  const result = await page.evaluate(async (tileNames, timeouts) => {
    // Catch unhandled promise rejections so they don't silently swallow errors
    window.addEventListener('unhandledrejection', e => {
      console.error('UNHANDLED REJECTION:', e.reason?.message || e.reason);
    });

    // ── 1. Fetch tile images from the static server ─────────────────
    try {
      const files = [];
      for (const name of tileNames) {
        const resp = await fetch('/test_images/' + name);
        if (!resp.ok) {
          return `FAIL: fetch ${name} → HTTP ${resp.status}`;
        }
        const blob = await resp.blob();
        // Use just the filename (without the directory prefix) for the File object
        const shortName = name.split('/').pop();
        files.push(new File([blob], shortName, { type: 'image/png' }));
      }
      console.log(`Fetched ${files.length} tile files OK`);

      // ── 2. Inject files into the <input type="file"> element ──────
      const input = document.getElementById('file-input');
      if (!input) return 'FAIL: no #file-input element found';

      const dt = new DataTransfer();
      files.forEach(f => dt.items.add(f));
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
      console.log('Files injected into file input');
    } catch (e) {
      return `FAIL: upload error: ${e.message}`;
    }

    // ── 3. Wait for the Stitch button to become enabled ─────────────
    // (Images need to be decoded / thumbnailed before the button unlocks)
    for (let i = 0; i < timeouts.stitchBtn; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const btn = document.getElementById('btn-stitch');
      if (btn && !btn.disabled) {
        console.log(`Stitch button enabled after ${i + 1}s`);
        break;
      }
      if (i === timeouts.stitchBtn - 1) {
        return 'FAIL: stitch button never became enabled';
      }
    }

    // ── 4. Click Stitch ─────────────────────────────────────────────
    const btn = document.getElementById('btn-stitch');
    btn.click();
    console.log('Clicked Stitch button');

    // ── 5. Poll the status bar for pipeline completion ───────────────
    let lastStatus = '';
    for (let i = 0; i < timeouts.pipeline; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const bar = document.getElementById('status-bar');
      const msg = bar?.querySelector('.status-msg');
      const status = (msg || bar)?.textContent || '';

      // Log every status change so we can follow progress
      if (status !== lastStatus) {
        console.log('Status: ' + status);
        lastStatus = status;
      }

      // Pipeline success signals
      if (status.includes('Pipeline complete') || status.includes('Composite complete')) {
        return 'SUCCESS: ' + status;
      }
      // Pipeline failure signals
      if (status.includes('Pipeline error') || status.includes('failed') ||
          status.includes('No matching pairs')) {
        return 'FAIL: ' + status;
      }
    }
    return 'TIMEOUT: last status = ' + lastStatus;
  }, TILE_NAMES, TIMEOUTS);

  // ── Report result ─────────────────────────────────────────────────
  console.log();
  console.log('═══════════════════════════════════════════');
  console.log('RESULT:', result);
  console.log('═══════════════════════════════════════════');

  await browser.close();
  process.exit(result.startsWith('SUCCESS') ? 0 : 1);
}

runTest().catch(err => {
  console.error('Test runner error:', err);
  process.exit(1);
});