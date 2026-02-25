import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function runTest() {
  const browser = await puppeteer.launch({
    headless: true,
    args: ['--no-sandbox', '--use-gl=angle', '--use-angle=swiftshader'],
    protocolTimeout: 900000,
  });
  const page = await browser.newPage();
  
  page.on('console', msg => console.log('PAGE:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err));

  await page.goto('http://localhost:5176', { waitUntil: 'networkidle0', timeout: 60000 });

  const imageNames = [
    'AXM_1754.jpeg', 'AXM_1755.jpeg', 'AXM_1756.jpeg',
    'AXM_1757.jpeg', 'AXM_1758.jpeg', 'AXM_1759.jpeg',
    'AXM_1760.jpeg', 'AXM_1761.jpeg', 'AXM_1762.jpeg',
  ];

  // All-in-one: upload, wait, click stitch, wait for pipeline
  const result = await page.evaluate(async (names) => {
    // Catch unhandled rejections
    window.addEventListener('unhandledrejection', e => {
      console.error('UNHANDLED:', e.reason?.message || e.reason);
    });

    // 1. Upload images
    try {
      const files = [];
      for (const name of names) {
        const resp = await fetch('/test_images/' + name);
        if (!resp.ok) return 'FAIL: fetch ' + name + ' status ' + resp.status;
        const blob = await resp.blob();
        files.push(new File([blob], name, { type: 'image/jpeg' }));
      }
      console.log('Fetched ' + files.length + ' files');
      const input = document.getElementById('file-input');
      if (!input) return 'FAIL: no file input';
      const dt = new DataTransfer();
      files.forEach(f => dt.items.add(f));
      input.files = dt.files;
      input.dispatchEvent(new Event('change', { bubbles: true }));
    } catch (e) {
      return 'FAIL: upload error: ' + e.message;
    }

    // 2. Wait for stitch button
    for (let i = 0; i < 60; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const btn = document.getElementById('btn-stitch');
      if (btn && !btn.disabled) {
        console.log('Stitch button enabled after ' + (i+1) + 's');
        break;
      }
      if (i === 59) return 'FAIL: stitch button never enabled';
    }

    // 3. Click stitch
    const btn = document.getElementById('btn-stitch');
    btn.click();
    console.log('Clicked stitch');

    // 4. Wait for pipeline
    let lastStatus = '';
    for (let i = 0; i < 600; i++) {
      await new Promise(r => setTimeout(r, 1000));
      const bar = document.getElementById('status-bar');
      const msg = bar?.querySelector('.status-msg');
      const status = (msg || bar)?.textContent || '';
      
      if (status !== lastStatus) {
        console.log('Status: ' + status);
        lastStatus = status;
      }
      
      if (status.includes('Pipeline complete') || status.includes('Composite complete')) {
        return 'SUCCESS: ' + status;
      }
      if (status.includes('Pipeline error') || status.includes('failed')) {
        return 'FAIL: ' + status;
      }
    }
    return 'TIMEOUT: ' + lastStatus;
  }, imageNames);

  console.log('RESULT:', result);
  await browser.close();
  process.exit(result.startsWith('SUCCESS') ? 0 : 1);
}

runTest().catch(err => {
  console.error(err);
  process.exit(1);
});