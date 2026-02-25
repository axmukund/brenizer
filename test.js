import puppeteer from 'puppeteer';
import path from 'path';
import { fileURLToPath } from 'url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

async function runTest() {
  const browser = await puppeteer.launch({ headless: 'new' });
  const page = await browser.newPage();
  
  // Listen to console logs
  page.on('console', msg => console.log('PAGE LOG:', msg.text()));
  page.on('pageerror', err => console.error('PAGE ERROR:', err));

  await page.goto('http://localhost:5175');

  // Upload images
  const fileInput = await page.$('#file-input');
  const images = [
    'AXM_1754.jpeg', 'AXM_1755.jpeg', 'AXM_1756.jpeg',
    'AXM_1757.jpeg', 'AXM_1758.jpeg', 'AXM_1759.jpeg',
    'AXM_1760.jpeg', 'AXM_1761.jpeg', 'AXM_1762.jpeg'
  ].map(f => path.join(__dirname, 'test_images/nsv_test', f));
  await fileInput.uploadFile(...images);

  // Wait for images to load
  await page.waitForSelector('#btn-stitch:not([disabled])');

  // Click stitch
  await page.click('#btn-stitch');

  // Wait for pipeline to finish
  try {
    await page.waitForFunction(() => {
      const status = document.getElementById('status-bar').textContent;
      return status.includes('Pipeline complete') || status.includes('Composite complete') || status.includes('failed') || status.includes('error');
    }, { timeout: 60000 });
    
    const finalStatus = await page.$eval('#status-bar', el => el.textContent);
    console.log('Final status:', finalStatus);
  } catch (e) {
    console.error('Timeout waiting for pipeline to finish');
  }

  await browser.close();
}

runTest().catch(console.error);