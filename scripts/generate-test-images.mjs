#!/usr/bin/env node
import fs from 'node:fs';
import path from 'node:path';
import zlib from 'node:zlib';

const ROOT = process.cwd();
const TEST_ROOT = path.join(ROOT, 'public', 'test_images');

const SETS = [
  {
    name: 'synth_orchard_3x3',
    seed: 1337,
    rows: 3,
    cols: 3,
    tileW: 512,
    tileH: 384,
    overlap: 0.35,
  },
  {
    name: 'synth_market_4x3',
    seed: 9001,
    rows: 3,
    cols: 4,
    tileW: 448,
    tileH: 320,
    overlap: 0.30,
  },
];

function mulberry32(seed) {
  let t = seed >>> 0;
  return function rng() {
    t += 0x6D2B79F5;
    let r = Math.imul(t ^ (t >>> 15), 1 | t);
    r ^= r + Math.imul(r ^ (r >>> 7), 61 | r);
    return ((r ^ (r >>> 14)) >>> 0) / 4294967296;
  };
}

function clamp8(v) {
  if (v <= 0) return 0;
  if (v >= 255) return 255;
  return v | 0;
}

function makeCrcTable() {
  const table = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) {
      c = (c & 1) ? (0xEDB88320 ^ (c >>> 1)) : (c >>> 1);
    }
    table[n] = c >>> 0;
  }
  return table;
}

const CRC_TABLE = makeCrcTable();

function crc32(buf) {
  let c = 0xFFFFFFFF;
  for (let i = 0; i < buf.length; i++) {
    c = CRC_TABLE[(c ^ buf[i]) & 0xFF] ^ (c >>> 8);
  }
  return (c ^ 0xFFFFFFFF) >>> 0;
}

function pngChunk(type, data) {
  const typeBuf = Buffer.from(type, 'ascii');
  const lenBuf = Buffer.alloc(4);
  lenBuf.writeUInt32BE(data.length, 0);
  const crcBuf = Buffer.alloc(4);
  const crc = crc32(Buffer.concat([typeBuf, data]));
  crcBuf.writeUInt32BE(crc, 0);
  return Buffer.concat([lenBuf, typeBuf, data, crcBuf]);
}

function encodePng(width, height, rgba) {
  const signature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);

  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(width, 0);
  ihdr.writeUInt32BE(height, 4);
  ihdr[8] = 8;  // bit depth
  ihdr[9] = 6;  // color type RGBA
  ihdr[10] = 0; // compression
  ihdr[11] = 0; // filter
  ihdr[12] = 0; // interlace

  const stride = width * 4;
  const raw = Buffer.alloc((stride + 1) * height);
  for (let y = 0; y < height; y++) {
    const rowStart = y * (stride + 1);
    raw[rowStart] = 0; // filter method 0
    const srcStart = y * stride;
    for (let x = 0; x < stride; x++) {
      raw[rowStart + 1 + x] = rgba[srcStart + x];
    }
  }

  const idat = zlib.deflateSync(raw, { level: 9 });
  return Buffer.concat([
    signature,
    pngChunk('IHDR', ihdr),
    pngChunk('IDAT', idat),
    pngChunk('IEND', Buffer.alloc(0)),
  ]);
}

function hashNoise(x, y, seed) {
  const n = Math.sin((x + seed * 0.13) * 12.9898 + (y - seed * 0.21) * 78.233) * 43758.5453;
  return n - Math.floor(n);
}

function drawPixel(buf, w, h, x, y, r, g, b, a = 255, alpha = 1) {
  if (x < 0 || y < 0 || x >= w || y >= h) return;
  const i = (y * w + x) * 4;
  const inv = 1 - alpha;
  buf[i] = clamp8(buf[i] * inv + r * alpha);
  buf[i + 1] = clamp8(buf[i + 1] * inv + g * alpha);
  buf[i + 2] = clamp8(buf[i + 2] * inv + b * alpha);
  buf[i + 3] = clamp8(buf[i + 3] * inv + a * alpha);
}

function drawRect(buf, w, h, x0, y0, rw, rh, color, alpha = 1) {
  const x1 = Math.min(w, Math.max(0, x0 + rw));
  const y1 = Math.min(h, Math.max(0, y0 + rh));
  const sx = Math.max(0, x0);
  const sy = Math.max(0, y0);
  for (let y = sy; y < y1; y++) {
    for (let x = sx; x < x1; x++) {
      drawPixel(buf, w, h, x, y, color[0], color[1], color[2], 255, alpha);
    }
  }
}

function drawCircle(buf, w, h, cx, cy, radius, color, alpha = 1) {
  const r2 = radius * radius;
  const x0 = Math.max(0, Math.floor(cx - radius));
  const y0 = Math.max(0, Math.floor(cy - radius));
  const x1 = Math.min(w - 1, Math.ceil(cx + radius));
  const y1 = Math.min(h - 1, Math.ceil(cy + radius));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const d2 = dx * dx + dy * dy;
      if (d2 > r2) continue;
      const edge = Math.max(0, Math.min(1, (radius - Math.sqrt(d2)) / Math.max(1, radius * 0.2)));
      drawPixel(buf, w, h, x, y, color[0], color[1], color[2], 255, alpha * (0.4 + 0.6 * edge));
    }
  }
}

function pointToSegmentDistance(px, py, ax, ay, bx, by) {
  const abx = bx - ax;
  const aby = by - ay;
  const apx = px - ax;
  const apy = py - ay;
  const ab2 = abx * abx + aby * aby;
  if (ab2 < 1e-6) return Math.hypot(px - ax, py - ay);
  let t = (apx * abx + apy * aby) / ab2;
  if (t < 0) t = 0;
  if (t > 1) t = 1;
  const qx = ax + abx * t;
  const qy = ay + aby * t;
  return Math.hypot(px - qx, py - qy);
}

function drawLine(buf, w, h, ax, ay, bx, by, thickness, color, alpha = 1) {
  const pad = thickness + 2;
  const x0 = Math.max(0, Math.floor(Math.min(ax, bx) - pad));
  const y0 = Math.max(0, Math.floor(Math.min(ay, by) - pad));
  const x1 = Math.min(w - 1, Math.ceil(Math.max(ax, bx) + pad));
  const y1 = Math.min(h - 1, Math.ceil(Math.max(ay, by) + pad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const d = pointToSegmentDistance(x + 0.5, y + 0.5, ax, ay, bx, by);
      if (d > thickness) continue;
      const f = Math.max(0, Math.min(1, 1 - d / Math.max(1, thickness)));
      drawPixel(buf, w, h, x, y, color[0], color[1], color[2], 255, alpha * (0.35 + 0.65 * f));
    }
  }
}

function createPanorama(width, height, seed) {
  const rng = mulberry32(seed);
  const rgba = new Uint8Array(width * height * 4);
  const twoPi = Math.PI * 2;

  // Base field: layered smooth gradients and procedural detail.
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const nx = x / Math.max(1, width - 1);
      const ny = y / Math.max(1, height - 1);
      const radialX = nx - 0.5;
      const radialY = ny - 0.5;
      const radial = Math.sqrt(radialX * radialX + radialY * radialY);

      const wave1 = Math.sin(twoPi * (1.4 * nx + 0.9 * ny) + seed * 0.013);
      const wave2 = Math.cos(twoPi * (0.5 * nx - 1.1 * ny) + seed * 0.021);
      const wave3 = Math.sin(twoPi * (4.5 * nx + 3.2 * ny) + seed * 0.007);
      const n0 = hashNoise(x * 0.9, y * 0.9, seed);
      const n1 = hashNoise(x * 2.3, y * 2.1, seed + 11);
      const n2 = hashNoise(x * 4.8, y * 4.5, seed + 37);

      let r = 120 + 48 * wave1 + 32 * wave2 + 22 * wave3 + 25 * (n1 - 0.5);
      let g = 108 + 38 * wave2 - 26 * wave3 + 35 * wave1 + 22 * (n0 - 0.5);
      let b = 98 + 30 * wave1 - 20 * wave2 + 16 * wave3 + 30 * (n2 - 0.5);

      // Soft vignette to resemble real captures.
      const vignette = 1 - 0.30 * radial * radial;
      r *= vignette;
      g *= vignette;
      b *= vignette;

      const i = (y * width + x) * 4;
      rgba[i] = clamp8(r);
      rgba[i + 1] = clamp8(g);
      rgba[i + 2] = clamp8(b);
      rgba[i + 3] = 255;
    }
  }

  // Structural overlays to create stable keypoints and repeated perspective cues.
  for (let i = 0; i < 30; i++) {
    const rw = Math.floor(width * (0.06 + rng() * 0.18));
    const rh = Math.floor(height * (0.05 + rng() * 0.16));
    const x = Math.floor((width - rw) * rng());
    const y = Math.floor((height - rh) * rng());
    const color = [
      clamp8(45 + rng() * 190),
      clamp8(45 + rng() * 190),
      clamp8(45 + rng() * 190),
    ];
    drawRect(rgba, width, height, x, y, rw, rh, color, 0.35 + rng() * 0.35);
  }

  for (let i = 0; i < 45; i++) {
    const radius = Math.floor(Math.min(width, height) * (0.015 + rng() * 0.06));
    const cx = Math.floor(rng() * width);
    const cy = Math.floor(rng() * height);
    const color = [
      clamp8(50 + rng() * 205),
      clamp8(50 + rng() * 205),
      clamp8(50 + rng() * 205),
    ];
    drawCircle(rgba, width, height, cx, cy, radius, color, 0.4 + rng() * 0.35);
  }

  for (let i = 0; i < 60; i++) {
    const ax = rng() * width;
    const ay = rng() * height;
    const bx = rng() * width;
    const by = rng() * height;
    const thickness = 1 + rng() * 3;
    const color = [
      clamp8(70 + rng() * 170),
      clamp8(70 + rng() * 170),
      clamp8(70 + rng() * 170),
    ];
    drawLine(rgba, width, height, ax, ay, bx, by, thickness, color, 0.45 + rng() * 0.35);
  }

  // Add a dense corner grid in selected regions to guarantee hard-feature areas.
  const gridBlockW = Math.floor(width * 0.22);
  const gridBlockH = Math.floor(height * 0.20);
  const gx0 = Math.floor(width * 0.08);
  const gy0 = Math.floor(height * 0.10);
  for (let y = gy0; y < gy0 + gridBlockH; y += 12) {
    drawLine(rgba, width, height, gx0, y, gx0 + gridBlockW, y, 1.2, [230, 230, 230], 0.55);
  }
  for (let x = gx0; x < gx0 + gridBlockW; x += 12) {
    drawLine(rgba, width, height, x, gy0, x, gy0 + gridBlockH, 1.2, [210, 210, 210], 0.55);
  }

  return rgba;
}

function cropTile(panorama, panoW, panoH, x0, y0, tileW, tileH, seed) {
  const rng = mulberry32(seed);
  const out = new Uint8Array(tileW * tileH * 4);
  const gain = 0.96 + rng() * 0.10;
  const gainR = gain * (0.98 + rng() * 0.05);
  const gainG = gain * (0.98 + rng() * 0.05);
  const gainB = gain * (0.98 + rng() * 0.05);
  const vignette = 0.08 + rng() * 0.08;

  for (let y = 0; y < tileH; y++) {
    for (let x = 0; x < tileW; x++) {
      const sx = x0 + x;
      const sy = y0 + y;
      const si = (sy * panoW + sx) * 4;
      const di = (y * tileW + x) * 4;

      const nx = (x / Math.max(1, tileW - 1)) - 0.5;
      const ny = (y / Math.max(1, tileH - 1)) - 0.5;
      const radial = nx * nx + ny * ny;
      const v = 1 - vignette * radial;

      out[di] = clamp8(panorama[si] * gainR * v);
      out[di + 1] = clamp8(panorama[si + 1] * gainG * v);
      out[di + 2] = clamp8(panorama[si + 2] * gainB * v);
      out[di + 3] = 255;
    }
  }

  return out;
}

function buildTileSet(cfg) {
  const stepX = Math.round(cfg.tileW * (1 - cfg.overlap));
  const stepY = Math.round(cfg.tileH * (1 - cfg.overlap));
  const panoW = cfg.tileW + (cfg.cols - 1) * stepX;
  const panoH = cfg.tileH + (cfg.rows - 1) * stepY;

  const setDir = path.join(TEST_ROOT, cfg.name);
  fs.rmSync(setDir, { recursive: true, force: true });
  fs.mkdirSync(setDir, { recursive: true });

  const panorama = createPanorama(panoW, panoH, cfg.seed);
  const panoPng = encodePng(panoW, panoH, panorama);
  fs.writeFileSync(path.join(setDir, 'panorama.png'), panoPng);

  const files = [];
  for (let r = 0; r < cfg.rows; r++) {
    for (let c = 0; c < cfg.cols; c++) {
      const x0 = c * stepX;
      const y0 = r * stepY;
      const tile = cropTile(
        panorama,
        panoW,
        panoH,
        x0,
        y0,
        cfg.tileW,
        cfg.tileH,
        cfg.seed + r * 97 + c * 131,
      );
      const name = `tile_r${r + 1}_c${c + 1}.png`;
      const tilePng = encodePng(cfg.tileW, cfg.tileH, tile);
      fs.writeFileSync(path.join(setDir, name), tilePng);
      files.push(name);
    }
  }

  const manifest = {
    generatedBy: 'scripts/generate-test-images.mjs',
    license: 'CC0-1.0',
    deterministic: true,
    set: cfg.name,
    seed: cfg.seed,
    rows: cfg.rows,
    cols: cfg.cols,
    tileWidth: cfg.tileW,
    tileHeight: cfg.tileH,
    overlap: cfg.overlap,
    panorama: { width: panoW, height: panoH },
    tiles: files,
  };
  fs.writeFileSync(
    path.join(setDir, 'manifest.json'),
    JSON.stringify(manifest, null, 2) + '\n',
    'utf8',
  );

  console.log(
    `Generated ${cfg.name}: ${cfg.rows}x${cfg.cols} tiles ` +
    `(${cfg.tileW}x${cfg.tileH}, overlap ${(cfg.overlap * 100).toFixed(0)}%), panorama ${panoW}x${panoH}`,
  );
}

function writeLicenseNote() {
  const notePath = path.join(TEST_ROOT, 'LICENSE.md');
  const body = [
    '# Test Image Licensing',
    '',
    '## Synthetic Sets (license-safe)',
    '- `synth_orchard_3x3/*` and `synth_market_4x3/*` are procedurally generated in this repository.',
    '- Generated by `scripts/generate-test-images.mjs` with deterministic seeds.',
    '- These generated files are dedicated to the public domain under **CC0-1.0**.',
    '',
    '## Legacy Sets',
    '- `lnd_test/*` and `AXM_*.jpeg` are retained for historical regression coverage.',
    '- Verify distribution rights before reusing legacy files outside this repository.',
    '',
  ].join('\n');
  fs.writeFileSync(notePath, body, 'utf8');
}

function main() {
  fs.mkdirSync(TEST_ROOT, { recursive: true });
  for (const set of SETS) {
    buildTileSet(set);
  }
  writeLicenseNote();
}

main();
