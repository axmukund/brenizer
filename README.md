# Bokeh Mosaic Stitcher

A single-page, fully client-side Brenizer-method panorama stitcher.  
Upload overlapping photos shot with a wide-aperture lens and get a single large-format image with impossibly shallow depth of field — all processed in the browser.

## Features

- **ORB feature extraction** and brute-force kNN matching via OpenCV.js (WASM)
- **RANSAC homography** estimation with MST-based global alignment
- **APAP local mesh warping** (As-Projective-As-Possible) for parallax reduction
- **Exposure compensation** — per-image scalar gain solved via luminance sampling
- **Graph-cut seam finding** (Edmonds-Karp maxflow on coarse block grid)
- **Laplacian pyramid multi-band blending** (GPU-accelerated)
- **Depth estimation** — optional ONNX Runtime MiDaS model for depth-guided seam costs
- **Export** with auto-crop at configurable scale (PNG / JPEG)
- Adaptive quality modes: Desktop HQ, Mobile Quality, Mobile Safe
- Built-in Capture Guide and Connectivity Diagnostics

## Supported Browsers

| Browser | Support |
|---------|---------|
| Chrome / Edge (desktop) | ✅ Recommended — full WebGL2 + WebGPU EP for depth |
| Firefox (desktop) | ✅ WebGL2, no WebGPU EP |
| Safari 15+ (desktop) | ✅ WebGL2 |
| Chrome / Safari (mobile) | ⚠ Works in Mobile Safe mode; may degrade resolution |

**File types:** JPG and PNG only.

## How to Capture Brenizer Sets

1. Use a fast lens (50mm f/1.4, 85mm f/1.8, etc.) wide open
2. **Lock focus** — manual focus or AF-lock on your subject
3. **Lock exposure** — manual or AE-lock
4. **Rotate from one spot** — pivot, don't step sideways
5. **Overlap 30–50%** between adjacent frames
6. Shoot a 3×3, 4×4, or 5×5 grid (9–25 frames)
7. Shoot quickly to minimize subject movement

## Local Development

```bash
npm install
npm run dev
```

Open http://localhost:5173 in Chrome.

### Prerequisites

The OpenCV.js WASM build must be placed at `public/opencv/opencv.js`.  
Download from [@techstark/opencv-js](https://www.npmjs.com/package/@techstark/opencv-js) v4.12:

```bash
cp node_modules/@techstark/opencv-js/dist/opencv.js public/opencv/opencv.js
```

## Build

```bash
npm run build
```

Output goes to `dist/`.

## Deploy to GitHub Pages

The Vite config supports subpath deployment via the `GH_PAGES_BASE` env variable:

```bash
GH_PAGES_BASE=/your-repo-name/ npm run build
```

### GitHub Actions (optional)

```yaml
name: Deploy
on:
  push:
    branches: [main]
jobs:
  deploy:
    runs-on: ubuntu-latest
    permissions:
      pages: write
      id-token: write
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20 }
      - run: npm ci
      - run: cp node_modules/@techstark/opencv-js/dist/opencv.js public/opencv/opencv.js
      - run: GH_PAGES_BASE=/${{ github.event.repository.name }}/ npm run build
      - uses: actions/upload-pages-artifact@v3
        with: { path: dist }
      - uses: actions/deploy-pages@v4
```

## Known Limitations

- **Large image sets** (>20 images at high resolution) may exceed GPU memory on some devices
- **WebGPU depth estimation** requires Chrome 113+ with WebGPU enabled
- OpenCV.js WASM file is ~11 MB (loaded on first use)
- Mobile browsers may limit texture sizes to 4096×4096
- Featureless regions (sky, plain walls) may fail to match — ensure textured overlap
- Moving subjects cause ghosting at seam boundaries

## Architecture

```
src/
  main.ts              — entry point, boot, rendering, export
  appState.ts          — reactive state store
  capabilities.ts      — browser feature detection
  presets.ts           — quality mode presets
  ui.ts                — DOM event wiring
  pipelineController.ts — orchestrates worker pipeline
  gl/                  — WebGL2 renderers, shaders, FBO management
  workers/
    workerManager.ts   — worker lifecycle + messaging
    workerTypes.ts     — TypeScript message types
    depth.worker.ts    — ONNX Runtime depth inference
public/
  workers/
    cv-worker.js       — OpenCV.js feature extraction + matching
    seam-worker.js     — graph-cut maxflow solver
  opencv/
    opencv.js          — OpenCV.js WASM build (gitignored)
```
