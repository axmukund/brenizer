# Bokeh Mosaic Stitcher

A single-page, fully client-side Brenizer-method panorama stitcher.  
Upload overlapping photos shot with a wide-aperture lens and get a single large-format image with impossibly shallow depth of field — all processed in the browser.

---

## Features

### Core Pipeline
- **ORB feature extraction** and brute-force kNN matching
- **MAGSAC++ robust estimation** for inlier classification
- **Levenberg–Marquardt bundle adjustment** for global refinement
- **APAP local mesh warping** (As-Projective-As-Possible) for parallax reduction
- **Graph-cut seam finding** (Edmonds–Karp max-flow on coarse block grid)
- **Laplacian pyramid multi-band blending** (GPU-accelerated)
- **Depth estimation** — optional ONNX Runtime MiDaS model

### AI / Saliency-Aware Compositing
- **Itti–Koch–Niebur saliency model** with Achanta frequency-tuned colour distinctness
- **Blur-aware seam placement** — seams routed through bokeh regions
- **Face-aware graph cut** — faces detected and protected from seam cuts

### Exposure & Vignetting
- **IRLS Huber-robust exposure compensation** — per-channel RGB gains
- **Reinhard HDR tone mapping** — auto-activated for extreme gain ratios
- **Polynomial vignetting correction** — PTGui-style radial model

### PTGui-Style Algorithms
- **Cylindrical projection** for wide-angle mosaics
- **Brown–Conrady lens distortion** estimation
- **Adaptive feathering** based on overlap width

### Export
- **Max resolution export** at full sensor resolution
- **PNG / JPEG** with auto-crop and configurable scale

---

## Algorithms & Mathematics

### 1. Feature Extraction — Oriented FAST + Rotated BRIEF (ORB)

ORB detects corner keypoints using the FAST-9 detector and computes 256-bit binary
descriptors based on intensity comparisons. Each descriptor bit is:

$$d_i = \begin{cases} 1 & \text{if } I(p_i) < I(q_i) \\\ 0 & \text{otherwise} \end{cases}$$

where $(p_i, q_i)$ are learned binary test pairs rotated by the keypoint orientation
$\theta = \text{atan2}(m_{01}, m_{10})$ computed from image moments.

Matching uses Hamming distance on descriptor pairs with Lowe's ratio test ($r < 0.8$).

### 2. Robust Estimation — MAGSAC++

MAGSAC++ (Barath et al., CVPR 2020) replaces the hard inlier/outlier threshold with
a marginalised model across a continuous range of noise scales $\sigma \in [\sigma_{\min}, \sigma_{\max}]$:

$$w_i = \int_{\sigma_{\min}}^{\sigma_{\max}} \frac{1}{\sigma} \exp\left(-\frac{r_i^2}{2\sigma^2}\right) d\sigma$$

where $r_i$ is the symmetric transfer error for correspondence $i$. The quality score
of a model is the sum of these marginalised weights: $Q(H) = \sum_{i=1}^{N} w_i$.

The implementation uses $\sigma_{\min} = 0.5\text{px}$ and $\sigma_{\max} = 10\text{px}$ with
iterative reweighting for the final homography.

### 3. Bundle Adjustment — Levenberg–Marquardt

After MST-based global alignment, all homographies are jointly refined by minimising
the reprojection error across all matched pairs:

$$\min_{\{H_k\}} \sum_{(i,j)} \sum_{(p,q) \in \mathcal{M}_{ij}} \rho\bigl(\|H_j^{-1} H_i\, \tilde{p} - \tilde{q}\|^2\bigr)$$

where $\tilde{p}$ denotes homogeneous coordinates and $\rho$ is the Huber loss.

Each LM iteration solves the damped normal equations:

$$(J^\top J + \lambda \operatorname{diag}(J^\top J)) \, \delta = -J^\top r$$

NaN protection is applied to the delta vector to prevent degenerate numeric solves.

### 4. APAP Local Mesh Warping

As-Projective-As-Possible warping (Zaragoza et al., CVPR 2013) partitions the image
into a regular mesh and computes a local homography $H^*_i$ at each vertex $v_i$ using
Tikhonov-regularised DLT:

$$H^*_i = \arg\min_H \sum_{k} w_k(v_i) \| A_k H \|^2 + \gamma \| H - H_{\text{global}} \|^2$$

The spatial weights follow Gaussian decay:

$$w_k(v) = \exp\left(-\frac{\|v - x_k\|^2}{2\sigma^2}\right)$$

where $\sigma = 100$ pixels (alignment scale). The regularisation parameter $\gamma$
controls how much outlier vertices fall back to the global homography.

### 5. AI Saliency Detection

The saliency map combines three complementary channels:

$$S = 0.4 \cdot S_{\text{grad}} + 0.3 \cdot S_{\text{color}} + 0.3 \cdot S_{\text{focus}}$$

- **Gradient magnitude** (Sobel): $S_{\text{grad}} = \sqrt{G_x^2 + G_y^2}$ normalised to $[0,1]$
- **Frequency-tuned colour distinctness** (Achanta et al., CVPR 2009): $S_{\text{color}}(x,y) = \| I_\mu - I_\omega(x,y) \|$ where $I_\mu$ is the mean image colour and $I_\omega$ is a Gaussian-blurred version.
- **Focus measure** (Laplacian variance): $S_{\text{focus}} = \text{Var}(\nabla^2 I)$ in a $7 \times 7$ window

### 6. IRLS Huber-Robust Exposure Compensation

Per-channel gains $g_{i,c}$ are estimated from overlap luminance ratios using
Iteratively Reweighted Least Squares with Huber loss:

$$\min_{\{g_{i,c}\}} \sum_{(i,j)} \sum_{c \in \{R,G,B\}} w_{ij} \bigl(\log g_{j,c} - \log g_{i,c} - r_{ij,c}\bigr)^2$$

where $r_{ij,c}$ is the median log-ratio in channel $c$. The Huber weights are:

$$w_{ij} = \begin{cases} 1 & \text{if } |e| \leq \delta \\\ \delta / |e| & \text{otherwise} \end{cases}$$

with $\delta = 0.5$ log-units. Gains are clamped to $[0.05, 20]$.

### 7. Reinhard HDR Tone Mapping

When gain ratios exceed $2\times$ or fall below $0.5\times$, the Reinhard global operator
is automatically applied in the warp shader:

$$L_d = \frac{L \left(1 + L / L_{\text{white}}^2\right)}{1 + L}$$

where $L$ is the scene luminance and $L_{\text{white}} = 4.0$ is the smallest luminance
that maps to pure white.

### 8. Polynomial Vignetting Correction

A PTGui-style radial vignetting model is estimated by binning luminance by radial
distance from the image centre (10 bins, least-squares fit):

$$V(r) = 1 + ar^2 + br^4$$

where $r = \sqrt{(x - c_x)^2 + (y - c_y)^2} / r_{\max}$ is the normalised radial
coordinate. Correction divides by $V(r)$ in the GPU warp shader.

### 9. Graph-Cut Seam Finding

Overlap regions are divided into a coarse block grid. Each block is a graph node with
binary label: *keep composite* (0) or *take new image* (1).

**Data costs** combine distance-from-boundary and colour difference:

$$D_s(\ell) = 0.8 \cdot (1 - d_\ell(s)) + 0.2 \cdot \Delta_{\text{color}}(s)$$

**Edge weights** use gradient-domain seam energy (inspired by Perez et al., SIGGRAPH 2003):

$$w_{st} = \bigl(0.4 \cdot E_{\text{strength}} + 0.6 \cdot G_{\text{agreement}}\bigr) \times B_{\text{discount}}$$

where $G_{\text{agreement}} = 1 - |\nabla I_{\text{comp}} - \nabla I_{\text{new}}| / 255$ and
the blur discount encourages seams through bokeh:

$$B_{\text{discount}} = 0.2 + 0.8 \cdot s(x,y)$$

where $s$ is the saliency value ($0 =$ blurred, $1 =$ sharp).

The min-cut is solved via Edmonds–Karp (BFS augmenting paths) in a Web Worker.

### 10. Laplacian Pyramid Multi-Band Blending

Each image is decomposed into a Laplacian pyramid:

$$L_k = G_k - \text{upsample}(G_{k+1})$$

Low-frequency bands are blended with wide feathering, high-frequency bands with narrow
feathering. Reconstruction: $I_{\text{blend}} = \sum_{k=0}^{K} L_k^{\text{blend}}$.

Default levels: $K = 4$ (desktop HQ), $K = 2$ (mobile safe).

### 11. Cylindrical Projection

For wide-angle mosaics, coordinates are projected onto a cylindrical surface:

$$x_{\text{cyl}} = f \cdot \arctan\frac{x - c_x}{f}, \quad y_{\text{cyl}} = f \cdot \frac{y - c_y}{\sqrt{(x-c_x)^2 + f^2}}$$

where $f$ is the estimated focal length and $(c_x, c_y)$ is the principal point.

### 12. Brown–Conrady Lens Distortion

Radial distortion coefficients are estimated from matched point residuals:

$$r_d = r\,(1 + k_1 r^2 + k_2 r^4)$$

where $r = \sqrt{x^2 + y^2}$ and $(k_1, k_2)$ are fit via least-squares.

---

## Supported Browsers

| Browser | Support |
|---------|---------|
| Chrome / Edge (desktop) | Recommended — full WebGL2 + WebGPU EP for depth |
| Firefox (desktop) | WebGL2, no WebGPU EP |
| Safari 15+ (desktop) | WebGL2 |
| Chrome / Safari (mobile) | Works in Mobile Safe mode; may degrade resolution |

**File types:** JPG, PNG, HEIC, HEIF, DNG.

## How to Capture Brenizer Sets

1. Use a fast lens (50mm f/1.4, 85mm f/1.8, etc.) wide open
2. **Lock focus** — manual focus or AF-lock on your subject
3. **Lock exposure** — manual or AE-lock
4. **Rotate from one spot** — pivot, don't step sideways
5. **Overlap 30–50%** between adjacent frames
6. Shoot a 3x3, 4x4, or 5x5 grid (9–25 frames)
7. Shoot quickly to minimise subject movement

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
- Mobile browsers may limit texture sizes to 4096x4096
- Featureless regions (sky, plain walls) may fail to match — ensure textured overlap
- Moving subjects cause ghosting at seam boundaries

## Architecture

```
src/
  main.ts              — entry point, boot, rendering, export
  appState.ts          — reactive state store
  capabilities.ts      — browser feature detection
  presets.ts           — quality mode presets
  ui.ts                — DOM event wiring, settings panel
  pipelineController.ts — orchestrates worker pipeline
  gl/
    composition.ts     — graph-cut seam costs, gradient-domain energy
    mesh.ts            — APAP mesh warp shader, vignette/HDR correction
    pyramid.ts         — Laplacian pyramid multi-band blending
    programs.ts        — WebGL2 shader compilation utilities
    index.ts           — GL module barrel export
  workers/
    workerManager.ts   — worker lifecycle + messaging
    workerTypes.ts     — TypeScript message types
    depth.worker.ts    — ONNX Runtime depth inference
public/
  workers/
    cv-worker.js       — ORB, MAGSAC++, LM, APAP, saliency, vignetting
    seam-worker.js     — graph-cut maxflow solver
  opencv/
    opencv.js          — OpenCV.js WASM build (gitignored)
```

## References

- Rublee et al., "ORB: An efficient alternative to SIFT or SURF," ICCV 2011
- Barath et al., "MAGSAC++: A fast, reliable and accurate robust estimator," CVPR 2020
- Zaragoza et al., "As-Projective-As-Possible Image Stitching with Moving DLT," CVPR 2013
- Achanta et al., "Frequency-tuned Salient Region Detection," CVPR 2009
- Itti, Koch & Niebur, "A Model of Saliency-Based Visual Attention," PAMI 1998
- Reinhard et al., "Photographic Tone Reproduction for Digital Images," SIGGRAPH 2002
- Perez et al., "Poisson Image Editing," SIGGRAPH 2003
- Brown & Lowe, "Automatic Panoramic Image Stitching using Invariant Features," IJCV 2007
- Burt & Adelson, "A Multiresolution Spline with Application to Image Mosaics," TOG 1983

## License

MIT
