BOKEH MOSAIC STITCHER — SINGLE‑PAGE, CLIENT‑ONLY (GITHUB PAGES)
Full integrated spec (desktop-first + optional mobile support)

You can hand this entire document to a coding agent. It is intended to be implementable as written.

--------------------------------------------------------------------------
0) Deliverables (what the agent must produce)

A. A working single-page web app that you can host on GitHub Pages (static assets only, no server), that:
- Accepts up to N images (JPG/PNG only; default cap varies by mode).
- Produces a rectilinear (“perspective”) stitched mosaic.
- Implements global refinement (bundle-adjustment-ish over homographies).
- Implements parallax correction via local mesh-based projective warps (APAP-inspired).
- Implements depth inference (monocular relative depth) to guide seam selection and local warps.
- Implements seam finding via a depth-guided graph cut on a coarse overlap grid.
- Implements multi-band (Laplacian pyramid) blending using the seam masks.
- Implements exposure compensation (per-image gains) to reduce brightness seams.
- Provides preview (fast, downscaled) and export (scaled by user; streamed; clamped to GPU limits).
- Supports an optional “mobile safe” mode that bounds memory/compute and degrades gracefully if necessary.

B. A starter repo with:
- Vite + TypeScript build
- Web Workers for CV, seam solver, depth inference
- WebGL2 shader pipeline for warp + seam cost + pyramid blending
- Optional cross-origin isolation “Turbo mode” via coi-serviceworker (same-origin static file)

C. A clear README that includes:
- Local dev instructions
- GH Pages deployment notes (base path)
- Recommended browsers and known limitations
- A short “How to get good Brenizer inputs” capture guide

--------------------------------------------------------------------------
1) Scope and constraints

Hard constraints:
- Hosting: GitHub Pages, fully static, all assets served from your own GH Pages origin.
- Input: JPG/PNG only. (If a browser can decode HEIC, you may accept it opportunistically, but do not rely on it.)
- Image count: configurable cap; must support ≤ 25 on desktop HQ mode; mobile-safe can enforce smaller caps.
- Output: rectilinear / perspective mosaic (not a spherical pano).
- Compute: client-only (no server).
- Primary use: desktop browsers. Mobile support is optional but required as a mode (see Section 4).

Non-goals (explicitly not required):
- Perfect results on moving subjects (ghosting can persist).
- Full 3D reconstruction / metric depth.
- HEIC/HEIF decode support across all browsers.
- PTGui-class robustness on extreme parallax or very large mosaics.

--------------------------------------------------------------------------
2) User experience spec (single-page)

Layout:
- Top bar: Upload / Auto-order / Stitch Preview / Export / Mode selector / Status
- Left panel: image list (thumb, filename, dims, exclude toggle, drag reorder, show warnings)
- Main panel: stitched preview canvas with zoom/pan; overlays; settings

Key UI controls:
1) Mode selector:
   - Auto
   - Desktop HQ
   - Mobile Quality
   - Mobile Safe
2) Mobile Safe flag (single switch):
   - ON forces Mobile Safe preset even on desktop
   - OFF allows chosen mode
3) Settings (show “Basic”; collapse “Advanced”):
   - Alignment max dimension: 1024 / 1536 / 2048
   - ORB features: 2000–8000
   - Pair window W: 2–12 + “Match all pairs”
   - Ratio test: 0.65–0.85
   - RANSAC threshold (px at alignment scale): 2–6
   - Global refinement iterations: 0–30
   - Mesh grid: 0 (off) / 8 / 10 / 12 / 16 / 20
   - Depth: ON/OFF
   - Seam method: Graph cut / Feather-only fallback
   - Seam block size: 8–32 (px at alignment scale)
   - Depth seam bias: 0–2
   - Feather width: 10–200 (px output-space)
   - Multi-band blending: ON/OFF + levels 3–7 (auto allowed)
   - Exposure compensation: ON/OFF
   - Export scale slider: 0.25–1.0 (clamped)
   - Export format: PNG or JPEG (JPEG quality slider)

Status and diagnostics:
- Step progress with timing per stage.
- A “Degraded features” panel that appears only when something is automatically downgraded (and must explain why).
- A “Connectivity / overlap diagnostics” view:
  - match graph connectivity summary
  - edge inlier heatmap / matrix
  - list of excluded/dropped images

Capture Guide tab (built into the SPA):
- Recommended shooting pattern grids (3×3, 4×4, 5×5)
- Overlap guidance (30–50%)
- “Rotate from one spot; don’t step; lock focus/exposure”
- Common failure causes and what to do

--------------------------------------------------------------------------
3) Runtime architecture

Threads:
- Main thread:
  - UI
  - WebGL2 renderer (warp, seam cost, mask blur, pyramid blending)
  - Orchestration / state machine
- cv-worker (classic worker):
  - OpenCV.js (WASM) initialization
  - feature extraction (ORB)
  - matching + RANSAC homographies
  - graph + MST ordering
  - global refinement (LM over homography params)
  - local mesh warp computation (APAP-inspired weighted DLT per vertex)
  - exposure compensation solve (per-image gains)
- depth-worker (module worker):
  - ONNX Runtime Web inference (select EP by capability)
  - returns normalized depth maps
- seam-worker (classic worker):
  - maxflow/mincut solver (WASM), operating on coarse block grids

Key principle:
- Alignment and inference always operate on downscaled images (alignment scale).
- High-res export is streamed: decode one image → upload texture → warp/blend → release → next.

--------------------------------------------------------------------------
4) Desktop + mobile option (modes, mobileSafe flag, capability detection)

4.1 Capabilities profiling (computed once after WebGL init)
Compute and store:
- isMobile (UA + screen heuristic)
- navigator.deviceMemory (if available; often absent on iOS)
- navigator.hardwareConcurrency
- WebGL2 available
- float render target support (EXT_color_buffer_float or equivalent)
- gl.MAX_TEXTURE_SIZE
- WebGPU availability: 'gpu' in navigator (note: availability ≠ guaranteed ORT EP support)
- crossOriginIsolated (self.crossOriginIsolated)

4.2 Presets (parameter bundles)
Desktop HQ (default on desktop):
- maxImages 25
- alignScale 1536 (allow 2048)
- ORB features 5000
- pairWindowW 6
- refineIters 30
- meshGrid 12
- seamBlock 16
- depthInputSize 256
- multiband levels auto up to 6
- exportScale default 0.5

Mobile Quality (full pipeline, smaller):
- maxImages 18
- alignScale 1024
- ORB features 3500
- pairWindowW 4
- refineIters 15
- meshGrid 10
- seamBlock 24
- depthInputSize 192
- multiband levels 4
- exportScale default 0.33

Mobile Safe (bounded + downgrade ladder enabled):
- maxImages 12
- alignScale 768
- ORB features 2000
- pairWindowW 3
- refineIters 8
- meshGrid 8
- seamBlock 32
- depthInputSize 128
- multiband levels 3
- exportScale default 0.25

Mobile Lite Fallback (only if necessary; must still output):
- maxImages 10
- alignScale 768
- ORB features 1500
- refineIters 0–4
- meshGrid OFF (0)
- seam: feather-only
- blending: feather-only (no pyramids)
- depth OFF

4.3 Mode selection logic
- mode ∈ {auto, desktopHQ, mobileQuality, mobileSafe}
- mobileSafe flag (boolean):
  - If ON, override effective settings to at most Mobile Safe, even on desktop.
  - If OFF, use selected mode or auto selection.
Auto selection:
- If isMobile: pick Mobile Safe by default; upgrade to Mobile Quality only if capability suggests enough headroom (e.g., deviceMemory >= 4 or hwConcurrency >= 6 and floatFBO true).
- If not mobile: pick Desktop HQ.

4.4 Depth inference EP selection (ORT Web)
ONNX Runtime Web docs state:
- WebGPU EP: available out-of-box in latest Chrome/Edge on Windows/macOS/Android; not supported in Safari iOS/macOS; WebGL is “maintenance mode.” (see ONNX Runtime Web docs)
Implementation rule:
- If you choose WebGPU EP, import `onnxruntime-web/webgpu` and create session with `executionProviders: ['webgpu']`.
- Otherwise import `onnxruntime-web` and select EP order:
  - prefer webgl if available
  - else wasm
You must show in UI which EP was selected and whether depth is enabled.

4.5 Mobile Safe downgrade ladder (mandatory for mobile support)
In Mobile Safe mode (including Auto→Mobile Safe), enforce time budgets per step. If exceeded, degrade in this order, with a visible “Degraded features” explanation:
1) Depth: reduce model input size (128) or switch EP (webgl→wasm) if init fails.
2) Reduce multi-band levels (3→2 if needed; if artifacts unacceptable, disable multi-band).
3) Increase seam block size (32→48) to reduce graph size.
4) Disable depth-guided seam term but keep graph cut (color/gradient only).
5) Disable graph cut → feather-only seam.
6) Disable mesh warp → global homography only.
7) Mobile Lite Fallback preset.

The app must still produce output (even if downgraded).

--------------------------------------------------------------------------
5) Processing pipeline (preview and export)

5.1 Common preprocessing
- Validate file types JPG/PNG.
- Enforce maxImages by mode (warn and block beyond).
- Decode with createImageBitmap if possible.
- Normalize orientation (EXIF). (Use exifr or equivalent; simplest reliable approach is OK.)
- Produce alignment-scale RGB and grayscale buffers:
  - alignment RGB for depth model input and seam cost
  - grayscale for ORB

Memory rule:
- Never keep full-res decoded RGBA for all images.
- Keep only file handles + downscaled artifacts + descriptors + depth maps.

5.2 Preview pipeline (downscaled)
Stages:
1) Depth inference (optional, recommended default ON)
2) ORB features per image
3) Candidate pair selection (window W; fallback to all pairs if disconnected)
4) Matching + ratio test + RANSAC homographies
5) Build weighted graph; select reference; build MST; initial global transforms
6) Global refinement over per-image homographies (reference fixed)
7) Exposure compensation (per-image gains)
8) Incremental composition in MST order:
   - compute mesh warp for each image relative to its parent edge (depth-gated)
   - warp into composite plane (WebGL mesh draw)
   - compute seam costs (GPU) over overlap block grid
   - seam mask via graph cut (seam-worker)
   - feather seam mask (GPU blur)
   - multi-band blend (pyramid)
9) Display preview

5.3 Export pipeline (streamed; scaled)
- Determine output dimensions from refined transforms and bounds.
- Apply export scale slider.
- Clamp by GPU max texture size:
  - query gl.MAX_TEXTURE_SIZE
  - if requested exceeds, clamp and warn
- Re-run incremental composition at export scale, streaming full-res decode one image at a time:
  - decode image → upload texture
  - warp + seam + blend
  - release image/texture data promptly
- Export PNG/JPEG via canvas.toBlob.

Note on max texture sizes:
- MAX_TEXTURE_SIZE is device-dependent; common values are 8192 and 16384, but you must query at runtime. Treat these as guidance only.

--------------------------------------------------------------------------
6) Algorithms (concrete implementation guidance)

6.1 Feature extraction (cv-worker, OpenCV.js)
- ORB, descriptors with Hamming distance.
- Defaults from preset.
- Output:
  - keypoints: Float32Array [x0,y0,x1,y1,...]
  - descriptors: Uint8Array (N×32)

6.2 Candidate pair selection
Default:
- Windowed: match i to j where 0 < (j-i) ≤ W
Fallback:
- If graph disconnected, expand W, then match all pairs (N≤25 => ≤300).

Optional prefilter:
- lightweight perceptual hash to avoid matching obviously unrelated frames (helps on unordered lists).

6.3 Matching + homography
For each pair:
- knnMatch k=2 + Lowe ratio (default 0.75)
- optional mutual cross-check
- findHomography with RANSAC threshold (px at align scale)
- Keep edge if:
  - inliers ≥ 35 (tune by mode)
  - inlier ratio sufficient (e.g., ≥ 0.2)
  - reprojection RMS reasonable
Store:
- H_ij (3×3), inlier correspondences.

6.4 Graph + MST + initial global transforms
- Edge weight: inliers / (RMS + ε)
- Reference selection: node with max sum edge weights (central hub)
- MST: maximum spanning tree from reference
- Propagate transforms along MST to get initial T_i.

6.5 Global refinement (LM over homographies)
Parameterization per image i (except reference):
- 8 params (a,b,c,d,e,f,g,h) with matrix:
  [a b c
   d e f
   g h 1]
Residual for each inlier match on edge (i,j):
- Xi = project(Ti, x_i)
- Xj = project(Tj, x_j)
- r = Xi - Xj (2D)
Minimize Σ Huber(||r||, delta) using Gauss-Newton + Levenberg damping.
Stop:
- max iterations from preset
- early stop if relative improvement small

Practical stabilization:
- Trim worst 5% residuals each iteration (optional)
- Add weak prior to initial transform (optional)

6.6 Exposure compensation (per-image scalar gain)
Compute from overlaps / inlier match luminance:
- For each edge (i,j), estimate mean log luminance ratio r_ij
Solve least squares:
- log g_j - log g_i ≈ r_ij
Fix reference gain = 1.
Apply gain in shader before seam/blend.

6.7 Depth inference (depth-worker)
Models:
- Provide at least 2 ONNX models: 256 and 128 input sizes (plus 192 optional).
- Use smaller by mode.

Runtime:
- WebGPU EP when supported; docs specify importing `onnxruntime-web/webgpu` and setting `executionProviders: ['webgpu']`.
- Otherwise WebGL or WASM EP; docs show WebGL is maintenance mode and WebGPU is preferred when available.

Output:
- depth map at model resolution
- normalize per image to [0,1] via percentile clipping (2%–98%)
Define depth convention:
- near = 1, far = 0 (consistent across pipeline)

6.8 Parallax correction via mesh-based local projective warps (APAP-inspired)
For each image i (non-reference) composed via parent p in MST:
- Use parent edge inlier correspondences:
  - x_i (source) in image i coords
  - x_p (parent) in image p coords
Convert targets to global plane:
- X_target = project(T_p, x_p)

Create mesh:
- grid GxG (from preset), vertices (G+1)×(G+1)
For each vertex v:
- compute local homography H_v mapping x_i → global X using weighted DLT:
  - spatial weight ws = exp(-||x_i - v||² / sigma²)
  - depth weight wd = exp(-|d(x_i) - d(v)|² / depthSigma²) if depth enabled
  - w = ws * wd
- If effective support < minSupport, fallback to refined global Ti for that vertex
- Warp vertex position to global plane using H_v

Render:
- triangles per cell using warped vertex positions and original UVs.
This yields a smooth, locally adaptive warp that tolerates mild parallax without optical flow.

6.9 Seam finding via depth-guided graph cut on coarse overlap grid
For each new image in incremental composition:
- Determine overlap region between existing composite and new warped image.
- Create block grid over overlap: blockSize from preset (mobile-safe uses larger blocks).
Compute per-block costs (GPU shader):
- color difference (mean abs RGB diff)
- gradient difference (optional Sobel)
- depth statistics from new image (mean near-ness)
Read back small cost texture.

Graph construction:
- Node per block.
- Labels: 0 = keep existing composite, 1 = take new image.
Hard constraints:
- outside coverage forces label.
Data term:
- D_new = depthBias * depthNear (penalize taking new near pixels; helps avoid ghosting on subject)
- D_comp = 0
Smoothness term:
- weights between adjacent blocks proportional to diff/grad (encourages seams through similar regions)

Solve min cut:
- seam-worker uses WASM maxflow implementation.
- Return label mask (block grid).
Upsample to pixel mask; then feather (Gaussian blur) to get soft alpha mask.

6.10 Multi-band blending (Laplacian pyramid) using seam masks
Maintain composite Laplacian pyramid Lc[0..L-1] (textures).
For each new image:
1) build Laplacian pyramid Ln for warped new image
2) build Gaussian pyramid M for seam mask (soft)
3) per level: Lc[level] = (1 - M[level]) * Lc[level] + M[level] * Ln[level]
After all images:
- reconstruct final image by upsampling and summing Laplacians.

Fallback rules:
- If floatFBO unsupported, reduce levels; if artifacts unacceptable, fall back to feather-only blending (must be explicitly reported as “degraded”).

--------------------------------------------------------------------------
7) WebGL2 rendering and GPU resource constraints

You must implement:
- Texture and FBO management with explicit disposal.
- A robust “capabilities check” that determines:
  - max texture size
  - float/half-float color buffer support
- Clamping logic for preview/export output sizes.

Shader passes (minimum set):
- warp mesh render (with exposure gain)
- overlap cost computation at block-grid resolution
- seam mask upsample + blur (separable)
- pyramid downsample / upsample
- laplacian compute
- laplacian blend
- reconstruct

Design note:
- Keep seam costs block-grid small so readPixels is cheap and reliable on mobile.

--------------------------------------------------------------------------
8) GH Pages deployment requirements (and optional Turbo mode)

8.1 Base path
- Vite config must support GH Pages subpath deployment:
  - base = "/REPO_NAME/" in production

8.2 Same-origin assets only
- Host OpenCV, ORT, models, wasm, shaders from your GH Pages origin.
- Avoid CDNs.

8.3 Optional Turbo mode (cross-origin isolation for threads)
- SharedArrayBuffer / WASM threads require cross-origin isolation (COOP+COEP).
- GH Pages can’t easily set those headers, so use coi-serviceworker if desired.
- coi-serviceworker explicitly targets cases where you can’t control headers (e.g., GH pages) and must be served from your own origin as a separate file.
- Turbo mode must not be required for correctness—only for speed.

--------------------------------------------------------------------------
9) Repo blueprint (starter tree)

/index.html
/vite.config.ts
/package.json
/tsconfig.json
/README.md

/public/
  /opencv/
    opencv.js
    opencv.wasm
  /ort/
    (ORT web assets as required by your chosen build)
  /models/
    depth_256.onnx
    depth_128.onnx
  /wasm/
    /maxflow/
      maxflow.js
      maxflow.wasm
  /shaders/
    warp.vert.glsl
    warp.frag.glsl
    cost_blocks.frag.glsl
    blur_h.frag.glsl
    blur_v.frag.glsl
    downsample.frag.glsl
    upsample.frag.glsl
    laplacian.frag.glsl
    blend_laplacian.frag.glsl
    reconstruct.frag.glsl
  /workers/
    cv-worker.js
    seam-worker.js
  /sw/
    coi-serviceworker.js

/src/
  main.ts
  appState.ts
  pipelineController.ts
  ui.ts
  capabilities.ts
  presets.ts
  gl/
    glContext.ts
    programs.ts
    textures.ts
    framebuffers.ts
    mesh.ts
    pyramid.ts
  workers/
    depth.worker.ts
    workerTypes.ts
  utils/
    exif.ts
    image.ts
    math.ts
    linalg.ts
    timing.ts
    download.ts
    budget.ts

--------------------------------------------------------------------------
10) Worker message contracts (must implement as specified)

10.1 depth.worker.ts (module worker)
Messages in:
- init { baseUrl, modelPath, preferWebGPU, targetSize, epPreference }
- infer { imageId, rgbaBuffer, width, height }
Messages out:
- progress { stage, done, total }
- result { imageId, depthUint16Buffer, depthW, depthH, nearIsOne, epUsed }
- error { imageId?, message }

10.2 cv-worker.js (classic worker)
Messages in:
- init { baseUrl, opencvPath }
- addImage { imageId, grayBuffer, width, height, rgbSmallBuffer?, depth? }
- computeFeatures { orbParams }
- matchGraph { windowW, ratio, ransacThreshPx, minInliers, matchAllPairs }
- buildGraph { }
- refine { maxIters, huberDeltaPx, lambdaInit }
- computeExposure { }
- buildMST { }
- computeLocalMesh { imageId, parentId, meshGrid, sigma, depthSigma, minSupport }
Messages out:
- progress { stage, percent, info }
- features { imageId, keypointsBuffer, descriptorsBuffer, descCols }
- edges { edges: [{i,j,HBuffer,inliersBuffer,rms,inlierCount}] }
- transforms { refId, transforms: [{imageId,TBuffer}] }
- exposure { gains: [{imageId,gain}] }
- mst { refId, order: [imageId...], parent: {imageId: parentId|null} }
- mesh { imageId, verticesBuffer, uvsBuffer, indicesBuffer, bounds }
- error { message }

10.3 seam-worker.js (classic worker)
Messages in:
- init { baseUrl, maxflowPath }
- solve { jobId, gridW, gridH, dataCostsBuffer, edgeWeightsBuffer?, hardConstraintsBuffer, params }
Messages out:
- result { jobId, labelsBuffer }
- error { jobId?, message }

Note:
- Seam solver must run on coarse grids; never send per-pixel costs.

--------------------------------------------------------------------------
11) Default settings per mode (explicit)

Auto:
- if isMobile → Mobile Safe unless capability is strong enough → Mobile Quality
- else Desktop HQ

Desktop HQ defaults:
- alignScale 1536
- ORB 5000
- W 6
- ratio 0.75
- RANSAC 3px
- refineIters 30
- meshGrid 12
- depth ON, input 256
- seam graph cut ON, block 16
- depth seam bias 1.0
- feather 60px
- multiband ON, levels auto (≤6)
- exportScale 0.5

Mobile Safe defaults:
- alignScale 768
- ORB 2000
- W 3
- refineIters 8
- meshGrid 8
- depth ON, input 128
- seam graph cut ON, block 32
- multiband ON, levels 3
- exportScale 0.25

--------------------------------------------------------------------------
12) Work plan checklist (sequenced tasks; buildable)

Phase 0 — Scaffold + GH Pages deploy
- Create Vite+TS app, confirm base path works on GH Pages.
- Basic UI layout, file import, thumbnails, reorder, exclude, validation.

Phase 1 — WebGL2 foundation
- GL context init, shader loader, texture/FBO helpers, simple full-screen draw.
- Warp mesh renderer (identity mesh) and zoom/pan.

Phase 2 — Depth worker
- Integrate ORT Web with conditional importing:
  - WebGPU path uses `onnxruntime-web/webgpu` and `executionProviders: ['webgpu']`.
  - else webgl/wasm fallback.
- Implement depth normalization and return Uint16.

Phase 3 — OpenCV worker
- Load OpenCV.js in classic worker.
- ORB feature extraction for alignment grayscale.
- Matching + ratio + findHomography(RANSAC).
- Return edges and inlier matches.

Phase 4 — Graph + initial stitch sanity
- Build graph, pick ref, MST, initial transforms.
- Render all warped images with global homography only (no seams) for validation.

Phase 5 — Global refinement
- Implement LM refinement and verify improved alignment.

Phase 6 — Exposure compensation
- Implement gain solve and apply in shader.

Phase 7 — Mesh parallax warp
- Implement per-vertex weighted DLT local homographies with depth gating.
- Render mesh warps; confirm reduced parallax artifacts vs global homography.

Phase 8 — Seam costs + graph cut
- GPU compute block-grid costs; read back.
- WASM maxflow in seam-worker; generate seam mask; feather blur.

Phase 9 — Multi-band blending
- Build pyramid pipeline, laplacian blend, reconstruct.
- Integrate seam masks into pyramid blending.

Phase 10 — Full incremental composition loop
- Compose in MST order: warp → seam → blend
- Preview output + overlays + debug views.

Phase 11 — Export (streamed, clamped)
- Compute bounds; apply exportScale; clamp by MAX_TEXTURE_SIZE.
- Export PNG/JPEG; ensure one-image-at-a-time decode/upload.

Phase 12 — Modes + mobile support
- Implement capabilities.ts + presets.ts.
- Implement Auto selection + mobileSafe flag.
- Implement downgrade ladder and “Degraded features” reporting.

Phase 13 — Capture Guide + diagnostics
- Capture guide tab content.
- Connectivity diagnostics: inlier matrix heatmap + disconnected component warnings.

Phase 14 — Robustness and QA
- Cancellation support.
- Fallback paths (no floatFBO; disable multi-band; feather-only).
- “Why failed” messages (disconnected graph, insufficient inliers, etc.).

Optional Phase 15 — Turbo mode (threads)
- Add coi-serviceworker (same-origin, separate file) per its usage notes.
- Detect crossOriginIsolated; enable threaded ORT/OpenCV builds only when available.

--------------------------------------------------------------------------
13) Acceptance criteria (definition of done)

Functional:
- Desktop HQ: stitches 10–25 images into a preview and exports successfully.
- Mobile Safe: stitches 6–12 images into a preview and exports at reduced scale without crashing, with clear degradations if needed.

Quality:
- Parallax mesh reduces visible double edges in at least one mild-parallax test set versus global homography-only.
- Depth-guided seam reduces ghosting around near subject in at least one portrait Brenizer set versus depth-off seam.

Stability:
- Graph disconnected → stitches largest component and reports excluded images.
- Refinement divergence → fallback to pre-refine transforms and reports.
- Output too large → clamps and reports.

Usability:
- Clear progress steps and cancellable long runs.
- Capture guide present.
- Diagnostics panel helps users understand failure modes.

--------------------------------------------------------------------------
14) Implementation notes that prevent common failures

- Do not do per-pixel graph cuts; seam must be solved on a coarse block grid.
- Do not keep full-res ImageData for all inputs; stream export.
- Keep all third-party assets same-origin; avoid CDNs.
- Always query and respect MAX_TEXTURE_SIZE; do not assume 16384.
- Treat ORT WebGPU as “nice when available” and follow ORT’s supported EP/browser matrix; WebGPU is not supported in Safari iOS per ORT’s own table.
- When using WebGPU EP in ORT, follow the required import + session option change.
- If you enable cross-origin isolation via coi-serviceworker, it must be a separate file served from your origin, and it will reload once on first load.

--------------------------------------------------------------------------
15) Minimal README outline (agent should write this)

- What it is / what it does
- Supported browsers:
  - Desktop: Chrome/Edge recommended
  - Mobile: works in Mobile Safe mode; may degrade
- File type limitations: JPG/PNG
- How to capture Brenizer sets (short checklist)
- Local dev: npm install / npm run dev
- Build: npm run build
- Deploy to GH Pages: base path, typical GitHub Actions snippet (optional)
- Known limitations and troubleshooting

End of full spec.
