import { detectCapabilities } from './capabilities';
import { resolveMode, getPreset } from './presets';
import { setState, getState, subscribe } from './appState';
import { initUI, renderCapabilities, setStatus, setRenderImagePreview, startProgress, endProgress, updateProgress, buildSettingsPanel } from './ui';
import {
  createGLContext, createWarpRenderer, createKeypointRenderer, createCompositor,
  createPyramidBlender,
  createIdentityMesh, createTextureFromImage, createEmptyTexture, createFBO,
  makeViewMatrix, computeBlockCosts, labelsToMask, featherMask,
  createMaskTexture,
  type GLContext, type WarpRenderer, type KeypointRenderer, type Compositor,
  type PyramidBlender, type ManagedTexture, type FaceRectComposite,
} from './gl';
import {
  runStitchPreview, getLastFeatures, getLastEdges, getLastTransforms, getLastRefId,
  getLastGains, getLastMeshes, getLastMstOrder, getLastMstParent, getWorkerManager,
  getLastFaces, getLastSaliency, getLastVignette,
} from './pipelineController';
import type { SeamResultMsg } from './workers/workerTypes';

let glCtx: GLContext | null = null;
let warpRenderer: WarpRenderer | null = null;
let kpRenderer: KeypointRenderer | null = null;
let compositor: Compositor | null = null;
let pyramidBlender: PyramidBlender | null = null;

/** Sanitize a gain value: replace NaN/Infinity/non-positive with 1.0. */
function safeGainVal(v: number): number {
  return Number.isFinite(v) && v > 0 ? v : 1.0;
}

/**
 * Project a per-image saliency map into composite pixel coordinates.
 * For each composite pixel, inverse-warp through T to image-space and sample.
 * Returns null if no saliency data is available.
 */
function projectSaliencyToComposite(
  imgId: string,
  T: Float64Array,
  minX: number, minY: number,
  compositeScale: number,
  compW: number, compH: number,
  saliencyMaps: Map<string, import('./pipelineController').SaliencyData>,
): Float32Array | null {
  const sal = saliencyMaps.get(imgId);
  if (!sal) return null;
  const sw = sal.width, sh = sal.height;
  const sMap = sal.saliency;
  if (sMap.length < sw * sh) return null;

  // Compute inverse of T for back-projection
  const a = T[0], b = T[1], c = T[2];
  const d = T[3], e = T[4], f = T[5];
  const g = T[6], h = T[7], k = T[8];
  const det = a * (e * k - f * h) - b * (d * k - f * g) + c * (d * h - e * g);
  if (Math.abs(det) < 1e-15) return null;
  const invDet = 1 / det;
  const Ti = new Float64Array([
    (e * k - f * h) * invDet, (c * h - b * k) * invDet, (b * f - c * e) * invDet,
    (f * g - d * k) * invDet, (a * k - c * g) * invDet, (c * d - a * f) * invDet,
    (d * h - e * g) * invDet, (b * g - a * h) * invDet, (a * e - b * d) * invDet,
  ]);

  const out = new Float32Array(compW * compH);
  // Sparse sampling: sample every 4th pixel for speed, fill gaps
  const step = 4;
  for (let cy = 0; cy < compH; cy += step) {
    for (let cx = 0; cx < compW; cx += step) {
      // Composite pixel → global alignment coords
      const gx = cx / compositeScale + minX;
      const gy = cy / compositeScale + minY;
      // Global → image coords via T^{-1}
      const denom = Ti[6] * gx + Ti[7] * gy + Ti[8];
      if (Math.abs(denom) < 1e-8) continue;
      const ix = (Ti[0] * gx + Ti[1] * gy + Ti[2]) / denom;
      const iy = (Ti[3] * gx + Ti[4] * gy + Ti[5]) / denom;
      if (ix < 0 || ix >= sw - 1 || iy < 0 || iy >= sh - 1) continue;
      // Nearest-neighbor sample from saliency map
      const si = Math.round(iy) * sw + Math.round(ix);
      const val = sMap[si] ?? 0;
      // Fill the step×step block
      for (let dy = 0; dy < step && cy + dy < compH; dy++) {
        for (let dx = 0; dx < step && cx + dx < compW; dx++) {
          out[(cy + dy) * compW + (cx + dx)] = val;
        }
      }
    }
  }
  return out;
}

/** Expose for UI to trigger image preview via WebGL */
export function getGLContext(): GLContext | null { return glCtx; }
export function getWarpRenderer(): WarpRenderer | null { return warpRenderer; }

/** Render a single uploaded image on the canvas via WebGL. */
export async function renderImagePreview(imageEntry: { file: File; width: number; height: number }): Promise<void> {
  if (!glCtx || !warpRenderer) return;
  const { gl, canvas } = glCtx;

  // Size canvas to container
  const container = document.getElementById('canvas-container')!;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  canvas.style.display = 'block';
  document.getElementById('canvas-placeholder')!.style.display = 'none';

  // Decode the image
  const bmp = await createImageBitmap(imageEntry.file);
  const tex = createTextureFromImage(gl, bmp, bmp.width, bmp.height);
  bmp.close();

  // Create identity mesh
  const mesh = createIdentityMesh(tex.width, tex.height, 4, 4);
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, tex.width, tex.height);

  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  warpRenderer.drawMesh(tex.texture, mesh, viewMat);
  tex.dispose();
}

/**
 * Draw keypoint overlay for a specific image.
 * The keypoints are in alignment-space coords so we must account for the scale.
 */
export function renderKeypointOverlay(imageId: string, imgW: number, imgH: number): void {
  if (!glCtx || !kpRenderer) return;
  const features = getLastFeatures().get(imageId);
  if (!features || features.keypoints.length < 2) return;

  const { gl, canvas } = glCtx;
  // keypoints are in alignment-scaled coords; map them to original coords
  // then build a viewMatrix that matches the one used for image preview
  const sf = features.scaleFactor;
  const origW = imgW;
  const origH = imgH;

  // Scale keypoints from alignment coords to original image coords
  const kps = features.keypoints;
  const scaledKps = new Float32Array(kps.length);
  for (let i = 0; i < kps.length; i += 2) {
    scaledKps[i] = kps[i] / sf;
    scaledKps[i + 1] = kps[i + 1] / sf;
  }

  // Use same viewMatrix as renderImagePreview
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, origW, origH);

  // Distinct colours per-image for multi-image display (cycle through palette)
  const palette: [number, number, number, number][] = [
    [0, 1, 0.3, 0.8],  // green
    [1, 0.3, 0, 0.8],  // red-orange
    [0.2, 0.6, 1, 0.8], // blue
    [1, 1, 0, 0.8],    // yellow
    [1, 0, 1, 0.8],    // magenta
    [0, 1, 1, 0.8],    // cyan
  ];
  const { images } = getState();
  const idx = images.findIndex(i => i.id === imageId);
  const color = palette[idx % palette.length];

  kpRenderer.drawKeypoints(scaledKps, viewMat, color, 6);
}

async function boot(): Promise<void> {
  // Init UI first so elements are wired
  initUI();
  setRenderImagePreview(renderImagePreview);

  // Detect capabilities
  setStatus('Detecting capabilities…');
  const caps = await detectCapabilities();
  setState({ capabilities: caps });
  renderCapabilities(caps);

  // Resolve mode and apply preset
  const { userMode, mobileSafeFlag } = getState();
  const resolved = resolveMode(userMode, mobileSafeFlag, caps);
  const settings = getPreset(resolved);
  setState({ resolvedMode: resolved, settings });
  setStatus(`Ready — mode: ${resolved}`);

  // Init WebGL2 context
  try {
    const canvas = document.getElementById('preview-canvas') as HTMLCanvasElement;
    glCtx = createGLContext(canvas);
    warpRenderer = createWarpRenderer(glCtx.gl);
    kpRenderer = createKeypointRenderer(glCtx.gl);
    compositor = createCompositor(glCtx.gl);
    pyramidBlender = createPyramidBlender(glCtx.gl, glCtx.floatFBO);
    console.log('WebGL2 context initialised, max tex:', glCtx.maxTextureSize);
  } catch (e) {
    console.warn('WebGL2 init failed:', e);
    setStatus('Warning: WebGL2 not available — rendering disabled');
  }

  // Re-resolve mode on setting changes
  subscribe(() => {
    const s = getState();
    if (s.capabilities) {
      const newMode = resolveMode(s.userMode, s.mobileSafeFlag, s.capabilities);
      if (newMode !== s.resolvedMode) {
        const newSettings = getPreset(newMode);
        setState({ resolvedMode: newMode, settings: newSettings });
        buildSettingsPanel(); // rebuild settings UI for new mode
        setStatus(`Mode changed to ${newMode}`);
      }
    }
  });

  // Wire Stitch Preview button
  document.getElementById('btn-stitch')!.addEventListener('click', () => {
    runStitchPreview().catch(err => {
      console.error('Pipeline error:', err);
      setStatus(`Pipeline error: ${err.message}`);
    });
  });

  // Wire Export button
  document.getElementById('btn-export')!.addEventListener('click', () => {
    exportComposite().catch(err => {
      console.error('Export error:', err);
      setStatus(`Export error: ${err.message}`);
    });
  });

  // Wire Full-Res Export button — uses a cloned settings object with maxResExport=true
  document.getElementById('btn-export-fullres')!.addEventListener('click', () => {
    const { settings } = getState();
    if (!settings) return;
    // Clone to avoid mutating shared state during async export
    setState({ settings: { ...settings, maxResExport: true } });
    exportComposite().catch(err => {
      console.error('Export error:', err);
      setStatus(`Export error: ${err.message}`);
    }).finally(() => {
      setState({ settings: { ...settings } }); // restore original
    });
  });

  // Wire Settings toggle
  document.getElementById('btn-settings')?.addEventListener('click', () => {
    document.getElementById('settings-panel')?.classList.toggle('open');
  });

  // Tab switching
  document.querySelectorAll('#main-tabs .tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#main-tabs .tab-btn').forEach(b => {
        (b as HTMLElement).style.borderBottomColor = 'transparent';
        (b as HTMLElement).style.color = 'var(--text-dim)';
        b.classList.remove('active');
      });
      (btn as HTMLElement).style.borderBottomColor = 'var(--accent)';
      (btn as HTMLElement).style.color = 'var(--text)';
      btn.classList.add('active');
      document.querySelectorAll('.tab-content').forEach(t => {
        (t as HTMLElement).style.display = 'none';
      });
      const tabId = (btn as HTMLElement).dataset.tab!;
      const target = document.getElementById(`tab-${tabId}`);
      if (target) {
        // Preview tab uses flex layout for canvas; others use block for scrollable content
        target.style.display = tabId === 'preview' ? 'flex' : 'block';
      }
    });
  });

  // Draw keypoint overlay after feature extraction completes
  window.addEventListener('features-ready', async () => {
    const { images } = getState();
    const active = images.filter(i => !i.excluded);
    if (active.length === 0 || !glCtx || !warpRenderer) return;

    // Re-render the first image then overlay its keypoints
    const first = active[0];
    await renderImagePreview(first);
    renderKeypointOverlay(first.id, first.width, first.height);
  });

  // Display match edge info after matching completes
  window.addEventListener('edges-ready', () => {
    const edges = getLastEdges();
    const { images } = getState();
    const active = images.filter(i => !i.excluded);
    if (edges.length === 0) return;

    // Log match matrix to console for diagnostics
    console.group('Match Graph');
    for (const e of edges) {
      const nameI = active.find(i => i.id === e.i)?.name ?? e.i;
      const nameJ = active.find(i => i.id === e.j)?.name ?? e.j;
      console.log(`${nameI} ↔ ${nameJ}: ${e.inlierCount} inliers, RMS=${e.rms.toFixed(2)}`);
    }
    console.groupEnd();

    // Render inline match heatmap in the status area
    renderMatchHeatmap(active, edges);

    // Populate diagnostics panel
    renderDiagnostics(active, edges);
  });

  // Render warped multi-image preview after transforms are computed
  window.addEventListener('transforms-ready', async () => {
    const { images } = getState();
    const active = images.filter(i => !i.excluded);
    const transforms = getLastTransforms();
    if (active.length === 0 || !glCtx || !warpRenderer || transforms.size === 0) return;

    await renderWarpedPreview(active, transforms);
  });
}

/** Render a simple text-based match inlier matrix in the capabilities bar. */
function renderMatchHeatmap(
  images: import('./appState').ImageEntry[],
  edges: import('./pipelineController').MatchEdge[],
): void {
  const bar = document.getElementById('capabilities-bar');
  if (!bar) return;

  // Remove any previous heatmap
  const prev = bar.querySelector('.inlier-matrix');
  if (prev) prev.remove();

  // Build NxN inlier count matrix
  const n = images.length;
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  const idToIdx = new Map(images.map((img, i) => [img.id, i]));

  for (const e of edges) {
    const a = idToIdx.get(e.i);
    const b = idToIdx.get(e.j);
    if (a !== undefined && b !== undefined) {
      matrix[a][b] = e.inlierCount;
      matrix[b][a] = e.inlierCount;
    }
  }

  // Find max for colour scaling
  let maxCount = 1;
  for (const row of matrix) for (const v of row) if (v > maxCount) maxCount = v;

  // Build HTML table
  /** Escape HTML entities to prevent XSS from crafted filenames. */
  const esc = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  const shortNames = images.map(i => esc(i.name.replace(/\.[^.]+$/, '').slice(0, 8)));
  let html = '<div class="inlier-matrix" style="margin-top:8px;font-size:11px;"><strong>Inlier Matrix</strong>';
  html += '<table style="border-collapse:collapse;margin-top:4px;">';
  html += '<tr><td></td>';
  for (const name of shortNames) html += `<td style="padding:2px 4px;font-size:10px;text-align:center;">${name}</td>`;
  html += '</tr>';
  for (let r = 0; r < n; r++) {
    html += `<tr><td style="padding:2px 4px;font-size:10px;">${shortNames[r]}</td>`;
    for (let c = 0; c < n; c++) {
      if (r === c) {
        html += '<td style="background:#333;width:24px;height:20px;text-align:center;font-size:9px;">–</td>';
      } else {
        const v = matrix[r][c];
        const t = v / maxCount;
        const r0 = Math.round(255 * (1 - t));
        const g0 = Math.round(255 * t);
        const bg = `rgb(${r0},${g0},60)`;
        html += `<td style="background:${bg};width:24px;height:20px;text-align:center;font-size:9px;color:#fff;">${v || ''}</td>`;
      }
    }
    html += '</tr>';
  }
  html += '</table></div>';
  bar.innerHTML += html;
}

/** Render connectivity diagnostics panel with heatmap canvas and summary. */
function renderDiagnostics(
  images: import('./appState').ImageEntry[],
  edges: import('./pipelineController').MatchEdge[],
): void {
  const emptyEl = document.getElementById('diagnostics-empty');
  const contentEl = document.getElementById('diagnostics-content');
  if (!emptyEl || !contentEl) return;
  emptyEl.style.display = 'none';
  contentEl.style.display = 'block';

  const n = images.length;
  const idToIdx = new Map(images.map((img, i) => [img.id, i]));
  const matrix: number[][] = Array.from({ length: n }, () => Array(n).fill(0));
  let maxCount = 1;
  for (const e of edges) {
    const a = idToIdx.get(e.i);
    const b = idToIdx.get(e.j);
    if (a !== undefined && b !== undefined) {
      matrix[a][b] = e.inlierCount;
      matrix[b][a] = e.inlierCount;
      if (e.inlierCount > maxCount) maxCount = e.inlierCount;
    }
  }

  // Summary text
  const summaryEl = document.getElementById('diag-summary')!;
  const totalEdges = edges.length;
  const possibleEdges = n * (n - 1) / 2;
  const avgInliers = totalEdges > 0 ? (edges.reduce((s, e) => s + e.inlierCount, 0) / totalEdges).toFixed(1) : '0';
  const minInliers = totalEdges > 0 ? Math.min(...edges.map(e => e.inlierCount)) : 0;
  const maxInliers = totalEdges > 0 ? Math.max(...edges.map(e => e.inlierCount)) : 0;

  // Detect connected components via union-find
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x: number): number { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
  for (const e of edges) {
    const a = idToIdx.get(e.i);
    const b = idToIdx.get(e.j);
    if (a !== undefined && b !== undefined && e.inlierCount >= 8) {
      parent[find(a)] = find(b);
    }
  }
  const components = new Map<number, number[]>();
  for (let i = 0; i < n; i++) {
    const root = find(i);
    if (!components.has(root)) components.set(root, []);
    components.get(root)!.push(i);
  }
  const numComponents = components.size;
  const isConnected = numComponents === 1;

  summaryEl.innerHTML =
    `<div>${n} images, ${totalEdges}/${possibleEdges} edges matched</div>` +
    `<div>Inliers: min ${minInliers}, max ${maxInliers}, avg ${avgInliers}</div>` +
    `<div>Connected components: ${numComponents} ${isConnected ? '✓ fully connected' : '⚠ disconnected!'}</div>`;

  if (!isConnected) {
    summaryEl.innerHTML += `<div style="color:var(--warn); margin-top:4px;">Warning: The match graph is disconnected. Some images may not be included in the final mosaic. Ensure sufficient overlap between all frames.</div>`;
  }

  // Heatmap canvas
  const heatmap = document.getElementById('inlier-heatmap') as HTMLCanvasElement;
  const cellSize = Math.max(12, Math.min(40, Math.floor(300 / n)));
  heatmap.width = cellSize * n;
  heatmap.height = cellSize * n;
  const ctx = heatmap.getContext('2d')!;
  ctx.clearRect(0, 0, heatmap.width, heatmap.height);

  for (let r = 0; r < n; r++) {
    for (let c = 0; c < n; c++) {
      if (r === c) {
        ctx.fillStyle = '#333';
      } else {
        const v = matrix[r][c];
        const t = v / maxCount;
        ctx.fillStyle = `rgb(${Math.round(255 * (1 - t))},${Math.round(255 * t)},60)`;
      }
      ctx.fillRect(c * cellSize, r * cellSize, cellSize - 1, cellSize - 1);
      if (matrix[r][c] > 0 && r !== c && cellSize >= 20) {
        ctx.fillStyle = '#fff';
        ctx.font = `${Math.max(8, cellSize * 0.4)}px sans-serif`;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(String(matrix[r][c]), c * cellSize + cellSize / 2, r * cellSize + cellSize / 2);
      }
    }
  }

  // Excluded images
  const excludedEl = document.getElementById('diag-excluded')!;
  const { images: allImages } = getState();
  const excludedImages = allImages.filter(i => i.excluded);
  if (excludedImages.length > 0) {
    excludedEl.innerHTML = excludedImages.map(i => {
      const safe = i.name.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      return `<div style="color:var(--warn);">⚠ ${safe} — excluded</div>`;
    }).join('');
  } else {
    excludedEl.textContent = 'No images excluded.';
  }

  // If disconnected, list component membership
  if (!isConnected) {
    let compHtml = '<h4 style="font-size:13px; margin:12px 0 4px; color:var(--warn);">Disconnected Components</h4>';
    let compIdx = 0;
    for (const [, members] of components) {
      compIdx++;
      const escName = (s: string) => s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
      const names = members.map(i => escName(images[i]?.name ?? `image-${i}`)).join(', ');
      compHtml += `<div style="font-size:12px; margin-bottom:4px;">Component ${compIdx} (${members.length}): ${names}</div>`;
    }
    excludedEl.innerHTML += compHtml;
  }
}

/**
 * Render incremental composition with seam finding.
 * For each image in MST order:
 *  1. Warp into global space via APAP mesh or global homography
 *  2. Compute overlap costs at block-grid resolution
 *  3. Solve graph cut via seam-worker
 *  4. Generate feathered seam mask
 *  5. Blend into composite
 */
async function renderWarpedPreview(
  images: import('./appState').ImageEntry[],
  transforms: Map<string, import('./pipelineController').GlobalTransform>,
): Promise<void> {
  if (!glCtx || !warpRenderer || !compositor) return;
  const { gl, canvas } = glCtx;
  const features = getLastFeatures();
  const gains = getLastGains();
  const meshes = getLastMeshes();
  const vignettes = getLastVignette();
  const { settings } = getState();
  const refId = getLastRefId();
  const wm = getWorkerManager();
  const mstParent = getLastMstParent();

  // Size canvas to container
  const container = document.getElementById('canvas-container')!;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  canvas.style.display = 'block';
  document.getElementById('canvas-placeholder')!.style.display = 'none';

  // Determine which images are reachable from the reference in the MST.
  // Disconnected images (no parent AND not the reference) get identity transforms
  // and would overlap the reference — exclude them.
  const connectedIds = new Set<string>();
  if (refId) connectedIds.add(refId);
  for (const [id, parentId] of Object.entries(mstParent)) {
    if (parentId !== null) connectedIds.add(id);
  }
  // If no MST data at all, treat all images with transforms as connected
  if (connectedIds.size === 0) {
    for (const img of images) {
      if (transforms.has(img.id)) connectedIds.add(img.id);
    }
  }

  // Compute bounding box across all warped image corners in global coords.
  // Prefer APAP mesh bounds when available; fall back to projecting corners
  // via the global homography, but skip points that project behind the camera
  // (negative denominator) to avoid extreme coordinates from perspective wrap.
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  // Compute a reasonable max extent for sanity clamping
  let refDim = 1536; // fallback
  for (const img of images) {
    if (img.id === refId) {
      const feat = features.get(img.id);
      const sf = feat?.scaleFactor ?? 1;
      refDim = Math.max(img.width * sf, img.height * sf);
      break;
    }
  }
  const maxReasonableExtent = refDim * images.length * 2;

  for (const img of images) {
    if (!connectedIds.has(img.id)) {
      console.log(`Skipping disconnected image: ${img.name}`);
      continue;
    }
    const t = transforms.get(img.id);
    if (!t) continue;

    // If this image has an APAP mesh, use its pre-computed bounds
    const apap = meshes.get(img.id);
    if (apap) {
      const b = apap.bounds;
      // Clamp APAP bounds to reasonable extent (some vertices may still be extreme)
      const clampedMinX = Math.max(b.minX, -maxReasonableExtent);
      const clampedMinY = Math.max(b.minY, -maxReasonableExtent);
      const clampedMaxX = Math.min(b.maxX, maxReasonableExtent);
      const clampedMaxY = Math.min(b.maxY, maxReasonableExtent);
      console.log(`Image ${img.name}: APAP bounds=[${clampedMinX.toFixed(1)},${clampedMinY.toFixed(1)},${clampedMaxX.toFixed(1)},${clampedMaxY.toFixed(1)}]`);
      minX = Math.min(minX, clampedMinX); minY = Math.min(minY, clampedMinY);
      maxX = Math.max(maxX, clampedMaxX); maxY = Math.max(maxY, clampedMaxY);
      continue;
    }

    // Fall back to projecting 4 corners + edge midpoints via global T
    const feat = features.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    const w = img.width * sf;
    const h = img.height * sf;
    const T = t.T;
    // Sample corners and edge midpoints for better coverage
    const samples = [
      [0, 0], [w, 0], [w, h], [0, h],
      [w / 2, 0], [w, h / 2], [w / 2, h], [0, h / 2], [w / 2, h / 2],
    ];
    let imgMinX = Infinity, imgMinY = Infinity, imgMaxX = -Infinity, imgMaxY = -Infinity;
    for (const [cx, cy] of samples) {
      const denom = T[6] * cx + T[7] * cy + T[8];
      // Skip if behind camera (denom <= 0) or near-singular
      if (denom < 1e-4) continue;
      const gx = (T[0] * cx + T[1] * cy + T[2]) / denom;
      const gy = (T[3] * cx + T[4] * cy + T[5]) / denom;
      // Sanity clamp: skip if projected too far from origin
      if (Math.abs(gx) > maxReasonableExtent || Math.abs(gy) > maxReasonableExtent) continue;
      imgMinX = Math.min(imgMinX, gx); imgMinY = Math.min(imgMinY, gy);
      imgMaxX = Math.max(imgMaxX, gx); imgMaxY = Math.max(imgMaxY, gy);
    }
    if (isFinite(imgMinX)) {
      console.log(`Image ${img.name}: T-bounds=[${imgMinX.toFixed(1)},${imgMinY.toFixed(1)},${imgMaxX.toFixed(1)},${imgMaxY.toFixed(1)}]`);
      minX = Math.min(minX, imgMinX); minY = Math.min(minY, imgMinY);
      maxX = Math.max(maxX, imgMaxX); maxY = Math.max(maxY, imgMaxY);
    } else {
      console.warn(`Image ${img.name}: all corners project behind camera or out of range, skipping from bounds`);
    }
  }
  if (!isFinite(minX)) return;

  console.log(`Global bounds: [${minX.toFixed(1)},${minY.toFixed(1)}] → [${maxX.toFixed(1)},${maxY.toFixed(1)}]`);

  const globalW = maxX - minX;
  const globalH = maxY - minY;

  // Clamp composite size to safe GPU limits
  const maxTexSize = glCtx.maxTextureSize;
  const compositeScale = Math.min(1, maxTexSize / Math.max(globalW, globalH));
  const compW = Math.round(globalW * compositeScale);
  const compH = Math.round(globalH * compositeScale);

  console.log(`Composite: ${compW}×${compH}, scale=${compositeScale.toFixed(4)}`);

  // Create FBOs for composition — inside try so finally always cleans up
  let compositeTexA: import('./gl').ManagedTexture | null = null;
  let compositeTexB: import('./gl').ManagedTexture | null = null;
  let compositeA: import('./gl').ManagedFBO | null = null;
  let compositeB: import('./gl').ManagedFBO | null = null;
  let newImageTex: import('./gl').ManagedTexture | null = null;
  let newImageFBO: import('./gl').ManagedFBO | null = null;

  try {
  compositeTexA = createEmptyTexture(gl, compW, compH);
  compositeTexB = createEmptyTexture(gl, compW, compH);
  compositeA = createFBO(gl, compositeTexA.texture);
  compositeB = createFBO(gl, compositeTexB.texture);
  newImageTex = createEmptyTexture(gl, compW, compH);
  newImageFBO = createFBO(gl, newImageTex.texture);

  // View matrix for composite space: maps [0, compW] × [0, compH] → clip [-1, 1]
  const compViewMat = new Float32Array([
    2 / compW, 0, 0,
    0, -2 / compH, 0,
    -1, 1, 1,
  ]);

  let currentCompTex = compositeTexA;
  let currentCompFBO = compositeA;
  let altCompTex = compositeTexB;
  let altCompFBO = compositeB;

  // Set up compositing progress
  startProgress([{ name: 'compositing', weight: 1 }]);
  setStatus('Compositing (0%)…');

  // Clear composite
  gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
  gl.viewport(0, 0, compW, compH);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // Determine MST order, falling back to image order with ref first.
  // Filter to only connected images.
  const mstOrder = (() => {
    const order: string[] = getLastMstOrder();
    if (order.length > 0) return order.filter(id => connectedIds.has(id));
    // Fallback: ref first, then rest
    const ids = images.map(i => i.id).filter(id => connectedIds.has(id));
    if (refId && ids.includes(refId)) {
      return [refId, ...ids.filter(id => id !== refId)];
    }
    return ids;
  })();

  const blockSize = settings?.seamBlockSize ?? 16;
  const featherWidth = settings?.featherWidth ?? 60;
  const useGraphCut = settings?.seamMethod === 'graphcut' && wm !== null;
  const useMultiband = settings?.multibandEnabled && pyramidBlender !== null;
  const mbLevels = (() => {
    const l = settings?.multibandLevels ?? 0;
    if (l > 0) return l;
    // ── Adaptive multi-band level selection ──────────────────
    // Instead of basing pyramid levels purely on image size, we estimate
    // the typical overlap width between adjacent images and choose levels
    // so the lowest-frequency band roughly matches the overlap zone.
    // This ensures the blend transition spans exactly the overlap region,
    // avoiding both too-narrow blends (visible seams) and too-wide blends
    // (ghosting from excessive smoothing across non-overlapping areas).
    //
    // The pyramid's coarsest level covers ~2^L pixels. We want 2^L ≈ overlapWidth.
    // Fallback to size-based if we can't estimate overlap.
    const sizeBasedLevels = Math.min(6, Math.max(3, Math.floor(Math.log2(Math.min(compW, compH))) - 3));

    // Estimate overlap width from mesh bounds
    if (meshes.size >= 2) {
      const boundsArr = Array.from(meshes.values()).map(m => m.bounds);
      let totalOverlap = 0;
      let overlapCount = 0;
      for (let a = 0; a < boundsArr.length; a++) {
        for (let b = a + 1; b < boundsArr.length; b++) {
          const oLeft = Math.max(boundsArr[a].minX, boundsArr[b].minX);
          const oRight = Math.min(boundsArr[a].maxX, boundsArr[b].maxX);
          const oTop = Math.max(boundsArr[a].minY, boundsArr[b].minY);
          const oBottom = Math.min(boundsArr[a].maxY, boundsArr[b].maxY);
          if (oRight > oLeft && oBottom > oTop) {
            const overlapW = Math.min(oRight - oLeft, oBottom - oTop);
            totalOverlap += overlapW;
            overlapCount++;
          }
        }
      }
      if (overlapCount > 0) {
        const avgOverlap = totalOverlap / overlapCount;
        // Choose levels so 2^L ≈ avgOverlap (clamped to [3, 6])
        const overlapLevels = Math.round(Math.log2(Math.max(8, avgOverlap)));
        return Math.min(6, Math.max(3, overlapLevels));
      }
    }
    return sizeBasedLevels;
  })();

  const gridN = 8;
  let imgIdx = 0;

  // ── Project faces into composite coordinates for face-aware seam placement ──
  const allFaces = getLastFaces();
  // Build a per-image list of face rects in composite coords.
  // compositeImages[imgId] = array of FaceRectComposite (in pixel coords on the composite).
  // We accumulate "composite-side" vs "new-image-side" as we stitch in order.
  const facesInCompositeCoords = new Map<string, FaceRectComposite[]>();
  for (const [imgId, faceArr] of allFaces) {
    if (!faceArr.length) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const T = t.T;
    const projected: FaceRectComposite[] = [];
    for (const face of faceArr) {
      // Project face corners through the homography to global, then to composite
      const corners = [
        [face.x, face.y],
        [face.x + face.width, face.y],
        [face.x + face.width, face.y + face.height],
        [face.x, face.y + face.height],
      ];
      let fMinX = Infinity, fMinY = Infinity, fMaxX = -Infinity, fMaxY = -Infinity;
      let valid = true;
      for (const [cx, cy] of corners) {
        const d = T[6] * cx + T[7] * cy + T[8];
        if (d < 1e-4) { valid = false; break; }
        const gx = (T[0] * cx + T[1] * cy + T[2]) / d;
        const gy = (T[3] * cx + T[4] * cy + T[5]) / d;
        const px = (gx - minX) * compositeScale;
        const py = (gy - minY) * compositeScale;
        fMinX = Math.min(fMinX, px);
        fMinY = Math.min(fMinY, py);
        fMaxX = Math.max(fMaxX, px);
        fMaxY = Math.max(fMaxY, py);
      }
      if (valid && isFinite(fMinX)) {
        projected.push({
          x: fMinX, y: fMinY,
          width: fMaxX - fMinX,
          height: fMaxY - fMinY,
          imageLabel: 1, // will be set properly during compositing
        });
      }
    }
    if (projected.length > 0) facesInCompositeCoords.set(imgId, projected);
  }

  // Pre-allocate readback buffers for compositing (reused each iteration)
  const _compPixels = new Uint8Array(compW * compH * 4);
  const _newPixels = new Uint8Array(compW * compH * 4);
  const saliencyMaps = getLastSaliency();

  for (const imgId of mstOrder) {
    const img = images.find(i => i.id === imgId);
    if (!img) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const feat = features.get(imgId);
    const sf = feat?.scaleFactor ?? 1;
    const alignW = Math.round(img.width * sf);
    const alignH = Math.round(img.height * sf);
    const T = t.T;
    const gainObj = gains.get(imgId);
    const gain: [number, number, number] = gainObj
      ? [safeGainVal(gainObj.gainR), safeGainVal(gainObj.gainG), safeGainVal(gainObj.gainB)]
      : [1.0, 1.0, 1.0];

    // Build warped mesh (APAP if available, else global transform)
    let mesh: import('./gl').MeshData;
    const apap = meshes.get(imgId);
    if (apap) {
      // Use APAP mesh (already in global coords), offset by -minX, -minY and scale
      const warpedPos = new Float32Array(apap.vertices.length);
      for (let i = 0; i < apap.vertices.length; i += 2) {
        warpedPos[i] = (apap.vertices[i] - minX) * compositeScale;
        warpedPos[i + 1] = (apap.vertices[i + 1] - minY) * compositeScale;
      }
      mesh = { positions: warpedPos, uvs: new Float32Array(apap.uvs), indices: new Uint32Array(apap.indices) };
    } else {
      // Global homography mesh — use dense grid and clamp vertices behind camera
      const baseMesh = createIdentityMesh(alignW, alignH, gridN, gridN);
      const warpedPositions = new Float32Array(baseMesh.positions.length);
      for (let i = 0; i < baseMesh.positions.length; i += 2) {
        const x = baseMesh.positions[i];
        const y = baseMesh.positions[i + 1];
        const denom = T[6] * x + T[7] * y + T[8];
        if (denom < 1e-4) {
          // Behind camera or near-singular — clamp to nearest valid position
          warpedPositions[i] = 0;
          warpedPositions[i + 1] = 0;
        } else {
          warpedPositions[i] = ((T[0] * x + T[1] * y + T[2]) / denom - minX) * compositeScale;
          warpedPositions[i + 1] = ((T[3] * x + T[4] * y + T[5]) / denom - minY) * compositeScale;
        }
      }
      baseMesh.positions = warpedPositions;
      mesh = baseMesh;
    }

    // Decode image and create texture
    const bmp = await createImageBitmap(img.file);
    const offscreen = new OffscreenCanvas(alignW, alignH);
    const ctx2d = offscreen.getContext('2d')!;
    ctx2d.drawImage(bmp, 0, 0, alignW, alignH);
    bmp.close();
    const resizedBmp = await createImageBitmap(offscreen);
    const imgTex = createTextureFromImage(gl, resizedBmp, alignW, alignH);
    resizedBmp.close();

    // Render new warped image to newImageFBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
    gl.viewport(0, 0, compW, compH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    // Pass vignetting correction coefficients and HDR tone mapping flag
    const vigParams = vignettes.get(imgId);
    const vigA = vigParams?.a ?? 0;
    const vigB = vigParams?.b ?? 0;
    // Enable Reinhard tone mapping when gain exceeds 2× to handle extreme exposure
    const needsToneMap = gain.some((g: number) => g > 2.0 || g < 0.5);
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0, vigA, vigB, needsToneMap);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    imgTex.dispose();

    if (imgIdx === 0) {
      // First image: copy directly to composite
      // Render new image to alt composite FBO
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, newImageFBO.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, currentCompFBO.fbo);
      gl.blitFramebuffer(0, 0, compW, compH, 0, 0, compW, compH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    } else if (useGraphCut) {
      // Seam finding via graph cut
      // Read back composite and new image at full resolution
      const compPixels = _compPixels;
      const newPixels = _newPixels;

      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Compute block costs
      // Collect face rects: faces in composite → label 0, faces in new image → label 1
      const seamFaces: FaceRectComposite[] = [];
      // Faces from previously composited images
      for (const prevId of mstOrder.slice(0, imgIdx)) {
        const pf = facesInCompositeCoords.get(prevId);
        if (pf) {
          for (const f of pf) seamFaces.push({ ...f, imageLabel: 0 });
        }
      }
      // Faces from current new image
      const curFaces = facesInCompositeCoords.get(imgId);
      if (curFaces) {
        for (const f of curFaces) seamFaces.push({ ...f, imageLabel: 1 });
      }
      const costs = computeBlockCosts(compPixels, newPixels, compW, compH, blockSize, 0, seamFaces,
        projectSaliencyToComposite(imgId, T, minX, minY, compositeScale, compW, compH, saliencyMaps));

      // Send to seam worker
      const dataCostsBuf = costs.dataCosts.buffer.slice(0) as ArrayBuffer;
      const edgeWeightsBuf = costs.edgeWeights.buffer.slice(0) as ArrayBuffer;
      const hardBuf = costs.hardConstraints.buffer.slice(0) as ArrayBuffer;

      const jobId = `seam-${imgId}`;
      const resultPromise = wm!.waitSeam('result', 15000) as Promise<SeamResultMsg>;

      wm!.sendSeam({
        type: 'solve',
        jobId,
        gridW: costs.gridW,
        gridH: costs.gridH,
        dataCostsBuffer: dataCostsBuf,
        edgeWeightsBuffer: edgeWeightsBuf,
        hardConstraintsBuffer: hardBuf,
        params: {},
      }, [dataCostsBuf, edgeWeightsBuf, hardBuf]);

      const seamResult = await resultPromise;
      const blockLabels = new Uint8Array(seamResult.labelsBuffer);

      // Convert block labels to pixel mask + feather
      const pixelMask = labelsToMask(blockLabels, costs.gridW, costs.gridH, blockSize, compW, compH);
      const feathered = featherMask(pixelMask, compW, compH, Math.max(1, Math.round(featherWidth * compositeScale)));

      // Upload mask as texture
      const maskTex = createMaskTexture(gl, feathered, compW, compH);

      // Blend using pyramid or simple compositor
      if (useMultiband) {
        pyramidBlender!.blend(
          currentCompTex.texture, newImageTex.texture,
          maskTex.texture, altCompFBO.fbo, compW, compH, mbLevels,
        );
      } else {
        compositor.blendWithMask(
          currentCompTex.texture, newImageTex.texture,
          maskTex.texture, altCompFBO.fbo, compW, compH,
        );
      }
      maskTex.dispose();

      // Swap composite buffers
      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
    } else {
      // Feather-only fallback: simple alpha blend
      // Create a mask where new image has alpha
      const newPixels = _newPixels;
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Create mask from new image alpha
      const alphaMask = new Uint8Array(compW * compH);
      for (let i = 0; i < compW * compH; i++) {
        alphaMask[i] = newPixels[i * 4 + 3];
      }
      const feathered = featherMask(alphaMask, compW, compH, Math.max(1, Math.round(featherWidth * compositeScale)));
      const maskTex = createMaskTexture(gl, feathered, compW, compH);

      if (useMultiband) {
        pyramidBlender!.blend(
          currentCompTex.texture, newImageTex.texture,
          maskTex.texture, altCompFBO.fbo, compW, compH, mbLevels,
        );
      } else {
        compositor.blendWithMask(
          currentCompTex.texture, newImageTex.texture,
          maskTex.texture, altCompFBO.fbo, compW, compH,
        );
      }
      maskTex.dispose();

      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
    }

    imgIdx++;
    const compPct = Math.round((imgIdx / mstOrder.length) * 100);
    setStatus(`Compositing (${compPct}%) — ${imgIdx}/${mstOrder.length}`);
    updateProgress('compositing', imgIdx / mstOrder.length);
  }

  // Display final composite on screen
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, compW, compH);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.BLEND);

  // Draw composite as fullscreen quad.
  // FBO textures have v=0 at bottom (GL convention), so flip V to display right-side up.
  const screenMesh = createIdentityMesh(compW, compH, 1, 1);
  for (let i = 1; i < screenMesh.uvs.length; i += 2) {
    screenMesh.uvs[i] = 1 - screenMesh.uvs[i];
  }
  warpRenderer.drawMesh(currentCompTex.texture, screenMesh, viewMat, 1.0, 1.0);

  // Cleanup FBOs
  } catch (err) {
    console.error('renderWarpedPreview error:', err);
    setStatus(`Pipeline error: ${err instanceof Error ? err.message : String(err)}`);
    endProgress();
    return;
  } finally {
  compositeA?.dispose(); compositeTexA?.dispose();
  compositeB?.dispose(); compositeTexB?.dispose();
  newImageFBO?.dispose(); newImageTex?.dispose();
  }

  endProgress();
  setStatus(`Composite complete — ${images.length} images blended.`);

  // Enable export button
  document.getElementById('btn-export')!.removeAttribute('disabled');
  document.getElementById('btn-export-fullres')!.removeAttribute('disabled');
}

/**
 * Export the current composite as a downloadable image file.
 * Re-renders at export scale, crops transparent edges, downloads via toBlob.
 */
async function exportComposite(): Promise<void> {
  if (!glCtx || !warpRenderer || !compositor) return;
  const { gl } = glCtx;
  const { images: allImages, settings } = getState();
  const active = allImages.filter(i => !i.excluded);
  const transforms = getLastTransforms();
  const features = getLastFeatures();
  const gains = getLastGains();
  const meshes = getLastMeshes();
  const refId = getLastRefId();

  if (active.length === 0 || transforms.size === 0 || !settings) {
    setStatus('Nothing to export. Run Stitch Preview first.');
    return;
  }

  setStatus('Exporting…');
  const exportFormat = settings.exportFormat;
  const jpegQuality = settings.exportJpegQuality;
  const mstParent = getLastMstParent();

  // Max resolution export: compute effective export scale as 1/alignment-scale
  // so each source pixel maps to ~1 output pixel (no downscale).
  // For normal export, use the preset's exportScale.
  let exportScale: number;
  if (settings.maxResExport) {
    // Find the alignment scale factor used during feature extraction.
    // The alignment dimensions are alignScale (e.g. 1536) long-edge pixels.
    // So sf = alignScale / max(origW, origH). Export at full res = 1/sf.
    const refImg = active.find(i => i.id === refId) ?? active[0];
    const refFeat = features.get(refImg.id);
    const sf = refFeat?.scaleFactor ?? 1;
    exportScale = 1 / sf;  // e.g. if sf=0.5, exportScale=2.0 → double alignment space
  } else {
    exportScale = settings.exportScale;
  }

  // Determine connected images (reachable from reference in MST)
  const connectedIds = new Set<string>();
  if (refId) connectedIds.add(refId);
  for (const [id, parentId] of Object.entries(mstParent)) {
    if (parentId !== null) connectedIds.add(id);
  }
  if (connectedIds.size === 0) {
    for (const img of active) {
      if (transforms.has(img.id)) connectedIds.add(img.id);
    }
  }

  // Compute reasonable max extent for sanity clamping
  let refDim = 1536;
  for (const img of active) {
    if (img.id === refId) {
      const feat = features.get(img.id);
      const sf = feat?.scaleFactor ?? 1;
      refDim = Math.max(img.width * sf, img.height * sf);
      break;
    }
  }
  const maxReasonableExtent = refDim * active.length * 2;

  // Compute global bounding box at alignment scale
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const img of active) {
    if (!connectedIds.has(img.id)) continue;
    const t = transforms.get(img.id);
    if (!t) continue;

    // Prefer APAP mesh bounds (clamped to reasonable extent)
    const apap = meshes.get(img.id);
    if (apap) {
      const b = apap.bounds;
      const clampedMinX = Math.max(b.minX, -maxReasonableExtent);
      const clampedMinY = Math.max(b.minY, -maxReasonableExtent);
      const clampedMaxX = Math.min(b.maxX, maxReasonableExtent);
      const clampedMaxY = Math.min(b.maxY, maxReasonableExtent);
      if (clampedMaxX > clampedMinX && clampedMaxY > clampedMinY) {
        minX = Math.min(minX, clampedMinX); minY = Math.min(minY, clampedMinY);
        maxX = Math.max(maxX, clampedMaxX); maxY = Math.max(maxY, clampedMaxY);
      }
      continue;
    }

    const feat = features.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    const w = img.width * sf;
    const h = img.height * sf;
    const T = t.T;
    const samples = [
      [0, 0], [w, 0], [w, h], [0, h],
      [w / 2, 0], [w, h / 2], [w / 2, h], [0, h / 2], [w / 2, h / 2],
    ];
    for (const [cx, cy] of samples) {
      const denom = T[6] * cx + T[7] * cy + T[8];
      if (denom < 1e-4) continue;
      const gx = (T[0] * cx + T[1] * cy + T[2]) / denom;
      const gy = (T[3] * cx + T[4] * cy + T[5]) / denom;
      if (Math.abs(gx) > maxReasonableExtent || Math.abs(gy) > maxReasonableExtent) continue;
      minX = Math.min(minX, gx); minY = Math.min(minY, gy);
      maxX = Math.max(maxX, gx); maxY = Math.max(maxY, gy);
    }
  }
  if (!isFinite(minX)) return;

  const globalW = maxX - minX;
  const globalH = maxY - minY;
  const maxTexSize = glCtx.maxTextureSize;
  let outW = Math.round(globalW * exportScale);
  let outH = Math.round(globalH * exportScale);
  if (outW > maxTexSize || outH > maxTexSize) {
    const clampScale = maxTexSize / Math.max(outW, outH);
    outW = Math.round(outW * clampScale);
    outH = Math.round(outH * clampScale);
    setStatus(`Export clamped to ${outW}×${outH} (GPU max ${maxTexSize}px)`);
  } else if (settings.maxResExport) {
    setStatus(`Exporting at full resolution: ${outW}×${outH}…`);
  }
  const compositeScale = outW / globalW;

  // Create FBOs for export rendering — inside try so finally always cleans up
  let compositeTexA: import('./gl').ManagedTexture | null = null;
  let compositeTexB: import('./gl').ManagedTexture | null = null;
  let compositeA: import('./gl').ManagedFBO | null = null;
  let compositeB: import('./gl').ManagedFBO | null = null;
  let newImageTex: import('./gl').ManagedTexture | null = null;
  let newImageFBO: import('./gl').ManagedFBO | null = null;

  try {
  compositeTexA = createEmptyTexture(gl, outW, outH);
  compositeTexB = createEmptyTexture(gl, outW, outH);
  compositeA = createFBO(gl, compositeTexA.texture);
  compositeB = createFBO(gl, compositeTexB.texture);
  newImageTex = createEmptyTexture(gl, outW, outH);
  newImageFBO = createFBO(gl, newImageTex.texture);

  const compViewMat = new Float32Array([2 / outW, 0, 0, 0, -2 / outH, 0, -1, 1, 1]);

  let currentCompTex = compositeTexA;
  let currentCompFBO = compositeA;
  let altCompTex = compositeTexB;
  let altCompFBO = compositeB;

  gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
  gl.viewport(0, 0, outW, outH);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  const mstOrder = (() => {
    const order = getLastMstOrder();
    if (order.length > 0) return order.filter(id => connectedIds.has(id));
    const ids = active.map(i => i.id).filter(id => connectedIds.has(id));
    if (refId && ids.includes(refId)) return [refId, ...ids.filter(id => id !== refId)];
    return ids;
  })();

  const blockSize = settings.seamBlockSize;
  const featherWidth = settings.featherWidth;
  const useMultiband = settings.multibandEnabled && pyramidBlender !== null;
  const mbLevels = (() => {
    if (settings.multibandLevels > 0) return settings.multibandLevels;
    // Adaptive: estimate overlap width from mesh bounds (same logic as preview)
    const sizeBasedLevels = Math.min(6, Math.max(3, Math.floor(Math.log2(Math.min(outW, outH))) - 3));
    if (meshes.size >= 2) {
      const boundsArr = Array.from(meshes.values()).map(m => m.bounds);
      let totalOverlap = 0, overlapCount = 0;
      for (let a = 0; a < boundsArr.length; a++) {
        for (let b = a + 1; b < boundsArr.length; b++) {
          const oLeft = Math.max(boundsArr[a].minX, boundsArr[b].minX);
          const oRight = Math.min(boundsArr[a].maxX, boundsArr[b].maxX);
          const oTop = Math.max(boundsArr[a].minY, boundsArr[b].minY);
          const oBottom = Math.min(boundsArr[a].maxY, boundsArr[b].maxY);
          if (oRight > oLeft && oBottom > oTop) {
            // Use export scale to adjust overlap to output pixel space
            const overlapW = Math.min(oRight - oLeft, oBottom - oTop) * exportScale;
            totalOverlap += overlapW;
            overlapCount++;
          }
        }
      }
      if (overlapCount > 0) {
        const avgOverlap = totalOverlap / overlapCount;
        return Math.min(6, Math.max(3, Math.round(Math.log2(Math.max(8, avgOverlap)))));
      }
    }
    return sizeBasedLevels;
  })();
  const wm = getWorkerManager();
  const useGraphCut = settings.seamMethod === 'graphcut' && wm !== null;

  let imgIdx = 0;
  const gridN = 8;

  // ── Project faces into export composite coordinates ──
  const allFacesExport = getLastFaces();
  const facesInExportCoords = new Map<string, FaceRectComposite[]>();
  for (const [imgId, faceArr] of allFacesExport) {
    if (!faceArr.length) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const T = t.T;
    const projected: FaceRectComposite[] = [];
    for (const face of faceArr) {
      const corners = [
        [face.x, face.y],
        [face.x + face.width, face.y],
        [face.x + face.width, face.y + face.height],
        [face.x, face.y + face.height],
      ];
      let fMinX = Infinity, fMinY = Infinity, fMaxX = -Infinity, fMaxY = -Infinity;
      let valid = true;
      for (const [cx, cy] of corners) {
        const d = T[6] * cx + T[7] * cy + T[8];
        if (d < 1e-4) { valid = false; break; }
        const gx = (T[0] * cx + T[1] * cy + T[2]) / d;
        const gy = (T[3] * cx + T[4] * cy + T[5]) / d;
        const px = (gx - minX) * compositeScale;
        const py = (gy - minY) * compositeScale;
        fMinX = Math.min(fMinX, px);
        fMinY = Math.min(fMinY, py);
        fMaxX = Math.max(fMaxX, px);
        fMaxY = Math.max(fMaxY, py);
      }
      if (valid && isFinite(fMinX)) {
        projected.push({
          x: fMinX, y: fMinY,
          width: fMaxX - fMinX,
          height: fMaxY - fMinY,
          imageLabel: 1,
        });
      }
    }
    if (projected.length > 0) facesInExportCoords.set(imgId, projected);
  }

  // Saliency maps + pre-allocate readback buffers for export compositing
  const exportSaliencyMaps = getLastSaliency();
  const _expCompPixels = new Uint8Array(outW * outH * 4);
  const _expNewPixels = new Uint8Array(outW * outH * 4);

  for (const imgId of mstOrder) {
    const img = active.find(i => i.id === imgId);
    if (!img) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const feat = features.get(imgId);
    const sf = feat?.scaleFactor ?? 1;
    const alignW = Math.round(img.width * sf);
    const alignH = Math.round(img.height * sf);
    const T = t.T;
    const gainObj = gains.get(imgId);
    const gain: [number, number, number] = gainObj
      ? [safeGainVal(gainObj.gainR), safeGainVal(gainObj.gainG), safeGainVal(gainObj.gainB)]
      : [1.0, 1.0, 1.0];

    // Build warped mesh
    let mesh: import('./gl').MeshData;
    const apap = meshes.get(imgId);
    if (apap) {
      const warpedPos = new Float32Array(apap.vertices.length);
      for (let i = 0; i < apap.vertices.length; i += 2) {
        warpedPos[i] = (apap.vertices[i] - minX) * compositeScale;
        warpedPos[i + 1] = (apap.vertices[i + 1] - minY) * compositeScale;
      }
      mesh = { positions: warpedPos, uvs: new Float32Array(apap.uvs), indices: new Uint32Array(apap.indices) };
    } else {
      const baseMesh = createIdentityMesh(alignW, alignH, gridN, gridN);
      const warpedPositions = new Float32Array(baseMesh.positions.length);
      for (let i = 0; i < baseMesh.positions.length; i += 2) {
        const x = baseMesh.positions[i], y = baseMesh.positions[i + 1];
        const denom = T[6] * x + T[7] * y + T[8];
        if (denom < 1e-4) {
          warpedPositions[i] = 0;
          warpedPositions[i + 1] = 0;
        } else {
          warpedPositions[i] = ((T[0] * x + T[1] * y + T[2]) / denom - minX) * compositeScale;
          warpedPositions[i + 1] = ((T[3] * x + T[4] * y + T[5]) / denom - minY) * compositeScale;
        }
      }
      baseMesh.positions = warpedPositions;
      mesh = baseMesh;
    }

    // Decode image — full original resolution for maxRes, alignment scale otherwise
    const bmp = await createImageBitmap(img.file);
    let texW: number, texH: number;
    if (settings.maxResExport) {
      // Full resolution: use original image dimensions (no downscale)
      texW = bmp.width;
      texH = bmp.height;
    } else {
      texW = alignW;
      texH = alignH;
    }
    const off = new OffscreenCanvas(texW, texH);
    const ctx2d = off.getContext('2d')!;
    ctx2d.drawImage(bmp, 0, 0, texW, texH);
    bmp.close();
    const resizedBmp = await createImageBitmap(off);
    const imgTex = createTextureFromImage(gl, resizedBmp, texW, texH);
    resizedBmp.close();

    // Warp to newImageFBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
    gl.viewport(0, 0, outW, outH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    // Vignetting correction + HDR tone mapping for export
    const expVigParams = getLastVignette().get(imgId);
    const expVigA = expVigParams?.a ?? 0;
    const expVigB = expVigParams?.b ?? 0;
    const expNeedsToneMap = gain.some((g: number) => g > 2.0 || g < 0.5);
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0, expVigA, expVigB, expNeedsToneMap);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    imgTex.dispose();

    if (imgIdx === 0) {
      // First image — just copy
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, newImageFBO.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, currentCompFBO.fbo);
      gl.blitFramebuffer(0, 0, outW, outH, 0, 0, outW, outH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
    } else if (useGraphCut) {
      // Graph cut seam finding — reuse pre-allocated buffers
      const compPixels = _expCompPixels;
      const newPixels = _expNewPixels;
      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const seamFacesExport: FaceRectComposite[] = [];
      for (const prevId of mstOrder.slice(0, imgIdx)) {
        const pf = facesInExportCoords.get(prevId);
        if (pf) {
          for (const f of pf) seamFacesExport.push({ ...f, imageLabel: 0 });
        }
      }
      const curFacesExport = facesInExportCoords.get(imgId);
      if (curFacesExport) {
        for (const f of curFacesExport) seamFacesExport.push({ ...f, imageLabel: 1 });
      }
      const costs = computeBlockCosts(compPixels, newPixels, outW, outH, blockSize, 0, seamFacesExport,
        projectSaliencyToComposite(imgId, T, minX, minY, compositeScale, outW, outH, exportSaliencyMaps));
      const dataCostsBuf = costs.dataCosts.buffer.slice(0) as ArrayBuffer;
      const edgeWeightsBuf = costs.edgeWeights.buffer.slice(0) as ArrayBuffer;
      const hardBuf = costs.hardConstraints.buffer.slice(0) as ArrayBuffer;

      const resultPromise = wm!.waitSeam('result', 30000) as Promise<SeamResultMsg>;
      wm!.sendSeam({
        type: 'solve', jobId: `export-${imgId}`,
        gridW: costs.gridW, gridH: costs.gridH,
        dataCostsBuffer: dataCostsBuf, edgeWeightsBuffer: edgeWeightsBuf,
        hardConstraintsBuffer: hardBuf, params: {},
      }, [dataCostsBuf, edgeWeightsBuf, hardBuf]);

      const seamResult = await resultPromise;
      const blockLabels = new Uint8Array(seamResult.labelsBuffer);
      const pixelMask = labelsToMask(blockLabels, costs.gridW, costs.gridH, blockSize, outW, outH);
      const feathered = featherMask(pixelMask, outW, outH, Math.max(1, Math.round(featherWidth * compositeScale)));
      const maskTex = createMaskTexture(gl, feathered, outW, outH);

      if (useMultiband) {
        pyramidBlender!.blend(currentCompTex.texture, newImageTex.texture, maskTex.texture, altCompFBO.fbo, outW, outH, mbLevels);
      } else {
        compositor.blendWithMask(currentCompTex.texture, newImageTex.texture, maskTex.texture, altCompFBO.fbo, outW, outH);
      }
      maskTex.dispose();
      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
    } else {
      // Feather-only fallback — reuse pre-allocated buffer
      const newPixels = _expNewPixels;
      newPixels.fill(0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const alphaMask = new Uint8Array(outW * outH);
      for (let i = 0; i < outW * outH; i++) alphaMask[i] = newPixels[i * 4 + 3];
      const feathered = featherMask(alphaMask, outW, outH, Math.max(1, Math.round(featherWidth * compositeScale)));
      const maskTex = createMaskTexture(gl, feathered, outW, outH);

      if (useMultiband) {
        pyramidBlender!.blend(currentCompTex.texture, newImageTex.texture, maskTex.texture, altCompFBO.fbo, outW, outH, mbLevels);
      } else {
        compositor.blendWithMask(currentCompTex.texture, newImageTex.texture, maskTex.texture, altCompFBO.fbo, outW, outH);
      }
      maskTex.dispose();
      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
    }

    imgIdx++;
    setStatus(`Export: ${imgIdx}/${mstOrder.length}`);
  }

  // Read back final composite
  setStatus('Encoding…');
  const pixels = new Uint8Array(outW * outH * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
  gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // WebGL reads bottom-up — flip vertically
  const flipped = new Uint8ClampedArray(outW * outH * 4);
  for (let y = 0; y < outH; y++) {
    const srcRow = (outH - 1 - y) * outW * 4;
    const dstRow = y * outW * 4;
    flipped.set(pixels.subarray(srcRow, srcRow + outW * 4), dstRow);
  }

  // Auto-crop transparent edges
  let cropMinX = outW, cropMinY = outH, cropMaxX = 0, cropMaxY = 0;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      if (flipped[(y * outW + x) * 4 + 3] > 10) {
        cropMinX = Math.min(cropMinX, x);
        cropMinY = Math.min(cropMinY, y);
        cropMaxX = Math.max(cropMaxX, x);
        cropMaxY = Math.max(cropMaxY, y);
      }
    }
  }

  const exportCanvas = new OffscreenCanvas(outW, outH);
  const exportCtx = exportCanvas.getContext('2d')!;
  exportCtx.putImageData(new ImageData(flipped, outW, outH), 0, 0);

  let finalCanvas: OffscreenCanvas;
  if (cropMaxX > cropMinX && cropMaxY > cropMinY) {
    const cw = cropMaxX - cropMinX + 1;
    const ch = cropMaxY - cropMinY + 1;
    finalCanvas = new OffscreenCanvas(cw, ch);
    const fctx = finalCanvas.getContext('2d')!;
    fctx.drawImage(exportCanvas, cropMinX, cropMinY, cw, ch, 0, 0, cw, ch);
  } else {
    finalCanvas = exportCanvas;
  }

  const mimeType = exportFormat === 'jpeg' ? 'image/jpeg' : 'image/png';
  const quality = exportFormat === 'jpeg' ? jpegQuality : undefined;
  const blob = await finalCanvas.convertToBlob({ type: mimeType, quality });

  // Trigger download
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = `mosaic.${exportFormat === 'jpeg' ? 'jpg' : 'png'}`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);

  setStatus(`Exported ${finalCanvas.width}×${finalCanvas.height} ${exportFormat.toUpperCase()}.`);

  // Cleanup
  } finally {
  compositeA?.dispose(); compositeTexA?.dispose();
  compositeB?.dispose(); compositeTexB?.dispose();
  newImageFBO?.dispose(); newImageTex?.dispose();
  }
}

boot().catch(err => {
  console.error('Boot failed:', err);
  document.getElementById('status-bar')!.textContent = `Error: ${err.message}`;
});
