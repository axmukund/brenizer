import { detectCapabilities } from './capabilities';
import { resolveMode, getPreset } from './presets';
import { setState, getState, subscribe } from './appState';
import { initUI, renderCapabilities, setStatus, setRenderImagePreview } from './ui';
import { createGLContext, createWarpRenderer, createKeypointRenderer, createIdentityMesh, createTextureFromImage, makeViewMatrix, type GLContext, type WarpRenderer, type KeypointRenderer } from './gl';
import { runStitchPreview, getLastFeatures, getLastEdges, getLastTransforms, getLastRefId, getLastGains } from './pipelineController';

let glCtx: GLContext | null = null;
let warpRenderer: WarpRenderer | null = null;
let kpRenderer: KeypointRenderer | null = null;

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
  const alignW = Math.round(origW * sf);
  const alignH = Math.round(origH * sf);

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

  // Draw keypoint overlay after feature extraction completes
  window.addEventListener('features-ready', async () => {
    const { images } = getState();
    const active = images.filter(i => !i.excluded);
    if (active.length === 0 || !glCtx || !warpRenderer) return;

    // Re-render the first image then overlay keypoints for all
    const first = active[0];
    await renderImagePreview(first);

    for (const img of active) {
      renderKeypointOverlay(img.id, img.width, img.height);
    }
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
  const shortNames = images.map(i => i.name.replace(/\.[^.]+$/, '').slice(0, 8));
  let html = '<div style="margin-top:8px;font-size:11px;"><strong>Inlier Matrix</strong>';
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

/**
 * Render a warped multi-image preview using global transforms.
 * Each image is drawn with its 3x3 homography applied through an identity mesh
 * whose vertices are transformed to global coords.
 */
async function renderWarpedPreview(
  images: import('./appState').ImageEntry[],
  transforms: Map<string, import('./pipelineController').GlobalTransform>,
): Promise<void> {
  if (!glCtx || !warpRenderer) return;
  const { gl, canvas } = glCtx;
  const features = getLastFeatures();
  const gains = getLastGains();

  // Size canvas to container
  const container = document.getElementById('canvas-container')!;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  canvas.style.display = 'block';
  document.getElementById('canvas-placeholder')!.style.display = 'none';

  // Compute bounding box across all warped image corners in global coords
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (const img of images) {
    const t = transforms.get(img.id);
    if (!t) continue;
    const feat = features.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    const w = img.width * sf;
    const h = img.height * sf;
    const T = t.T;
    // Transform 4 corners through T (row-major 3x3)
    const corners = [[0, 0], [w, 0], [w, h], [0, h]];
    for (const [cx, cy] of corners) {
      const denom = T[6] * cx + T[7] * cy + T[8];
      if (Math.abs(denom) < 1e-10) continue;
      const gx = (T[0] * cx + T[1] * cy + T[2]) / denom;
      const gy = (T[3] * cx + T[4] * cy + T[5]) / denom;
      minX = Math.min(minX, gx);
      minY = Math.min(minY, gy);
      maxX = Math.max(maxX, gx);
      maxY = Math.max(maxY, gy);
    }
  }

  if (!isFinite(minX)) return;

  const globalW = maxX - minX;
  const globalH = maxY - minY;

  // View matrix: maps global coords → clip space
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, globalW, globalH);

  // Clear canvas
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // Draw each image warped into global space
  const gridN = 8; // grid subdivision for mesh (helps with projective distortion)
  for (const img of images) {
    const t = transforms.get(img.id);
    if (!t) continue;
    const feat = features.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    const alignW = Math.round(img.width * sf);
    const alignH = Math.round(img.height * sf);
    const T = t.T;

    // Build a mesh in image coords, project vertices through T, then offset by -minX/-minY
    const mesh = createIdentityMesh(alignW, alignH, gridN, gridN);
    const warpedPositions = new Float32Array(mesh.positions.length);
    for (let i = 0; i < mesh.positions.length; i += 2) {
      const x = mesh.positions[i];
      const y = mesh.positions[i + 1];
      const denom = T[6] * x + T[7] * y + T[8];
      if (Math.abs(denom) < 1e-10) {
        warpedPositions[i] = x;
        warpedPositions[i + 1] = y;
      } else {
        warpedPositions[i] = (T[0] * x + T[1] * y + T[2]) / denom - minX;
        warpedPositions[i + 1] = (T[3] * x + T[4] * y + T[5]) / denom - minY;
      }
    }
    mesh.positions = warpedPositions;

    // Decode image and create texture
    const bmp = await createImageBitmap(img.file);
    // Resize to alignment scale
    const offscreen = new OffscreenCanvas(alignW, alignH);
    const ctx2d = offscreen.getContext('2d')!;
    ctx2d.drawImage(bmp, 0, 0, alignW, alignH);
    bmp.close();
    const resizedBmp = await createImageBitmap(offscreen);
    const tex = createTextureFromImage(gl, resizedBmp, alignW, alignH);
    resizedBmp.close();

    warpRenderer.drawMesh(tex.texture, mesh, viewMat, gains.get(img.id) ?? 1.0, 0.7);
    tex.dispose();
  }

  gl.disable(gl.BLEND);
}

boot().catch(err => {
  console.error('Boot failed:', err);
  document.getElementById('status-bar')!.textContent = `Error: ${err.message}`;
});
