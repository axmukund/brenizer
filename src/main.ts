import { detectCapabilities } from './capabilities';
import { resolveMode, getPreset } from './presets';
import { setState, getState, subscribe } from './appState';
import { initUI, renderCapabilities, setStatus, setRenderImagePreview } from './ui';
import {
  createGLContext, createWarpRenderer, createKeypointRenderer, createCompositor,
  createPyramidBlender,
  createIdentityMesh, createTextureFromImage, createEmptyTexture, createFBO,
  makeViewMatrix, computeBlockCosts, labelsToMask, featherMask, createMaskTexture,
  type GLContext, type WarpRenderer, type KeypointRenderer, type Compositor,
  type PyramidBlender, type ManagedTexture,
} from './gl';
import {
  runStitchPreview, getLastFeatures, getLastEdges, getLastTransforms, getLastRefId,
  getLastGains, getLastMeshes, getLastMstOrder, getWorkerManager,
} from './pipelineController';
import type { SeamResultMsg } from './workers/workerTypes';

let glCtx: GLContext | null = null;
let warpRenderer: WarpRenderer | null = null;
let kpRenderer: KeypointRenderer | null = null;
let compositor: Compositor | null = null;
let pyramidBlender: PyramidBlender | null = null;

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
    compositor = createCompositor(glCtx.gl);
    pyramidBlender = createPyramidBlender(glCtx.gl);
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

  // Wire Export button
  document.getElementById('btn-export')!.addEventListener('click', () => {
    exportComposite().catch(err => {
      console.error('Export error:', err);
      setStatus(`Export error: ${err.message}`);
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
  const { settings } = getState();
  const refId = getLastRefId();
  const wm = getWorkerManager();

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

  // Clamp composite size to safe GPU limits
  const maxTexSize = glCtx.maxTextureSize;
  const compositeScale = Math.min(1, maxTexSize / Math.max(globalW, globalH));
  const compW = Math.round(globalW * compositeScale);
  const compH = Math.round(globalH * compositeScale);

  // Create FBOs for composition
  const compositeTexA = createEmptyTexture(gl, compW, compH);
  const compositeTexB = createEmptyTexture(gl, compW, compH);
  const compositeA = createFBO(gl, compositeTexA.texture);
  const compositeB = createFBO(gl, compositeTexB.texture);
  const newImageTex = createEmptyTexture(gl, compW, compH);
  const newImageFBO = createFBO(gl, newImageTex.texture);

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

  // Clear composite
  gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
  gl.viewport(0, 0, compW, compH);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // Determine MST order, falling back to image order with ref first
  const mstOrder = (() => {
    const order: string[] = getLastMstOrder();
    if (order.length > 0) return order;
    // Fallback: ref first, then rest
    const ids = images.map(i => i.id);
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
    // Auto: based on composite size
    return Math.min(6, Math.max(3, Math.floor(Math.log2(Math.min(compW, compH))) - 3));
  })();

  const gridN = 8;
  let imgIdx = 0;

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
    const gain = gains.get(imgId) ?? 1.0;

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
      // Global homography mesh
      const baseMesh = createIdentityMesh(alignW, alignH, gridN, gridN);
      const warpedPositions = new Float32Array(baseMesh.positions.length);
      for (let i = 0; i < baseMesh.positions.length; i += 2) {
        const x = baseMesh.positions[i];
        const y = baseMesh.positions[i + 1];
        const denom = T[6] * x + T[7] * y + T[8];
        if (Math.abs(denom) < 1e-10) {
          warpedPositions[i] = (x - minX) * compositeScale;
          warpedPositions[i + 1] = (y - minY) * compositeScale;
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
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0);
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
      const compPixels = new Uint8Array(compW * compH * 4);
      const newPixels = new Uint8Array(compW * compH * 4);

      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Compute block costs
      const costs = computeBlockCosts(compPixels, newPixels, compW, compH, blockSize);

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
      const feathered = featherMask(pixelMask, compW, compH, featherWidth / compositeScale);

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
      const newPixels = new Uint8Array(compW * compH * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Create mask from new image alpha
      const alphaMask = new Uint8Array(compW * compH);
      for (let i = 0; i < compW * compH; i++) {
        alphaMask[i] = newPixels[i * 4 + 3];
      }
      const feathered = featherMask(alphaMask, compW, compH, featherWidth / compositeScale);
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
    setStatus(`Compositing: ${imgIdx}/${mstOrder.length}`);
  }

  // Display final composite on screen
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, compW, compH);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.disable(gl.BLEND);

  // Draw composite as fullscreen quad
  const screenMesh = createIdentityMesh(compW, compH, 1, 1);
  warpRenderer.drawMesh(currentCompTex.texture, screenMesh, viewMat, 1.0, 1.0);

  // Cleanup FBOs
  compositeA.dispose(); compositeTexA.dispose();
  compositeB.dispose(); compositeTexB.dispose();
  newImageFBO.dispose(); newImageTex.dispose();

  setStatus(`Composite complete — ${images.length} images blended.`);

  // Enable export button
  document.getElementById('btn-export')!.removeAttribute('disabled');
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
  const exportScale = settings.exportScale;
  const exportFormat = settings.exportFormat;
  const jpegQuality = settings.exportJpegQuality;

  // Compute global bounding box at alignment scale
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const img of active) {
    const t = transforms.get(img.id);
    if (!t) continue;
    const feat = features.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    const w = img.width * sf;
    const h = img.height * sf;
    const T = t.T;
    for (const [cx, cy] of [[0, 0], [w, 0], [w, h], [0, h]]) {
      const denom = T[6] * cx + T[7] * cy + T[8];
      if (Math.abs(denom) < 1e-10) continue;
      minX = Math.min(minX, (T[0] * cx + T[1] * cy + T[2]) / denom);
      minY = Math.min(minY, (T[3] * cx + T[4] * cy + T[5]) / denom);
      maxX = Math.max(maxX, (T[0] * cx + T[1] * cy + T[2]) / denom);
      maxY = Math.max(maxY, (T[3] * cx + T[4] * cy + T[5]) / denom);
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
    setStatus(`Export clamped to ${outW}×${outH} (GPU max ${maxTexSize})`);
  }
  const compositeScale = outW / globalW;

  // Create FBOs for export rendering
  const compositeTexA = createEmptyTexture(gl, outW, outH);
  const compositeTexB = createEmptyTexture(gl, outW, outH);
  const compositeA = createFBO(gl, compositeTexA.texture);
  const compositeB = createFBO(gl, compositeTexB.texture);
  const newImageTex = createEmptyTexture(gl, outW, outH);
  const newImageFBO = createFBO(gl, newImageTex.texture);

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
    if (order.length > 0) return order;
    const ids = active.map(i => i.id);
    if (refId && ids.includes(refId)) return [refId, ...ids.filter(id => id !== refId)];
    return ids;
  })();

  const blockSize = settings.seamBlockSize;
  const featherWidth = settings.featherWidth;
  const useMultiband = settings.multibandEnabled && pyramidBlender !== null;
  const mbLevels = settings.multibandLevels > 0
    ? settings.multibandLevels
    : Math.min(6, Math.max(3, Math.floor(Math.log2(Math.min(outW, outH))) - 3));
  const wm = getWorkerManager();
  const useGraphCut = settings.seamMethod === 'graphcut' && wm !== null;

  let imgIdx = 0;
  const gridN = 8;

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
    const gain = gains.get(imgId) ?? 1.0;

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
        if (Math.abs(denom) < 1e-10) {
          warpedPositions[i] = (x - minX) * compositeScale;
          warpedPositions[i + 1] = (y - minY) * compositeScale;
        } else {
          warpedPositions[i] = ((T[0] * x + T[1] * y + T[2]) / denom - minX) * compositeScale;
          warpedPositions[i + 1] = ((T[3] * x + T[4] * y + T[5]) / denom - minY) * compositeScale;
        }
      }
      baseMesh.positions = warpedPositions;
      mesh = baseMesh;
    }

    // Decode image at alignment scale
    const bmp = await createImageBitmap(img.file);
    const off = new OffscreenCanvas(alignW, alignH);
    const ctx2d = off.getContext('2d')!;
    ctx2d.drawImage(bmp, 0, 0, alignW, alignH);
    bmp.close();
    const resizedBmp = await createImageBitmap(off);
    const imgTex = createTextureFromImage(gl, resizedBmp, alignW, alignH);
    resizedBmp.close();

    // Warp to newImageFBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
    gl.viewport(0, 0, outW, outH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0);
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
      // Graph cut seam finding
      const compPixels = new Uint8Array(outW * outH * 4);
      const newPixels = new Uint8Array(outW * outH * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const costs = computeBlockCosts(compPixels, newPixels, outW, outH, blockSize);
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
      const feathered = featherMask(pixelMask, outW, outH, featherWidth);
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
      // Feather-only fallback
      const newPixels = new Uint8Array(outW * outH * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      const alphaMask = new Uint8Array(outW * outH);
      for (let i = 0; i < outW * outH; i++) alphaMask[i] = newPixels[i * 4 + 3];
      const feathered = featherMask(alphaMask, outW, outH, featherWidth);
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

  // Cleanup
  compositeA.dispose(); compositeTexA.dispose();
  compositeB.dispose(); compositeTexB.dispose();
  newImageFBO.dispose(); newImageTex.dispose();

  setStatus(`Exported ${finalCanvas.width}×${finalCanvas.height} ${exportFormat.toUpperCase()}.`);
}

boot().catch(err => {
  console.error('Boot failed:', err);
  document.getElementById('status-bar')!.textContent = `Error: ${err.message}`;
});
