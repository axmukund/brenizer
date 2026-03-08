/**
 * main.ts — Application entry point and rendering pipeline.
 *
 * Responsibilities:
 *  - WebGL2 context initialization (canvas, shaders, compositors)
 *  - Image preview rendering (single image + keypoint overlay)
 *  - Warped multi-image compositing (APAP mesh warp → seam find → blend)
 *  - Full-resolution export with auto-crop
 *  - UI event wiring (stitch button, export, tabs, drag-drop)
 *
 * Compositing pipeline (per image in MST order):
 *  1. Decode image → resize to alignment scale → upload as GPU texture
 *  2. Warp through APAP mesh (or global homography) into composite space
 *  3. Read back composite + warped pixels for CPU-side seam cost computation
 *  4. Solve graph-cut min-cut via seam-worker (or feather-only fallback)
 *  5. Convert block labels → pixel mask → feather → upload mask texture
 *  6. Blend into composite via Laplacian pyramid (or simple mask blend)
 *  7. Ping-pong between two FBOs to avoid read-write hazard
 *
 * The same pipeline is used for both preview and export, with export using
 * a separate exportScale (or full-resolution when maxResExport is enabled).
 */
import { detectCapabilities, type Capabilities } from './capabilities';
import { resolveMode, getPreset } from './presets';
import { setState, getState, subscribe } from './appState';
import { initUI, renderCapabilities, setStatus, setRenderImagePreview, startProgress, endProgress, updateProgress, buildSettingsPanel, promptOptimizeCameraSettingsChoice } from './ui';
import {
  createGLContext, createWarpRenderer, createKeypointRenderer, createCompositor,
  createPyramidBlender,
  createIdentityMesh, createTextureFromImage, createEmptyTexture, createFBO,
  makeViewMatrix, computeBlockCosts, labelsToMask, buildAdaptiveBlendMask,
  createMaskTexture, createSeamAccelerator,
  type GLContext, type WarpRenderer, type KeypointRenderer, type Compositor,
  type PyramidBlender, type ManagedTexture, type FaceRectComposite, type MeshData,
  type SeamAccelerator, type SeamAccelerationTier, type CompactSeamGraphBuildResult,
} from './gl';
import {
  runStitchPreview, getLastFeatures, getLastEdges, getLastTransforms, getLastRefId,
  runFirstPassOptimization,
  getLastGains, getLastMeshes, getLastMstOrder, getLastMstParent, getWorkerManager,
  getLastFaces, getLastSaliency, getLastVignette,
} from './pipelineController';
import type { SeamResultMsg } from './workers/workerTypes';
import { getTurboModePreference } from './runtimeAcceleration';
import {
  createWebGPUSeamBuilder,
  type WebGPUSeamBuilder,
  type WebGPUSeamCompositeState,
} from './webgpu/seamBuilder';

let glCtx: GLContext | null = null;
let warpRenderer: WarpRenderer | null = null;
let kpRenderer: KeypointRenderer | null = null;
let compositor: Compositor | null = null;
let pyramidBlender: PyramidBlender | null = null;
let seamAccelerator: SeamAccelerator | null = null;
let webgpuSeamBuilder: WebGPUSeamBuilder | null = null;
let editorZoom = 1.0;
let editorRotationDeg = 0.0;
let editorPanX = 0.0;
let editorPanY = 0.0;
let editorPanPointerId: number | null = null;
let editorPanStartX = 0.0;
let editorPanStartY = 0.0;
let editorPanBaseX = 0.0;
let editorPanBaseY = 0.0;

interface PreviewOutlinePolygon {
  points: Array<[number, number]>; // display-content coordinates (before canvas fit)
}

interface PreviewDisplayRegion {
  dispW: number;
  dispH: number;
}

interface SeamJobMetrics {
  stage: 'preview' | 'export';
  imageId: string;
  tier: SeamAccelerationTier;
  builderBackend: string;
  solverBackend: string;
  gridW: number;
  gridH: number;
  blockSize: number;
  readbackBytes: number;
  summaryMs: number;
  graphBuildMs: number;
  solverMs: number;
  maskMs: number;
  totalMs: number;
}

let previewImageOutlines = new Map<string, PreviewOutlinePolygon>();
let previewDisplayRegion: PreviewDisplayRegion | null = null;
let hoveredPreviewImageId: string | null = null;

declare global {
  interface Window {
    __brenizerPerf?: {
      seamJobs: SeamJobMetrics[];
    };
    __brenizerRuntime?: {
      caps: Capabilities | null;
      turboModeEnabled: boolean;
    };
  }
}

const EXPORT_SEAM_TARGET_GRID_NODES = 50000;
const EXPORT_SEAM_HARD_GRID_NODES = 90000;
const EXPORT_SEAM_MAX_BLOCK_SIZE = 256;
const PREVIEW_SEAM_PROGRESS_INTERVAL_MS = 1000;
const EXPORT_SEAM_PROGRESS_INTERVAL_MS = 1200;
const SEAM_INITIAL_STALL_TIMEOUT_MS = 30000;
const SEAM_MIN_STALL_TIMEOUT_MS = 15000;
const SEAM_MAX_STALL_TIMEOUT_MS = 180000;
const SEAM_STALL_GRACE_MULTIPLIER = 6;
const EXPORT_ENCODING_STATUS_INTERVAL_MS = 1000;

type ExportSeamTimeoutKind = 'stall';

class ExportSeamTimeoutError extends Error {
  kind: ExportSeamTimeoutKind;
  constructor(kind: ExportSeamTimeoutKind, message: string) {
    super(message);
    this.name = 'ExportSeamTimeoutError';
    this.kind = kind;
  }
}

interface ExportSeamPlan {
  blockSize: number;
  gridW: number;
  gridH: number;
  nodes: number;
  forceFeather: boolean;
}

function chooseExportSeamPlan(outW: number, outH: number, requestedBlockSize: number): ExportSeamPlan {
  const quantize = (v: number): number => Math.max(4, Math.round(v / 4) * 4);
  const evaluate = (blockSize: number) => {
    const gridW = Math.ceil(outW / blockSize);
    const gridH = Math.ceil(outH / blockSize);
    return { blockSize, gridW, gridH, nodes: gridW * gridH };
  };

  const requested = evaluate(Math.max(4, requestedBlockSize));
  let planned = requested;
  if (planned.nodes > EXPORT_SEAM_TARGET_GRID_NODES) {
    const scale = Math.sqrt(planned.nodes / EXPORT_SEAM_TARGET_GRID_NODES);
    const grown = Math.min(EXPORT_SEAM_MAX_BLOCK_SIZE, quantize(planned.blockSize * scale));
    planned = evaluate(grown);
  }
  if (planned.nodes > EXPORT_SEAM_HARD_GRID_NODES && planned.blockSize < EXPORT_SEAM_MAX_BLOCK_SIZE) {
    const scale = Math.sqrt(planned.nodes / EXPORT_SEAM_HARD_GRID_NODES);
    const grown = Math.min(EXPORT_SEAM_MAX_BLOCK_SIZE, quantize(planned.blockSize * scale));
    planned = evaluate(grown);
  }

  return {
    ...planned,
    forceFeather: planned.nodes > EXPORT_SEAM_HARD_GRID_NODES,
  };
}

function formatElapsedSeconds(ms: number): string {
  return `${Math.max(0, Math.round(ms / 1000))}s`;
}

type SeamWorkerClient = Pick<NonNullable<ReturnType<typeof getWorkerManager>>, 'onSeam' | 'sendSeam'>;

interface SeamSolveStatusSnapshot {
  elapsedMs: number;
  percent: number | null;
  remainingMs: number | null;
  info: string;
  staleMs: number;
  stallLimitMs: number;
}

function clampNumber(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function estimateRemainingFromPercent(percent: number | null, elapsedMs: number): number | null {
  if (percent === null || percent <= 0 || percent >= 100) return null;
  const remainingMs = Math.round((elapsedMs * (100 - percent)) / percent);
  return Number.isFinite(remainingMs) && remainingMs >= 0 ? remainingMs : null;
}

function formatSeamEstimate(percent: number | null, remainingMs: number | null): string {
  const pctText = percent === null ? 'estimating' : `${Math.round(percent)}%`;
  const remainingText = remainingMs === null ? 'estimating remaining' : `${formatElapsedSeconds(remainingMs)} remaining`;
  return `${pctText}, ${remainingText}`;
}

async function createScaledImageTexture(
  gl: WebGL2RenderingContext,
  file: File,
  width: number,
  height: number,
): Promise<ManagedTexture> {
  const bmp = await createImageBitmap(file);
  try {
    const offscreen = new OffscreenCanvas(width, height);
    const ctx2d = offscreen.getContext('2d')!;
    ctx2d.drawImage(bmp, 0, 0, width, height);
    return createTextureFromImage(gl, offscreen, width, height);
  } finally {
    bmp.close();
  }
}

async function waitForSeamSolveAdaptive(
  wm: SeamWorkerClient,
  opts: {
    jobId: string;
    gridW: number;
    gridH: number;
    dataCostsBuf: ArrayBuffer;
    edgeWeightsHBuf: ArrayBuffer;
    edgeWeightsVBuf: ArrayBuffer;
    hardBuf: ArrayBuffer;
    forceLegacy?: boolean;
    progressEveryMs: number;
    onStatus: (snapshot: SeamSolveStatusSnapshot) => void;
  },
): Promise<SeamResultMsg> {
  const {
    jobId,
    gridW,
    gridH,
    dataCostsBuf,
    edgeWeightsHBuf,
    edgeWeightsVBuf,
    hardBuf,
    forceLegacy,
    progressEveryMs,
    onStatus,
  } = opts;

  const startedAt = performance.now();
  let lastProgressAt = startedAt;
  let lastHeartbeatAt = startedAt;
  let heartbeatEwmaMs = progressEveryMs;
  let sawHeartbeat = false;
  let bestPercent = 0;
  let lastInfo = `grid ${gridW}×${gridH}`;
  let lastRemainingMs: number | null = null;
  let done = false;

  const makeSnapshot = (now: number): SeamSolveStatusSnapshot => {
    const elapsedMs = now - startedAt;
    const percent = bestPercent > 0 ? bestPercent : null;
    const remainingMs = lastRemainingMs ?? estimateRemainingFromPercent(percent, elapsedMs);
    const staleMs = now - lastProgressAt;
    const adaptiveStallLimitMs = sawHeartbeat
      ? clampNumber(
          Math.round(heartbeatEwmaMs * SEAM_STALL_GRACE_MULTIPLIER),
          SEAM_MIN_STALL_TIMEOUT_MS,
          SEAM_MAX_STALL_TIMEOUT_MS,
        )
      : SEAM_INITIAL_STALL_TIMEOUT_MS;
    return {
      elapsedMs,
      percent,
      remainingMs,
      info: lastInfo,
      staleMs,
      stallLimitMs: adaptiveStallLimitMs,
    };
  };

  return await new Promise<SeamResultMsg>((resolve, reject) => {
    let monitorTimer = 0;
    const cleanup = (unsub: () => void): void => {
      if (monitorTimer !== 0) window.clearInterval(monitorTimer);
      unsub();
    };

    const unsub = wm.onSeam((msg) => {
      if (done) return;
      if (msg.type === 'progress') {
        if (msg.jobId && msg.jobId !== jobId) return;
        if (!msg.stage.startsWith('seam-solve')) return;
        const now = performance.now();
        if (sawHeartbeat) {
          const intervalMs = Math.max(1, now - lastHeartbeatAt);
          heartbeatEwmaMs = heartbeatEwmaMs * 0.8 + intervalMs * 0.2;
        }
        sawHeartbeat = true;
        lastHeartbeatAt = now;
        lastProgressAt = now;
        lastInfo = msg.info ?? msg.stage;
        if (Number.isFinite(msg.percent)) {
          bestPercent = Math.max(bestPercent, clampNumber(msg.percent, 0, 100));
        }
        if (typeof msg.remainingMs === 'number' && Number.isFinite(msg.remainingMs) && msg.remainingMs >= 0) {
          lastRemainingMs = Math.round(msg.remainingMs);
        } else {
          lastRemainingMs = estimateRemainingFromPercent(bestPercent > 0 ? bestPercent : null, now - startedAt);
        }
        return;
      }
      if (msg.type === 'result') {
        if (msg.jobId !== jobId) return;
        done = true;
        cleanup(unsub);
        resolve(msg);
        return;
      }
      if (msg.type === 'error') {
        if (msg.jobId && msg.jobId !== jobId) return;
        done = true;
        cleanup(unsub);
        reject(new Error(msg.message || 'Seam worker error'));
      }
    });

    const tick = (): void => {
      if (done) return;
      const snapshot = makeSnapshot(performance.now());
      onStatus(snapshot);
      if (snapshot.staleMs > snapshot.stallLimitMs) {
        done = true;
        cleanup(unsub);
        reject(new Error(
          `Seam solve stalled (${formatElapsedSeconds(snapshot.staleMs)} without progress heartbeat; limit ${formatElapsedSeconds(snapshot.stallLimitMs)}).`,
        ));
      }
    };

    monitorTimer = window.setInterval(tick, 1000);
    onStatus(makeSnapshot(startedAt));

    try {
      wm.sendSeam({
        type: 'solve',
        jobId,
        gridW,
        gridH,
        dataCostsBuffer: dataCostsBuf,
        edgeWeightsHBuffer: edgeWeightsHBuf,
        edgeWeightsVBuffer: edgeWeightsVBuf,
        hardConstraintsBuffer: hardBuf,
        params: { progressEveryMs, forceLegacy: forceLegacy ? 1 : 0 },
      }, [dataCostsBuf, edgeWeightsHBuf, edgeWeightsVBuf, hardBuf]);
    } catch (err) {
      done = true;
      cleanup(unsub);
      reject(err instanceof Error ? err : new Error(String(err)));
    }
  });
}

/** Sanitize a gain value: replace NaN/Infinity/non-positive with 1.0. */
function safeGainVal(v: number): number {
  return Number.isFinite(v) && v > 0 ? v : 1.0;
}

function clampGainTowardUnity(value: number, mix = 1, minGain = 0.05, maxGain = 20): number {
  const safe = Math.max(minGain, Math.min(maxGain, safeGainVal(value)));
  const mixed = Math.exp(Math.log(safe) * mix);
  return Math.max(minGain, Math.min(maxGain, mixed));
}

function resolveAppliedPhotometricAdjustment(
  sameCameraSettings: boolean,
  gainObj: { gain?: number; gainR?: number; gainG?: number; gainB?: number } | undefined,
  vignetteParams: { a?: number; b?: number; c?: number } | undefined,
): {
  gain: [number, number, number];
  vignette: { a: number; b: number; c: number };
  toneMap: boolean;
} {
  const rawGain: [number, number, number] = gainObj
    ? [safeGainVal(gainObj.gainR ?? gainObj.gain ?? 1), safeGainVal(gainObj.gainG ?? gainObj.gain ?? 1), safeGainVal(gainObj.gainB ?? gainObj.gain ?? 1)]
    : [1, 1, 1];
  const rawVignette = {
    a: Number.isFinite(vignetteParams?.a) ? vignetteParams!.a! : 0,
    b: Number.isFinite(vignetteParams?.b) ? vignetteParams!.b! : 0,
    c: Number.isFinite(vignetteParams?.c) ? vignetteParams!.c! : 0,
  };

  if (!sameCameraSettings) {
    return {
      gain: rawGain,
      vignette: rawVignette,
      toneMap: rawGain.some((g) => g > 2.0 || g < 0.5),
    };
  }

  const avgLogGain = (Math.log(rawGain[0]) + Math.log(rawGain[1]) + Math.log(rawGain[2])) / 3;
  const scalarGain = clampGainTowardUnity(Math.exp(avgLogGain), 0.35, 0.97, 1.03);
  let scaledVignette = {
    a: rawVignette.a * 0.35,
    b: rawVignette.b * 0.35,
    c: rawVignette.c * 0.35,
  };
  const cornerBoost = 1 + scaledVignette.a * 2 + scaledVignette.b * 4 + scaledVignette.c * 8;
  const maxCornerBoost = 1.05;
  if (cornerBoost > maxCornerBoost) {
    const scale = (maxCornerBoost - 1) / Math.max(1e-6, cornerBoost - 1);
    scaledVignette = {
      a: scaledVignette.a * scale,
      b: scaledVignette.b * scale,
      c: scaledVignette.c * scale,
    };
  }
  return {
    gain: [scalarGain, scalarGain, scalarGain],
    vignette: scaledVignette,
    toneMap: false,
  };
}

function getResolvedSeamTier(): SeamAccelerationTier {
  return getState().capabilities?.seamAccelerationTier ?? 'legacyCpu';
}

function ensurePerfStore(): NonNullable<Window['__brenizerPerf']> {
  if (!window.__brenizerPerf) {
    window.__brenizerPerf = { seamJobs: [] };
  }
  return window.__brenizerPerf;
}

function recordSeamMetrics(metrics: SeamJobMetrics): void {
  ensurePerfStore().seamJobs.push(metrics);
  console.log(
    `[seam] ${metrics.stage}:${metrics.imageId} tier=${metrics.tier} builder=${metrics.builderBackend} ` +
    `solver=${metrics.solverBackend} grid=${metrics.gridW}x${metrics.gridH} block=${metrics.blockSize} ` +
    `readback=${metrics.readbackBytes}B summary=${metrics.summaryMs.toFixed(1)}ms graph=${metrics.graphBuildMs.toFixed(1)}ms ` +
    `solverMs=${metrics.solverMs.toFixed(1)}ms mask=${metrics.maskMs.toFixed(1)}ms total=${metrics.totalMs.toFixed(1)}ms`,
  );
}

function computeTriangleDoubleArea(
  ax: number, ay: number,
  bx: number, by: number,
  cx: number, cy: number,
): number {
  return Math.abs((bx - ax) * (cy - ay) - (cx - ax) * (by - ay));
}

function meshSanityReport(
  mesh: import('./gl').MeshData,
  outW: number,
  outH: number,
): { tooDistorted: boolean; offscreenFraction: number; extremeTriangleFraction: number } {
  const pos = mesh.positions;
  const idx = mesh.indices;
  if (!pos || pos.length < 6 || !idx || idx.length < 3) {
    return { tooDistorted: false, offscreenFraction: 0, extremeTriangleFraction: 0 };
  }

  const guardExtent = Math.max(outW, outH) * 0.75;
  let offscreenCount = 0;
  let vertexCount = 0;
  for (let i = 0; i < pos.length; i += 2) {
    const x = pos[i];
    const y = pos[i + 1];
    if (!Number.isFinite(x) || !Number.isFinite(y)
      || x < -guardExtent || x > outW + guardExtent
      || y < -guardExtent || y > outH + guardExtent) {
      offscreenCount++;
    }
    vertexCount++;
  }
  const offscreenFraction = vertexCount > 0 ? offscreenCount / vertexCount : 0;

  const areas: number[] = [];
  for (let i = 0; i < idx.length; i += 3) {
    const ia = idx[i] * 2;
    const ib = idx[i + 1] * 2;
    const ic = idx[i + 2] * 2;
    if (ic + 1 >= pos.length) continue;
    const area2 = computeTriangleDoubleArea(
      pos[ia], pos[ia + 1],
      pos[ib], pos[ib + 1],
      pos[ic], pos[ic + 1],
    );
    if (Number.isFinite(area2) && area2 > 1e-3) {
      areas.push(area2);
    }
  }
  if (areas.length < 6) {
    return { tooDistorted: offscreenFraction > 0.45, offscreenFraction, extremeTriangleFraction: 0 };
  }
  const sorted = [...areas].sort((a, b) => a - b);
  const med = sorted[Math.floor(sorted.length / 2)];
  const lo = med / 80;
  const hi = med * 80;
  let extreme = 0;
  for (const a of areas) {
    if (a < lo || a > hi) extreme++;
  }
  const extremeTriangleFraction = extreme / areas.length;
  const tooDistorted = offscreenFraction > 0.50 || extremeTriangleFraction > 0.60;
  return { tooDistorted, offscreenFraction, extremeTriangleFraction };
}

function clampMeshPositions(
  mesh: import('./gl').MeshData,
  outW: number,
  outH: number,
): void {
  const pos = mesh.positions;
  const guardExtent = Math.max(outW, outH) * 0.75;
  const minX = -guardExtent;
  const maxX = outW + guardExtent;
  const minY = -guardExtent;
  const maxY = outH + guardExtent;
  for (let i = 0; i < pos.length; i += 2) {
    let x = pos[i];
    let y = pos[i + 1];
    if (!Number.isFinite(x)) x = 0;
    if (!Number.isFinite(y)) y = 0;
    pos[i] = Math.max(minX, Math.min(maxX, x));
    pos[i + 1] = Math.max(minY, Math.min(maxY, y));
  }
}

function buildGlobalWarpMesh(
  alignW: number,
  alignH: number,
  gridN: number,
  T: Float64Array,
  minX: number,
  minY: number,
  compositeScale: number,
  outW: number,
  outH: number,
): import('./gl').MeshData {
  const baseMesh = createIdentityMesh(alignW, alignH, gridN, gridN);
  const warpedPositions = new Float32Array(baseMesh.positions.length);
  const affineDen = Math.abs(T[8]) > 1e-8 ? T[8] : 1;
  const guardExtent = Math.max(outW, outH) * 0.75;
  const clampXMin = -guardExtent;
  const clampXMax = outW + guardExtent;
  const clampYMin = -guardExtent;
  const clampYMax = outH + guardExtent;

  for (let i = 0; i < baseMesh.positions.length; i += 2) {
    const x = baseMesh.positions[i];
    const y = baseMesh.positions[i + 1];

    let gx: number;
    let gy: number;
    const denom = T[6] * x + T[7] * y + T[8];
    if (denom > 1e-4) {
      gx = (T[0] * x + T[1] * y + T[2]) / denom;
      gy = (T[3] * x + T[4] * y + T[5]) / denom;
    } else {
      // Perspective denominator became unstable; fall back to affine prediction.
      gx = (T[0] * x + T[1] * y + T[2]) / affineDen;
      gy = (T[3] * x + T[4] * y + T[5]) / affineDen;
    }

    let px = (gx - minX) * compositeScale;
    let py = (gy - minY) * compositeScale;
    if (!Number.isFinite(px)) px = 0;
    if (!Number.isFinite(py)) py = 0;
    warpedPositions[i] = Math.max(clampXMin, Math.min(clampXMax, px));
    warpedPositions[i + 1] = Math.max(clampYMin, Math.min(clampYMax, py));
  }

  baseMesh.positions = warpedPositions;
  return baseMesh;
}

function forceKeyImageFirst(order: string[], keyImageId: string | null): string[] {
  if (!keyImageId) return order;
  const idx = order.indexOf(keyImageId);
  if (idx <= 0) return order;
  return [keyImageId, ...order.slice(0, idx), ...order.slice(idx + 1)];
}

function enforceKeyForegroundMask(
  mask: Uint8Array,
  keyCoverageMask: Uint8Array | null,
  keyImageId: string | null,
  currentImageId: string,
  compPixels: Uint8Array,
  newPixels: Uint8Array,
): void {
  if (!keyCoverageMask || !keyImageId || currentImageId === keyImageId) return;
  const n = mask.length;
  for (let px = 0; px < n; px++) {
    if (keyCoverageMask[px] < 10) continue;
    const off = px * 4;
    if (compPixels[off + 3] > 10 && newPixels[off + 3] > 10) {
      mask[px] = 0;
    }
  }
}

function applyPreviewEditorTransform(): void {
  const transform = `translate(${editorPanX.toFixed(1)}px, ${editorPanY.toFixed(1)}px) scale(${editorZoom.toFixed(3)}) rotate(${editorRotationDeg.toFixed(2)}deg)`;
  const canvas = document.getElementById('preview-canvas') as HTMLCanvasElement | null;
  if (canvas) {
    canvas.style.transformOrigin = '50% 50%';
    canvas.style.transform = transform;
  }
  const overlay = document.getElementById('preview-overlay') as HTMLCanvasElement | null;
  if (overlay) {
    overlay.style.transformOrigin = '50% 50%';
    overlay.style.transform = transform;
  }
}

function setPanCursorState(isPanning: boolean): void {
  const container = document.getElementById('canvas-container');
  if (!container) return;
  container.classList.add('pan-enabled');
  container.classList.toggle('panning', isPanning);
}

function clearPreviewHoverOverlay(): void {
  const overlay = document.getElementById('preview-overlay') as HTMLCanvasElement | null;
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  overlay.style.display = 'none';
}

function drawPreviewHoverOverlay(): void {
  const overlay = document.getElementById('preview-overlay') as HTMLCanvasElement | null;
  if (!overlay) return;
  const ctx = overlay.getContext('2d');
  if (!ctx) return;
  ctx.clearRect(0, 0, overlay.width, overlay.height);

  if (!hoveredPreviewImageId || !previewDisplayRegion) {
    overlay.style.display = 'none';
    return;
  }
  const poly = previewImageOutlines.get(hoveredPreviewImageId);
  if (!poly || poly.points.length < 3) {
    overlay.style.display = 'none';
    return;
  }

  const { dispW, dispH } = previewDisplayRegion;
  const scale = Math.min(overlay.width / Math.max(1, dispW), overlay.height / Math.max(1, dispH));
  const offX = (overlay.width - dispW * scale) * 0.5;
  const offY = (overlay.height - dispH * scale) * 0.5;

  ctx.save();
  ctx.strokeStyle = 'rgba(255, 48, 48, 0.95)';
  ctx.fillStyle = 'rgba(255, 48, 48, 0.12)';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 5]);
  ctx.beginPath();
  for (let i = 0; i < poly.points.length; i++) {
    const [px, py] = poly.points[i];
    const sx = offX + px * scale;
    const sy = offY + py * scale;
    if (i === 0) ctx.moveTo(sx, sy);
    else ctx.lineTo(sx, sy);
  }
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  ctx.restore();
  overlay.style.display = 'block';
}

function setHoveredPreviewImage(imageId: string | null): void {
  hoveredPreviewImageId = imageId;
  drawPreviewHoverOverlay();
}

function refreshEditorControlLabels(): void {
  const zoomVal = document.getElementById('editor-zoom-val');
  const rotVal = document.getElementById('editor-rotate-val');
  if (zoomVal) zoomVal.textContent = `${Math.round(editorZoom * 100)}%`;
  if (rotVal) rotVal.textContent = `${editorRotationDeg.toFixed(1)}°`;
}

function rotateRgbaImage(src: Uint8ClampedArray, width: number, height: number, angleRad: number): Uint8ClampedArray {
  if (Math.abs(angleRad) < 1e-6) return src;
  const out = new Uint8ClampedArray(src.length);
  const cx = width / 2;
  const cy = height / 2;
  const cosA = Math.cos(-angleRad);
  const sinA = Math.sin(-angleRad);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const dx = x - cx;
      const dy = y - cy;
      const sx = cosA * dx - sinA * dy + cx;
      const sy = sinA * dx + cosA * dy + cy;
      const sx0 = Math.floor(sx);
      const sy0 = Math.floor(sy);
      const outIdx = (y * width + x) * 4;
      if (sx0 < 0 || sx0 >= width - 1 || sy0 < 0 || sy0 >= height - 1) {
        out[outIdx] = out[outIdx + 1] = out[outIdx + 2] = out[outIdx + 3] = 0;
        continue;
      }
      const fx = sx - sx0;
      const fy = sy - sy0;
      const i00 = (sy0 * width + sx0) * 4;
      const i10 = i00 + 4;
      const i01 = i00 + width * 4;
      const i11 = i01 + 4;
      for (let ch = 0; ch < 4; ch++) {
        const v = (1 - fx) * (1 - fy) * src[i00 + ch]
                + fx * (1 - fy) * src[i10 + ch]
                + (1 - fx) * fy * src[i01 + ch]
                + fx * fy * src[i11 + ch];
        out[outIdx + ch] = Math.round(v);
      }
    }
  }
  return out;
}

/**
 * Project a per-image saliency map into composite pixel coordinates.
 *
 * For each composite pixel, inverse-warp through T⁻¹ to image-space and
 * sample the saliency value. Uses sparse 4×4 sampling with block fill for
 * performance (saliency is inherently low-frequency, so nearest-neighbor
 * at 1/4 resolution is sufficient).
 *
 * The projected saliency map is used by computeBlockCosts() to penalise
 * seam placement through high-saliency regions (objects, faces, texture).
 *
 * Returns null if no saliency data is available for the given image.
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

function projectSaliencyToGrid(
  imgId: string,
  T: Float64Array,
  minX: number,
  minY: number,
  compositeScale: number,
  gridW: number,
  gridH: number,
  blockSize: number,
  saliencyMaps: Map<string, import('./pipelineController').SaliencyData>,
): Float32Array | null {
  const sal = saliencyMaps.get(imgId);
  if (!sal) return null;
  const sw = sal.width;
  const sh = sal.height;
  const sMap = sal.saliency;
  if (sMap.length < sw * sh) return null;

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

  const out = new Float32Array(gridW * gridH);
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      let sum = 0;
      let count = 0;
      for (let sy = 0; sy < 2; sy++) {
        for (let sx = 0; sx < 2; sx++) {
          const cx = gx * blockSize + ((sx + 0.5) * blockSize * 0.5);
          const cy = gy * blockSize + ((sy + 0.5) * blockSize * 0.5);
          const gxWorld = cx / compositeScale + minX;
          const gyWorld = cy / compositeScale + minY;
          const denom = Ti[6] * gxWorld + Ti[7] * gyWorld + Ti[8];
          if (Math.abs(denom) < 1e-8) continue;
          const ix = (Ti[0] * gxWorld + Ti[1] * gyWorld + Ti[2]) / denom;
          const iy = (Ti[3] * gxWorld + Ti[4] * gyWorld + Ti[5]) / denom;
          if (ix < 0 || ix >= sw - 1 || iy < 0 || iy >= sh - 1) continue;
          sum += sMap[Math.round(iy) * sw + Math.round(ix)] ?? 0;
          count++;
        }
      }
      out[gy * gridW + gx] = count > 0 ? sum / count : 0;
    }
  }
  return out;
}

function resolvePreviewSeamBlockSize(
  baseBlockSize: number,
  tier: SeamAccelerationTier,
  compW: number,
  compH: number,
): number {
  if (tier === 'desktopTurbo' || tier === 'legacyCpu') return baseBlockSize;
  const nodes = Math.ceil(compW / baseBlockSize) * Math.ceil(compH / baseBlockSize);
  if (nodes <= 22000) return baseBlockSize;
  return Math.max(4, Math.round((baseBlockSize + 4) / 4) * 4);
}

interface BuildAndSolveSeamOptions {
  stage: 'preview' | 'export';
  imageId: string;
  currentCompTex: WebGLTexture;
  newImageTex: WebGLTexture;
  width: number;
  height: number;
  blockSize: number;
  featherRadius: number;
  sameCameraSettings: boolean;
  faceRects: FaceRectComposite[];
  saliencyGrid: Float32Array | null;
  keyCoverageTex?: WebGLTexture | null;
  wm: SeamWorkerClient;
  progressEveryMs: number;
  forceLegacy?: boolean;
  webgpuSeam?: {
    builder: WebGPUSeamBuilder;
    compositeState: WebGPUSeamCompositeState;
    imageFile: File;
    sourceWidth: number;
    sourceHeight: number;
    mesh: MeshData;
    viewMatrix: Float32Array;
    gain: [number, number, number];
    vignette: { a: number; b: number; c: number };
    toneMap: boolean;
  } | null;
  onStatus: (snapshot: SeamSolveStatusSnapshot) => void;
}

interface ResolvedSeamMaskResult {
  maskTex: ManagedTexture;
  blendTex: WebGLTexture;
  correctedTex: ManagedTexture | null;
  graph: CompactSeamGraphBuildResult;
  solver: SeamResultMsg;
}

function splitLegacySeamEdgeWeights(
  edgeWeights: Float32Array,
  gridW: number,
  gridH: number,
): { edgeWeightsHBuf: ArrayBuffer; edgeWeightsVBuf: ArrayBuffer } {
  const nHorizontal = Math.max(0, (gridW - 1) * gridH);
  const edgeWeightsH = edgeWeights.slice(0, nHorizontal);
  const edgeWeightsV = edgeWeights.slice(nHorizontal);
  return {
    edgeWeightsHBuf: edgeWeightsH.buffer as ArrayBuffer,
    edgeWeightsVBuf: edgeWeightsV.buffer as ArrayBuffer,
  };
}

function buildGpuFeatherMask(
  imageId: string,
  width: number,
  height: number,
  featherRadius: number,
  currentCompTex: WebGLTexture,
  newImageTex: WebGLTexture,
  keyCoverageTex?: WebGLTexture | null,
): ManagedTexture {
  if (!seamAccelerator) throw new Error('Seam accelerator unavailable');
  const labels = new Uint8Array([255]);
  return seamAccelerator.buildMaskTexture({
    labels,
    gridW: 1,
    gridH: 1,
    width,
    height,
    blockSize: Math.max(width, height),
    featherRadius,
    ghostPenalty: new Float32Array([0]),
    ghostThreshold: 1,
    lightingSoftStart: 1,
    lightingSoftEnd: 1,
    compositeTex: currentCompTex,
    newTex: newImageTex,
    keyCoverageTex,
  });
}

function canUseDedicatedWebGPUSeamBuilder(tier: SeamAccelerationTier): boolean {
  return (tier === 'desktopTurbo' || tier === 'webgpu') && !!webgpuSeamBuilder;
}

function captureCompositeStateForDedicatedWebGPUSeam(
  sourceTex: WebGLTexture,
  width: number,
  height: number,
  blockSize: number,
  tier: SeamAccelerationTier,
): WebGPUSeamCompositeState | null {
  if (!seamAccelerator) return null;
  const summary = seamAccelerator.summarizeTexture({
    sourceTex,
    width,
    height,
    blockSize,
    tier,
  });
  if (!summary) return null;
  return {
    gridW: summary.gridW,
    gridH: summary.gridH,
    blockSize: summary.blockSize,
    sampleGrid: summary.sampleGrid,
    compMean: summary.mean,
    compSq: summary.sq,
  };
}

async function tryBuildWebGPUCompactGraph(
  opts: BuildAndSolveSeamOptions,
  tier: SeamAccelerationTier,
  colorTransfer?: Pick<NonNullable<CompactSeamGraphBuildResult['colorTransferStats']>, 'gain' | 'offset'> | null,
): Promise<CompactSeamGraphBuildResult | null> {
  if (!opts.webgpuSeam) return null;
  try {
    return await opts.webgpuSeam.builder.buildCompactGraph({
      imageId: opts.imageId,
      imageFile: opts.webgpuSeam.imageFile,
      sourceWidth: opts.webgpuSeam.sourceWidth,
      sourceHeight: opts.webgpuSeam.sourceHeight,
      width: opts.width,
      height: opts.height,
      blockSize: opts.blockSize,
      mesh: opts.webgpuSeam.mesh,
      viewMatrix: opts.webgpuSeam.viewMatrix,
      gain: opts.webgpuSeam.gain,
      vignette: opts.webgpuSeam.vignette,
      toneMap: opts.webgpuSeam.toneMap,
      faceRects: opts.faceRects,
      saliencyGrid: opts.saliencyGrid,
      tier,
      compositeState: opts.webgpuSeam.compositeState,
      colorTransfer: colorTransfer ?? null,
    });
  } catch (err) {
    console.warn(`[seam] WebGPU compact graph build failed for ${opts.stage}:${opts.imageId}; falling back to WebGL summaries.`, err);
    return null;
  }
}

async function buildAndSolveSeam(opts: BuildAndSolveSeamOptions): Promise<ResolvedSeamMaskResult> {
  if (!seamAccelerator) throw new Error('Seam accelerator unavailable');
  const tier = getResolvedSeamTier();
  const startedAt = performance.now();
  let graph = await tryBuildWebGPUCompactGraph(opts, tier);
  if (!graph) {
    graph = seamAccelerator.buildCompactGraph({
      compositeTex: opts.currentCompTex,
      newTex: opts.newImageTex,
      width: opts.width,
      height: opts.height,
      blockSize: opts.blockSize,
      faceRects: opts.faceRects,
      saliencyGrid: opts.saliencyGrid,
      tier,
    });
  }
  if (!graph) {
    throw new Error('Compact seam graph unavailable');
  }

  let correctedTex: ManagedTexture | null = null;
  let blendTex = opts.newImageTex;
  let totalSummaryMs = graph.summaryMs;
  let totalGraphMs = graph.buildMs;
  let totalReadbackBytes = graph.readbackBytes;

  if (!opts.sameCameraSettings && graph.colorTransferStats.apply) {
    correctedTex = seamAccelerator.applyColorTransfer(opts.newImageTex, opts.width, opts.height, graph.colorTransferStats);
    blendTex = correctedTex.texture;
    const correctedGraph = await tryBuildWebGPUCompactGraph(opts, tier, graph.colorTransferStats)
      ?? seamAccelerator.buildCompactGraph({
        compositeTex: opts.currentCompTex,
        newTex: blendTex,
        width: opts.width,
        height: opts.height,
        blockSize: opts.blockSize,
        faceRects: opts.faceRects,
        saliencyGrid: opts.saliencyGrid,
        tier,
      });
    if (correctedGraph) {
      totalSummaryMs += correctedGraph.summaryMs;
      totalGraphMs += correctedGraph.buildMs;
      totalReadbackBytes += correctedGraph.readbackBytes;
      graph = correctedGraph;
    }
  }

  const dataCostsBuf = graph.dataCosts.buffer.slice(0) as ArrayBuffer;
  const edgeWeightsHBuf = graph.edgeWeightsH.buffer.slice(0) as ArrayBuffer;
  const edgeWeightsVBuf = graph.edgeWeightsV.buffer.slice(0) as ArrayBuffer;
  const hardBuf = graph.hardConstraints.buffer.slice(0) as ArrayBuffer;

  const solver = await waitForSeamSolveAdaptive(opts.wm, {
    jobId: `${opts.stage}-${opts.imageId}`,
    gridW: graph.gridW,
    gridH: graph.gridH,
    dataCostsBuf,
    edgeWeightsHBuf,
    edgeWeightsVBuf,
    hardBuf,
    forceLegacy: opts.forceLegacy,
    progressEveryMs: opts.progressEveryMs,
    onStatus: opts.onStatus,
  });

  const maskStartedAt = performance.now();
  const labels = new Uint8Array(solver.labelsBuffer);
  const maskTex = seamAccelerator.buildMaskTexture({
    labels,
    gridW: graph.gridW,
    gridH: graph.gridH,
    width: opts.width,
    height: opts.height,
    blockSize: graph.resolvedBlockSize,
    featherRadius: opts.featherRadius,
    ghostPenalty: graph.ghostPenaltyBuffer,
    ghostThreshold: graph.ghostThreshold,
    lightingSoftStart: graph.lightingSoftStart,
    lightingSoftEnd: graph.lightingSoftEnd,
    compositeTex: opts.currentCompTex,
    newTex: blendTex,
    keyCoverageTex: opts.keyCoverageTex,
  });
  const maskMs = performance.now() - maskStartedAt;

  recordSeamMetrics({
    stage: opts.stage,
    imageId: opts.imageId,
    tier,
    builderBackend: graph.backendId,
    solverBackend: solver.backendId,
    gridW: graph.gridW,
    gridH: graph.gridH,
    blockSize: graph.resolvedBlockSize,
    readbackBytes: totalReadbackBytes,
    summaryMs: totalSummaryMs,
    graphBuildMs: totalGraphMs,
    solverMs: solver.solverMs,
    maskMs,
    totalMs: performance.now() - startedAt,
  });

  return {
    maskTex,
    blendTex,
    correctedTex,
    graph: {
      ...graph,
      summaryMs: totalSummaryMs,
      buildMs: totalGraphMs,
      readbackBytes: totalReadbackBytes,
    },
    solver,
  };
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
  const overlay = document.getElementById('preview-overlay') as HTMLCanvasElement | null;
  if (overlay) {
    overlay.width = canvas.width;
    overlay.height = canvas.height;
  }
  previewImageOutlines.clear();
  previewDisplayRegion = null;
  hoveredPreviewImageId = null;
  clearPreviewHoverOverlay();
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
  applyPreviewEditorTransform();
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

export async function boot(): Promise<void> {
  // Init UI first so elements are wired
  initUI();
  setRenderImagePreview(renderImagePreview);

  // Hovering image list items highlights their stitched footprint in preview.
  window.addEventListener('preview-image-hover', (ev: Event) => {
    const detail = (ev as CustomEvent<{ imageId?: string | null }>).detail;
    setHoveredPreviewImage(detail?.imageId ?? null);
  });

  // Detect capabilities
  setStatus('Detecting capabilities…');
  const turboModeEnabled = getTurboModePreference();
  const caps = await detectCapabilities();
  setState({ capabilities: caps, turboModeEnabled });
  renderCapabilities(caps);
  ensurePerfStore().seamJobs.length = 0;
  window.__brenizerRuntime = { caps, turboModeEnabled };

  // Resolve mode and apply preset
  const { userMode, mobileSafeFlag } = getState();
  const resolved = resolveMode(userMode, mobileSafeFlag, caps);
  const settings = getPreset(resolved);
  setState({ resolvedMode: resolved, settings });
  buildSettingsPanel();
  setStatus(`Ready — mode: ${resolved}`);

  // Init WebGL2 context
  try {
    const canvas = document.getElementById('preview-canvas') as HTMLCanvasElement;
    glCtx = createGLContext(canvas);
    warpRenderer = createWarpRenderer(glCtx.gl);
    kpRenderer = createKeypointRenderer(glCtx.gl);
    compositor = createCompositor(glCtx.gl);
    pyramidBlender = createPyramidBlender(glCtx.gl, glCtx.floatFBO);
    seamAccelerator = createSeamAccelerator(glCtx.gl, glCtx.floatFBO);
    webgpuSeamBuilder = caps.webgpuAvailable ? createWebGPUSeamBuilder() : null;
    console.log('WebGL2 context initialised, max tex:', glCtx.maxTextureSize);
    console.log('Seam acceleration tier:', caps.seamAccelerationTier);
    console.log('Cross-origin isolation mode:', caps.crossOriginIsolationMode);
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

  // Wire PTGui-style editor view controls
  const zoomSlider = document.getElementById('editor-zoom') as HTMLInputElement | null;
  const rotateSlider = document.getElementById('editor-rotate') as HTMLInputElement | null;
  const resetViewBtn = document.getElementById('editor-reset') as HTMLButtonElement | null;
  const previewCanvas = document.getElementById('preview-canvas') as HTMLCanvasElement | null;
  const applyEditorControlsFromUI = () => {
    if (zoomSlider) {
      editorZoom = clampNumber(Number(zoomSlider.value), 0.25, 3.0);
    }
    if (rotateSlider) {
      editorRotationDeg = clampNumber(Number(rotateSlider.value), -20, 20);
    }
    refreshEditorControlLabels();
    applyPreviewEditorTransform();
  };
  if (zoomSlider) {
    zoomSlider.value = editorZoom.toFixed(2);
    zoomSlider.addEventListener('input', applyEditorControlsFromUI);
  }
  if (rotateSlider) {
    rotateSlider.value = editorRotationDeg.toFixed(1);
    rotateSlider.addEventListener('input', applyEditorControlsFromUI);
  }
  if (resetViewBtn) {
    resetViewBtn.addEventListener('click', () => {
      editorZoom = 1.0;
      editorRotationDeg = 0.0;
      editorPanX = 0.0;
      editorPanY = 0.0;
      if (zoomSlider) zoomSlider.value = editorZoom.toFixed(2);
      if (rotateSlider) rotateSlider.value = editorRotationDeg.toFixed(1);
      refreshEditorControlLabels();
      applyPreviewEditorTransform();
    });
  }
  if (previewCanvas) {
    const endPan = (ev?: PointerEvent) => {
      if (ev && editorPanPointerId !== null && ev.pointerId !== editorPanPointerId) return;
      if (editorPanPointerId !== null && previewCanvas.hasPointerCapture(editorPanPointerId)) {
        previewCanvas.releasePointerCapture(editorPanPointerId);
      }
      editorPanPointerId = null;
      setPanCursorState(false);
    };
    previewCanvas.addEventListener('pointerdown', (ev) => {
      if (ev.button !== 0) return;
      editorPanPointerId = ev.pointerId;
      editorPanStartX = ev.clientX;
      editorPanStartY = ev.clientY;
      editorPanBaseX = editorPanX;
      editorPanBaseY = editorPanY;
      previewCanvas.setPointerCapture(ev.pointerId);
      setPanCursorState(true);
      ev.preventDefault();
    });
    previewCanvas.addEventListener('pointermove', (ev) => {
      if (editorPanPointerId === null || ev.pointerId !== editorPanPointerId) return;
      editorPanX = editorPanBaseX + (ev.clientX - editorPanStartX);
      editorPanY = editorPanBaseY + (ev.clientY - editorPanStartY);
      applyPreviewEditorTransform();
      ev.preventDefault();
    });
    previewCanvas.addEventListener('pointerup', endPan);
    previewCanvas.addEventListener('pointercancel', endPan);
    previewCanvas.addEventListener('lostpointercapture', () => endPan());
  }
  refreshEditorControlLabels();
  applyPreviewEditorTransform();

  // Wire Stitch Preview button
  document.getElementById('btn-stitch')!.addEventListener('click', () => {
    runStitchPreview().catch(err => {
      console.error('Pipeline error:', err);
      setStatus(`Pipeline error: ${err.message}`);
    });
  });

  // Wire first-pass optimization button
  document.getElementById('btn-optimize')!.addEventListener('click', async () => {
    const { settings } = getState();
    if (!settings) return;
    const sameCameraSettings = await promptOptimizeCameraSettingsChoice(!!settings.sameCameraSettings);
    const nextSettings = { ...settings, sameCameraSettings };
    setState({ settings: nextSettings });
    buildSettingsPanel();
    setStatus(
      sameCameraSettings
        ? 'Optimization configured for locked aperture / ISO / white balance.'
        : 'Optimization configured for mixed or varying camera settings.',
    );
    runFirstPassOptimization()
      .then(() => buildSettingsPanel())
      .catch(err => {
        console.error('Optimization error:', err);
        setStatus(`Optimization error: ${err.message}`);
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
    // Only override maxResExport for this export; restore afterwards
    const prevMaxRes = settings.maxResExport;
    setState({ settings: { ...settings, maxResExport: true } });
    exportComposite().catch(err => {
      console.error('Export error:', err);
      setStatus(`Export error: ${err.message}`);
    }).finally(() => {
      // Restore only maxResExport to avoid clobbering user changes during export
      const cur = getState().settings;
      if (cur) setState({ settings: { ...cur, maxResExport: prevMaxRes } });
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
    const activeNameById = new Map(active.map((img) => [img.id, img.name]));

    // Log match matrix to console for diagnostics
    console.group('Match Graph');
    for (const e of edges) {
      const nameI = activeNameById.get(e.i) ?? e.i;
      const nameJ = activeNameById.get(e.j) ?? e.j;
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
    if (active.length === 0 || !glCtx || !warpRenderer || transforms.size === 0) {
      setState({ pipelineStatus: 'idle' });
      return;
    }

    try {
      await renderWarpedPreview(active, transforms);
    } finally {
      // Mark pipeline idle now that compositing is done
      if (getState().pipelineStatus === 'running') {
        setState({ pipelineStatus: 'idle' });
      }
    }
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

  // Detect connected components via union-find.
  // Use adaptive threshold consistent with the pipeline's minInliers
  // (based on average alignment dimension), not a hardcoded value.
  // This ensures diagnostics match the actual connectivity used for stitching.
  const avgImgDim = images.length > 0
    ? images.reduce((s, img) => s + Math.max(img.width, img.height), 0) / images.length
    : 1024;
  const connectivityThreshold = Math.max(4, Math.min(15, Math.round(avgImgDim / 70)));
  const parent = Array.from({ length: n }, (_, i) => i);
  function find(x: number): number { return parent[x] === x ? x : (parent[x] = find(parent[x])); }
  for (const e of edges) {
    const a = idToIdx.get(e.i);
    const b = idToIdx.get(e.j);
    if (a !== undefined && b !== undefined && e.inlierCount >= connectivityThreshold) {
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
  const selectedKeyImageId = getState().keyImageId;
  const keyImageId = selectedKeyImageId && images.some((img) => img.id === selectedKeyImageId)
    ? selectedKeyImageId
    : null;
  const wm = getWorkerManager();
  const mstParent = getLastMstParent();

  // Size canvas to container
  const container = document.getElementById('canvas-container')!;
  canvas.width = container.clientWidth;
  canvas.height = container.clientHeight;
  canvas.style.display = 'block';
  const overlay = document.getElementById('preview-overlay') as HTMLCanvasElement | null;
  if (overlay) {
    overlay.width = canvas.width;
    overlay.height = canvas.height;
  }
  previewImageOutlines.clear();
  previewDisplayRegion = null;
  clearPreviewHoverOverlay();
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
  /** Track how many images were actually composited (for the status message). */
  let compositedImageCount = 0;

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
  // Then re-sort by confidence: greedily pick the next image that has the
  // best edge quality (highest inlier count / lowest RMS) to already-composited
  // images. This ensures high-confidence pairs blend first, producing a
  // better foundation for subsequent incremental additions.
  let mstOrder = (() => {
    const baseOrder: string[] = getLastMstOrder();
    const order = baseOrder.length > 0
      ? baseOrder.filter(id => connectedIds.has(id))
      : (() => {
          const ids = images.map(i => i.id).filter(id => connectedIds.has(id));
          if (refId && ids.includes(refId)) {
            return [refId, ...ids.filter(id => id !== refId)];
          }
          return ids;
        })();

    if (order.length <= 2) return order;

    // Build edge quality index: for each pair, store best inlier count and RMS
    const edgeQuality = new Map<string, { inlierCount: number; rms: number; objectScore: number; exifScore: number; lineScore: number }>();
    const allEdges = getLastEdges();
    for (const e of allEdges) {
      const key1 = `${e.i}|${e.j}`;
      const key2 = `${e.j}|${e.i}`;
        const q = {
          inlierCount: e.inlierCount,
          rms: e.rms,
          objectScore: e.objectScore ?? 0,
          exifScore: e.exifScore ?? 0,
          lineScore: e.lineScore ?? 0,
        };
      edgeQuality.set(key1, q);
      edgeQuality.set(key2, q);
    }

    // Greedy re-ordering: start with reference, then always pick the
    // remaining image with the highest "connection score" to already-placed images.
    // Connection score = max over placed neighbours of (inlierCount / (rms + 1)).
    const placed = new Set<string>();
    const result: string[] = [order[0]];
    placed.add(order[0]);
    const remaining = new Set(order.slice(1));

    while (remaining.size > 0) {
      let bestId = '';
      let bestScore = -Infinity;

      for (const cand of remaining) {
        let candScore = 0;
        for (const pid of placed) {
          const q = edgeQuality.get(`${pid}|${cand}`);
          if (q) {
            const score = (q.inlierCount / (q.rms + 1)) * (1 + q.objectScore * 0.35 + q.exifScore * 0.04);
            const lineBoost = 1 + q.lineScore * 0.20;
            candScore = Math.max(candScore, score * lineBoost);
          }
        }
        if (candScore > bestScore) {
          bestScore = candScore;
          bestId = cand;
        }
      }

      if (bestId) {
        result.push(bestId);
        placed.add(bestId);
        remaining.delete(bestId);
      } else {
        // No edge connection found — append remaining in original order
        for (const id of order) {
          if (remaining.has(id)) {
            result.push(id);
            remaining.delete(id);
          }
        }
        break;
      }
    }

    return result;
  })();
  mstOrder = forceKeyImageFirst(mstOrder, keyImageId);

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
  const rawImageOutlines = new Map<string, Array<[number, number]>>();

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

  const seamTier = getResolvedSeamTier();
  const useAcceleratedSeam = seamTier !== 'legacyCpu' && !!seamAccelerator && glCtx.floatFBO;
  const previewSeamBlockSize = resolvePreviewSeamBlockSize(blockSize, seamTier, compW, compH);
  let useDedicatedWebGPUSeamPrep = useGraphCut && useAcceleratedSeam && canUseDedicatedWebGPUSeamBuilder(seamTier);
  let previewWebgpuCompositeState: WebGPUSeamCompositeState | null = null;
  const _compPixels = useAcceleratedSeam ? null : new Uint8Array(compW * compH * 4);
  const _newPixels = useAcceleratedSeam ? null : new Uint8Array(compW * compH * 4);
  let keyCoverageMask: Uint8Array | null = null;
  let keyCoverageTex: ManagedTexture | null = null;
  const saliencyMaps = getLastSaliency();
  const imageById = new Map(images.map((img) => [img.id, img]));

  for (const imgId of mstOrder) {
    const img = imageById.get(imgId);
    if (!img) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const feat = features.get(imgId);
    const sf = feat?.scaleFactor ?? 1;
    const alignW = Math.round(img.width * sf);
    const alignH = Math.round(img.height * sf);
    const T = t.T;
    const vigParams = vignettes.get(imgId);
    const photometric = resolveAppliedPhotometricAdjustment(!!settings?.sameCameraSettings, gains.get(imgId), vigParams);
    const gain = photometric.gain;

    // Build warped mesh with sanity fallback to avoid extreme geometric distortion.
    let mesh: import('./gl').MeshData;
    let globalMesh: import('./gl').MeshData | null = null;
    const ensureGlobalMesh = (): import('./gl').MeshData => {
      if (!globalMesh) {
        globalMesh = buildGlobalWarpMesh(
          alignW, alignH, gridN, T, minX, minY, compositeScale, compW, compH,
        );
      }
      return globalMesh;
    };
    const apap = meshes.get(imgId);
    if (apap) {
      // Use APAP mesh (already in global coords), offset by -minX, -minY and scale.
      const warpedPos = new Float32Array(apap.vertices.length);
      for (let i = 0; i < apap.vertices.length; i += 2) {
        warpedPos[i] = (apap.vertices[i] - minX) * compositeScale;
        warpedPos[i + 1] = (apap.vertices[i + 1] - minY) * compositeScale;
      }
      const apapMesh: import('./gl').MeshData = {
        positions: warpedPos,
        uvs: apap.uvs,
        indices: apap.indices,
      };
      const apapSanity = meshSanityReport(apapMesh, compW, compH);
      if (apapSanity.tooDistorted) {
        console.warn(
          `APAP mesh sanity fallback for ${imgId}: offscreen=${(apapSanity.offscreenFraction * 100).toFixed(1)}% ` +
          `extreme=${(apapSanity.extremeTriangleFraction * 100).toFixed(1)}%`,
        );
        mesh = ensureGlobalMesh();
      } else {
        mesh = apapMesh;
      }
    } else {
      mesh = ensureGlobalMesh();
    }

    const meshSanity = meshSanityReport(mesh, compW, compH);
    if (meshSanity.tooDistorted) {
      clampMeshPositions(mesh, compW, compH);
      const clampedSanity = meshSanityReport(mesh, compW, compH);
      if (clampedSanity.tooDistorted) {
        console.warn(
          `Mesh remained distorted after clamping for ${imgId}: offscreen=${(clampedSanity.offscreenFraction * 100).toFixed(1)}% ` +
          `extreme=${(clampedSanity.extremeTriangleFraction * 100).toFixed(1)}%`,
        );
      }
    }

    // Track a simple footprint polygon (AABB in composite-space) for hover highlighting.
    let pMinX = Infinity, pMinY = Infinity, pMaxX = -Infinity, pMaxY = -Infinity;
    for (let i = 0; i < mesh.positions.length; i += 2) {
      const px = mesh.positions[i];
      const py = mesh.positions[i + 1];
      if (!Number.isFinite(px) || !Number.isFinite(py)) continue;
      pMinX = Math.min(pMinX, px);
      pMinY = Math.min(pMinY, py);
      pMaxX = Math.max(pMaxX, px);
      pMaxY = Math.max(pMaxY, py);
    }
    if (Number.isFinite(pMinX) && pMaxX > pMinX && pMaxY > pMinY) {
      rawImageOutlines.set(imgId, [
        [pMinX, pMinY],
        [pMaxX, pMinY],
        [pMaxX, pMaxY],
        [pMinX, pMaxY],
      ]);
    }

    // Decode image and create texture
    const imgTex = await createScaledImageTexture(gl, img.file, alignW, alignH);

    // Render new warped image to newImageFBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
    gl.viewport(0, 0, compW, compH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    // Pass vignetting correction coefficients and HDR tone mapping flag
    const vigA = photometric.vignette.a;
    const vigB = photometric.vignette.b;
    const vigC = photometric.vignette.c;
    const needsToneMap = photometric.toneMap;
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0, vigA, vigB, vigC, needsToneMap);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    imgTex.dispose();

    if (imgId === keyImageId && !keyCoverageMask && !keyCoverageTex) {
      if (useAcceleratedSeam && seamAccelerator) {
        keyCoverageTex = seamAccelerator.copyTexture(newImageTex.texture, compW, compH);
      } else if (_newPixels) {
        const keyPixels = _newPixels;
        gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
        gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, keyPixels);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        keyCoverageMask = new Uint8Array(compW * compH);
        for (let px = 0; px < compW * compH; px++) {
          keyCoverageMask[px] = keyPixels[px * 4 + 3];
        }
      }
    }

    if (imgIdx === 0) {
      // First image: copy directly to composite
      // Render new image to alt composite FBO
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, newImageFBO.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, currentCompFBO.fbo);
      gl.blitFramebuffer(0, 0, compW, compH, 0, 0, compW, compH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      if (useDedicatedWebGPUSeamPrep) {
        try {
          previewWebgpuCompositeState = captureCompositeStateForDedicatedWebGPUSeam(
            currentCompTex.texture,
            compW,
            compH,
            previewSeamBlockSize,
            seamTier,
          );
          if (!previewWebgpuCompositeState) {
            useDedicatedWebGPUSeamPrep = false;
          }
        } catch (err) {
          console.warn(`[seam] Failed to capture preview composite seam state for ${imgId}; falling back to WebGL summaries.`, err);
          useDedicatedWebGPUSeamPrep = false;
          previewWebgpuCompositeState = null;
        }
      }
    } else if (useGraphCut && useAcceleratedSeam) {
      const seamFaces: FaceRectComposite[] = [];
      for (const prevId of mstOrder.slice(0, imgIdx)) {
        const pf = facesInCompositeCoords.get(prevId);
        if (pf) {
          for (const f of pf) seamFaces.push({ ...f, imageLabel: 0 });
        }
      }
      const curFaces = facesInCompositeCoords.get(imgId);
      if (curFaces) {
        for (const f of curFaces) seamFaces.push({ ...f, imageLabel: 1 });
      }

      const saliencyGrid = settings?.blurAwareStitching
        ? projectSaliencyToGrid(imgId, T, minX, minY, compositeScale, Math.ceil(compW / previewSeamBlockSize), Math.ceil(compH / previewSeamBlockSize), previewSeamBlockSize, saliencyMaps)
        : null;
      const seamResult = await buildAndSolveSeam({
        stage: 'preview',
        imageId: imgId,
        currentCompTex: currentCompTex.texture,
        newImageTex: newImageTex.texture,
        width: compW,
        height: compH,
        blockSize: previewSeamBlockSize,
        featherRadius: Math.max(1, Math.round(featherWidth * compositeScale)),
        sameCameraSettings: !!settings?.sameCameraSettings,
        faceRects: seamFaces,
        saliencyGrid,
        keyCoverageTex: keyCoverageTex?.texture ?? null,
        wm: wm!,
        progressEveryMs: PREVIEW_SEAM_PROGRESS_INTERVAL_MS,
        webgpuSeam: useDedicatedWebGPUSeamPrep && webgpuSeamBuilder && previewWebgpuCompositeState
          ? {
            builder: webgpuSeamBuilder,
            compositeState: previewWebgpuCompositeState,
            imageFile: img.file,
            sourceWidth: alignW,
            sourceHeight: alignH,
            mesh,
            viewMatrix: compViewMat,
              gain,
              vignette: { a: vigA, b: vigB, c: vigC },
              toneMap: needsToneMap,
            }
          : null,
        onStatus: ({ percent, remainingMs, info }) => {
          setStatus(`Compositing ${imgIdx + 1}/${mstOrder.length} — seam ${formatSeamEstimate(percent, remainingMs)} (${info})`);
        },
      });

      if (useMultiband) {
        pyramidBlender!.blend(
          currentCompTex.texture,
          seamResult.blendTex,
          seamResult.maskTex.texture,
          altCompFBO.fbo,
          compW,
          compH,
          mbLevels,
        );
      } else {
        compositor.blendWithMask(
          currentCompTex.texture,
          seamResult.blendTex,
          seamResult.maskTex.texture,
          altCompFBO.fbo,
          compW,
          compH,
        );
      }
      seamResult.maskTex.dispose();
      seamResult.correctedTex?.dispose();

      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
      if (useDedicatedWebGPUSeamPrep) {
        if (seamResult.graph.backendId !== 'compact-webgpu-grid') {
          useDedicatedWebGPUSeamPrep = false;
          previewWebgpuCompositeState = null;
        } else {
          previewWebgpuCompositeState = captureCompositeStateForDedicatedWebGPUSeam(
            currentCompTex.texture,
            compW,
            compH,
            seamResult.graph.resolvedBlockSize,
            seamTier,
          );
          if (!previewWebgpuCompositeState) {
            console.warn(`[seam] Failed to refresh preview composite seam state after blending ${imgId}; falling back to WebGL summaries.`);
            useDedicatedWebGPUSeamPrep = false;
            previewWebgpuCompositeState = null;
          }
        }
      }
    } else if (useGraphCut) {
      // Seam finding via graph cut
      // Read back composite and new image at full resolution
      const compPixels = _compPixels!;
      const newPixels = _newPixels!;

      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // ── Reinhard-style color transfer in overlap region ──────────
      // Beyond linear gain, compute per-channel mean+std in the overlap
      // zone and apply an affine correction so the new image matches the
      // composite's color distribution.  This handles non-linear differences
      // (white balance shifts, vignetting residuals, etc.).
      // Even when the user indicates the images share camera settings, residual
      // overlap mismatch is still useful signal for white-balance drift and
      // imperfect vignette correction. Trust the measured overlap statistics
      // instead of suppressing this pass from metadata alone.
      // Collect overlap pixel statistics
      let nOverlap = 0;
      const sumC = [0, 0, 0]; // composite R, G, B sums
      const sumN = [0, 0, 0]; // new image R, G, B sums
      const sumC2 = [0, 0, 0];
      const sumN2 = [0, 0, 0];

      // Sample every 4th pixel for speed (the stats converge quickly)
      for (let px = 0; px < compW * compH; px += 4) {
        const off = px * 4;
        if (compPixels[off + 3] > 10 && newPixels[off + 3] > 10) {
          nOverlap++;
          for (let ch = 0; ch < 3; ch++) {
            const cv = compPixels[off + ch];
            const nv = newPixels[off + ch];
            sumC[ch] += cv;
            sumN[ch] += nv;
            sumC2[ch] += cv * cv;
            sumN2[ch] += nv * nv;
          }
        }
      }

      // Only apply if we have meaningful overlap (>100 sampled pixels)
      if (!settings?.sameCameraSettings && nOverlap > 100) {
        const gains = [0, 0, 0];
        const offsets = [0, 0, 0];
        let needsTransfer = false;

        for (let ch = 0; ch < 3; ch++) {
          const meanC = sumC[ch] / nOverlap;
          const meanN = sumN[ch] / nOverlap;
          const stdC = Math.sqrt(Math.max(0, sumC2[ch] / nOverlap - meanC * meanC));
          const stdN = Math.sqrt(Math.max(0, sumN2[ch] / nOverlap - meanN * meanN));

          if (stdN > 2) { // avoid div-by-zero for flat regions
            const g = stdC / stdN;
            // Clamp gain to reasonable range to avoid extreme corrections
            gains[ch] = Math.max(0.5, Math.min(2.0, g));
            offsets[ch] = meanC - gains[ch] * meanN;
            // Only flag if correction is non-trivial
            if (Math.abs(gains[ch] - 1.0) > 0.03 || Math.abs(offsets[ch]) > 3) {
              needsTransfer = true;
            }
          } else {
            gains[ch] = 1;
            offsets[ch] = 0;
          }
        }

        if (needsTransfer) {
          // Apply affine per-channel correction to new image pixels
          for (let px = 0; px < compW * compH; px++) {
            if (newPixels[px * 4 + 3] < 10) continue;
            for (let ch = 0; ch < 3; ch++) {
              const corrected = gains[ch] * newPixels[px * 4 + ch] + offsets[ch];
              newPixels[px * 4 + ch] = Math.max(0, Math.min(255, Math.round(corrected)));
            }
          }
          // Re-upload corrected pixels to the new image texture
          gl.bindTexture(gl.TEXTURE_2D, newImageTex!.texture);
          gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
          gl.bindTexture(gl.TEXTURE_2D, null);
        }
      }

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
        settings?.blurAwareStitching
          ? projectSaliencyToComposite(imgId, T, minX, minY, compositeScale, compW, compH, saliencyMaps)
          : null);

      // Send to seam worker
      const dataCostsBuf = costs.dataCosts.buffer.slice(0) as ArrayBuffer;
      const { edgeWeightsHBuf, edgeWeightsVBuf } = splitLegacySeamEdgeWeights(costs.edgeWeights, costs.gridW, costs.gridH);
      const hardBuf = costs.hardConstraints.buffer.slice(0) as ArrayBuffer;

      const jobId = `seam-${imgId}`;
      const seamResult = await waitForSeamSolveAdaptive(wm!, {
        jobId,
        gridW: costs.gridW,
        gridH: costs.gridH,
        dataCostsBuf,
        edgeWeightsHBuf,
        edgeWeightsVBuf,
        hardBuf,
        forceLegacy: true,
        progressEveryMs: PREVIEW_SEAM_PROGRESS_INTERVAL_MS,
        onStatus: ({ percent, remainingMs, info }) => {
          setStatus(`Compositing ${imgIdx + 1}/${mstOrder.length} — seam ${formatSeamEstimate(percent, remainingMs)} (${info})`);
        },
      });
      const blockLabels = new Uint8Array(seamResult.labelsBuffer);

      // Convert block labels to pixel mask + feather
      const pixelMask = labelsToMask(blockLabels, costs.gridW, costs.gridH, blockSize, compW, compH);

      // ── Adaptive feather width ─────────────────────────────────
      // Use the actual overlap extent to size the feather, rather than a
      // fixed pixel value.  This matches PTGui's behaviour: narrow overlaps
      // get narrow feathers (avoids bleeding) while wide overlaps get wide
      // feathers (smooth transitions).
      const baseFW = Math.max(1, Math.round(featherWidth * compositeScale));
      const maskResult = buildAdaptiveBlendMask(
        pixelMask,
        compPixels,
        newPixels,
        compW,
        compH,
        baseFW,
        { ghostBlockSize: blockSize * 2 },
      );
      const feathered = maskResult.mask;
      if (maskResult.ghostPixels > 0) {
        console.log(
          `[ghost] Forced ${maskResult.ghostPixels} pixels to binary mask ` +
          `(threshold=${maskResult.ghostThreshold.toFixed(1)}, median=${maskResult.ghostMedianDiff.toFixed(1)})`,
        );
      }

      enforceKeyForegroundMask(feathered, keyCoverageMask, keyImageId, imgId, compPixels, newPixels);

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
    } else if (useAcceleratedSeam) {
      const maskTex = buildGpuFeatherMask(
        imgId,
        compW,
        compH,
        Math.max(1, Math.round(featherWidth * compositeScale)),
        currentCompTex.texture,
        newImageTex.texture,
        keyCoverageTex?.texture ?? null,
      );

      if (useMultiband) {
        pyramidBlender!.blend(
          currentCompTex.texture,
          newImageTex.texture,
          maskTex.texture,
          altCompFBO.fbo,
          compW,
          compH,
          mbLevels,
        );
      } else {
        compositor.blendWithMask(
          currentCompTex.texture,
          newImageTex.texture,
          maskTex.texture,
          altCompFBO.fbo,
          compW,
          compH,
        );
      }
      maskTex.dispose();
      [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
      [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
    } else {
      // Feather-only fallback: simple alpha blend
      // Create a mask where new image has alpha
      const newPixels = _newPixels!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);

      // Also read composite for adaptive feather + alpha clamping
      const compPixels = _compPixels!;
      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, compPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      // Create mask from new image alpha
      const alphaMask = new Uint8Array(compW * compH);
      for (let i = 0; i < compW * compH; i++) {
        alphaMask[i] = newPixels[i * 4 + 3];
      }
      // Adaptive feather
      const baseFW = Math.max(1, Math.round(featherWidth * compositeScale));
      const feathered = buildAdaptiveBlendMask(
        alphaMask,
        compPixels,
        newPixels,
        compW,
        compH,
        baseFW,
        { ghostBlockSize: blockSize * 2 },
      ).mask;
      enforceKeyForegroundMask(feathered, keyCoverageMask, keyImageId, imgId, compPixels, newPixels);
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
    compositedImageCount = imgIdx;
    const compPct = Math.round((imgIdx / mstOrder.length) * 100);
    setStatus(`Compositing (${compPct}%) — ${imgIdx}/${mstOrder.length}`);
    updateProgress('compositing', imgIdx / mstOrder.length);
  }

  // ── Read back composite for auto-leveling + auto-crop ──
  const cropPixels = new Uint8Array(compW * compH * 4);
  gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
  gl.readPixels(0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, cropPixels);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);

  // ── Auto-leveling: correct panorama horizon tilt ─────────────
  // Estimate the composite's dominant tilt by computing the principal axis
  // of the content boundary.  We scan the top and bottom content edges
  // and fit a line via least-squares.  The angle of that line gives the
  // tilt to correct.  This mimics PTGui's "Optimize → Straighten Panorama".
  let levelAngle = 0; // radians, counter-clockwise correction
  {
    // Find top and bottom content edge scanlines
    const topEdge: [number, number][] = [];
    const botEdge: [number, number][] = [];
    // Sample every 8 columns for speed
    for (let x = 0; x < compW; x += 8) {
      let topY = -1, botY = -1;
      for (let y = 0; y < compH; y++) {
        if (cropPixels[(y * compW + x) * 4 + 3] > 10) {
          if (topY < 0) topY = y;
          botY = y;
        }
      }
      if (topY >= 0) {
        topEdge.push([x, topY]);
        botEdge.push([x, botY]);
      }
    }

    // Fit a line to the combined top+bottom edge points via least-squares
    // to estimate horizontal tilt.  We use both edges to be robust against
    // irregular content boundaries.
    const allEdge = [...topEdge, ...botEdge];
    if (allEdge.length >= 10) {
      let sx = 0, sy = 0, sxx = 0, sxy = 0;
      const n = allEdge.length;
      for (const [ex, ey] of allEdge) {
        sx += ex; sy += ey;
        sxx += ex * ex; sxy += ex * ey;
      }
      const denom = n * sxx - sx * sx;
      if (Math.abs(denom) > 1e-6) {
        const slope = (n * sxy - sx * sy) / denom;
        const angle = Math.atan(slope);
        // Only correct if tilt is small (< 15°) — large tilts are likely
        // intentional (e.g., vertical panorama) or indicate a bad fit
        if (Math.abs(angle) > 0.002 && Math.abs(angle) < Math.PI / 12) {
          levelAngle = -angle;
          console.log(`[auto-level] Detected tilt: ${(angle * 180 / Math.PI).toFixed(2)}°, correcting by ${(levelAngle * 180 / Math.PI).toFixed(2)}°`);
        }
      }
    }
  }

  // If leveling is needed, render a rotated version of the composite
  if (Math.abs(levelAngle) > 0.001) {
    // Create a rotation matrix in composite-space
    const cos = Math.cos(levelAngle);
    const sin = Math.sin(levelAngle);
    const cx = compW / 2;
    const cy = compH / 2;
    // Rotate around center: T_translate_back * T_rotate * T_translate_to_origin
    // Then map to clip space via the standard ortho projection
    // Combined: view = ortho * translate(cx,cy) * rotate * translate(-cx,-cy)
    const rotViewMat = new Float32Array([
      2 * cos / compW, -2 * sin / compH, 0,
      2 * sin / compW, 2 * cos / compH, 0,   // note: Y-flipped
      // Translation: rotate center then re-center
      (-2 * cos * cx - 2 * sin * cy) / compW + 1 + 2 * cx / compW - 1,
      (2 * sin * cx - 2 * cos * cy) / compH + 1 + 2 * cy / compH - 1,
      1,
    ]);
    // Simpler approach: use a fullscreen quad draw with the rotation matrix.
    // Actually — let's just rotate the crop readback pixels on CPU.
    // For small angles this is fast and avoids complex GPU matrix math.

    // Rotate the cropPixels in-place via bilinear sampling
    const rotated = new Uint8Array(compW * compH * 4);
    const cosA = Math.cos(-levelAngle); // inverse rotation for sampling
    const sinA = Math.sin(-levelAngle);
    for (let y = 0; y < compH; y++) {
      for (let x = 0; x < compW; x++) {
        // Map output (x,y) back through inverse rotation to source
        const dx = x - cx;
        const dy = y - cy;
        const sx = cosA * dx - sinA * dy + cx;
        const sy = sinA * dx + cosA * dy + cy;
        const sx0 = Math.floor(sx);
        const sy0 = Math.floor(sy);
        if (sx0 < 0 || sx0 >= compW - 1 || sy0 < 0 || sy0 >= compH - 1) {
          // Out of bounds → transparent
          const oi = (y * compW + x) * 4;
          rotated[oi] = rotated[oi + 1] = rotated[oi + 2] = rotated[oi + 3] = 0;
          continue;
        }
        // Bilinear interpolation
        const fx = sx - sx0;
        const fy = sy - sy0;
        const i00 = (sy0 * compW + sx0) * 4;
        const i10 = i00 + 4;
        const i01 = i00 + compW * 4;
        const i11 = i01 + 4;
        const oi = (y * compW + x) * 4;
        for (let ch = 0; ch < 4; ch++) {
          const v = (1 - fx) * (1 - fy) * cropPixels[i00 + ch]
                  + fx * (1 - fy) * cropPixels[i10 + ch]
                  + (1 - fx) * fy * cropPixels[i01 + ch]
                  + fx * fy * cropPixels[i11 + ch];
          rotated[oi + ch] = Math.round(v);
        }
      }
    }
    // Replace cropPixels with rotated version
    cropPixels.set(rotated);

    // Re-upload to composite texture for display
    gl.bindTexture(gl.TEXTURE_2D, currentCompTex.texture);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, compW, compH, gl.RGBA, gl.UNSIGNED_BYTE, cropPixels);
    gl.bindTexture(gl.TEXTURE_2D, null);
  }

  // ── Auto-crop: find content bounding box by scanning alpha ──
  // Read back alpha from the composite FBO to locate the actual image
  // content region, skipping the black (alpha=0) borders left by warping.
  // (cropPixels may have been updated by auto-leveling above)

  // ── Largest inscribed rectangle (content-aware optimal crop) ──
  // Instead of a simple bounding box (which includes transparent corners),
  // find the largest axis-aligned rectangle that contains only opaque pixels.
  // This is the "maximum rectangle under a histogram" algorithm applied to
  // each row of the alpha mask.  O(W×H) time.
  //
  // First, build a height-map: for each pixel, how many consecutive opaque
  // pixels are above (and including) it.
  const alphaThresh = 10;
  const heights = new Uint16Array(compW * compH);
  // First row
  for (let x = 0; x < compW; x++) {
    heights[x] = cropPixels[x * 4 + 3] > alphaThresh ? 1 : 0;
  }
  // Subsequent rows
  for (let y = 1; y < compH; y++) {
    for (let x = 0; x < compW; x++) {
      if (cropPixels[(y * compW + x) * 4 + 3] > alphaThresh) {
        heights[y * compW + x] = heights[(y - 1) * compW + x] + 1;
      } else {
        heights[y * compW + x] = 0;
      }
    }
  }

  // For each row, find the largest rectangle in the histogram using stack method
  let bestArea = 0;
  let bestCropL = 0, bestCropR = 0, bestCropB = 0, bestCropT = 0;
  const stack: number[] = [];

  for (let y = 0; y < compH; y++) {
    stack.length = 0;
    for (let x = 0; x <= compW; x++) {
      const h = x < compW ? heights[y * compW + x] : 0;
      while (stack.length > 0 && heights[y * compW + stack[stack.length - 1]] > h) {
        const topH = heights[y * compW + stack.pop()!];
        const width = stack.length === 0 ? x : x - stack[stack.length - 1] - 1;
        const area = topH * width;
        if (area > bestArea) {
          bestArea = area;
          bestCropR = x - 1;
          bestCropL = stack.length === 0 ? 0 : stack[stack.length - 1] + 1;
          bestCropT = y;
          bestCropB = y - topH + 1;
        }
      }
      stack.push(x);
    }
  }

  // Fallback: if inscribed rect is too small compared to bounding box,
  // fall back to simple bounding box (the panorama may be genuinely non-rectangular)
  let cropL: number, cropB: number, cropR: number, cropT: number;

  // Also compute simple bounding box for comparison
  let bbL = compW, bbB = compH, bbR = 0, bbT = 0;
  for (let y = 0; y < compH; y++) {
    for (let x = 0; x < compW; x++) {
      if (cropPixels[(y * compW + x) * 4 + 3] > alphaThresh) {
        bbL = Math.min(bbL, x); bbB = Math.min(bbB, y);
        bbR = Math.max(bbR, x); bbT = Math.max(bbT, y);
      }
    }
  }
  const bbArea = (bbR - bbL + 1) * (bbT - bbB + 1);

  // Use inscribed rectangle if it covers ≥60% of the bounding box area
  if (bestArea > 0 && bestArea >= bbArea * 0.6) {
    cropL = bestCropL;
    cropR = bestCropR;
    cropB = bestCropB;
    cropT = bestCropT;
    console.log(`[auto-crop] Inscribed rect: ${bestCropR - bestCropL + 1}×${bestCropT - bestCropB + 1} (${(bestArea / bbArea * 100).toFixed(1)}% of bbox)`);
  } else {
    // Bounding box fallback (with 1px margin)
    cropL = Math.max(0, bbL - 1);
    cropB = Math.max(0, bbB - 1);
    cropR = Math.min(compW - 1, bbR + 1);
    cropT = Math.min(compH - 1, bbT + 1);
    console.log(`[auto-crop] Bounding box fallback (inscribed too small: ${(bestArea / Math.max(1, bbArea) * 100).toFixed(1)}%)`);
  }

  const cropW = cropR - cropL + 1;
  const cropH = cropT - cropB + 1;
  const hasCrop = cropW > 0 && cropH > 0 && (cropW < compW || cropH < compH);

  // Use cropped dimensions for the view if auto-crop found a smaller region
  const dispW = hasCrop ? cropW : compW;
  const dispH = hasCrop ? cropH : compH;

  // Display final composite on screen
  const viewMat = makeViewMatrix(canvas.width, canvas.height, 0, 0, 1, dispW, dispH);
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);
  gl.clearColor(0.05, 0.05, 0.1, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  // Draw composite as a quad, adjusting UVs to show only the content region.
  // FBO textures have v=0 at bottom (GL convention), so flip V to display right-side up.
  const screenMesh = createIdentityMesh(dispW, dispH, 1, 1);
  if (hasCrop) {
    // Remap UVs from [0,1] to the cropped sub-region of the texture
    const uMin = cropL / compW;
    const uMax = (cropR + 1) / compW;
    const vMin = cropB / compH; // GL bottom
    const vMax = (cropT + 1) / compH; // GL top
    for (let i = 0; i < screenMesh.uvs.length; i += 2) {
      screenMesh.uvs[i] = uMin + screenMesh.uvs[i] * (uMax - uMin);
      // Flip V: vMax becomes top of screen (v=0 in screen coords)
      screenMesh.uvs[i + 1] = vMax - screenMesh.uvs[i + 1] * (vMax - vMin);
    }
  } else {
    for (let i = 1; i < screenMesh.uvs.length; i += 2) {
      screenMesh.uvs[i] = 1 - screenMesh.uvs[i];
    }
  }
  warpRenderer.drawMesh(currentCompTex.texture, screenMesh, viewMat, 1.0, 1.0);
  gl.disable(gl.BLEND);
  const uMin = hasCrop ? cropL / compW : 0;
  const uMax = hasCrop ? (cropR + 1) / compW : 1;
  const vMin = hasCrop ? cropB / compH : 0;
  const vMax = hasCrop ? (cropT + 1) / compH : 1;
  const uSpan = Math.max(1e-6, uMax - uMin);
  const vSpan = Math.max(1e-6, vMax - vMin);
  const cosA = Math.cos(levelAngle);
  const sinA = Math.sin(levelAngle);
  const centerX = compW * 0.5;
  const centerY = compH * 0.5;

  const displayOutlines = new Map<string, PreviewOutlinePolygon>();
  for (const [imageId, poly] of rawImageOutlines) {
    const mapped: Array<[number, number]> = [];
    for (const [x0, y0] of poly) {
      let x = x0;
      let y = y0;
      if (Math.abs(levelAngle) > 0.001) {
        const dx = x0 - centerX;
        const dy = y0 - centerY;
        x = cosA * dx - sinA * dy + centerX;
        y = sinA * dx + cosA * dy + centerY;
      }

      const u = x / compW;
      const v = 1 - (y / compH);
      const dispX = ((u - uMin) / uSpan) * dispW;
      const dispY = ((vMax - v) / vSpan) * dispH;
      mapped.push([dispX, dispY]);
    }
    if (mapped.length >= 3) {
      displayOutlines.set(imageId, { points: mapped });
    }
  }
  previewImageOutlines = displayOutlines;
  previewDisplayRegion = { dispW, dispH };
  drawPreviewHoverOverlay();
  applyPreviewEditorTransform();

  keyCoverageTex?.dispose();

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
  // Report the actual number of composited (connected) images — disconnected
  // images are not included in mstOrder and are skipped during compositing
  const skippedCount = images.length - compositedImageCount;
  const skipNote = skippedCount > 0 ? ` (${skippedCount} disconnected, skipped)` : '';
  setStatus(`Composite complete — ${compositedImageCount} images blended.${skipNote}`);

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
  const selectedKeyImageId = getState().keyImageId;
  const keyImageId = selectedKeyImageId && active.some((img) => img.id === selectedKeyImageId)
    ? selectedKeyImageId
    : null;

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

  let mstOrder = (() => {
    const order = getLastMstOrder();
    if (order.length > 0) return order.filter(id => connectedIds.has(id));
    const ids = active.map(i => i.id).filter(id => connectedIds.has(id));
    if (refId && ids.includes(refId)) return [refId, ...ids.filter(id => id !== refId)];
    return ids;
  })();
  mstOrder = forceKeyImageFirst(mstOrder, keyImageId);

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
  const seamTier = getResolvedSeamTier();
  const useAcceleratedSeam = seamTier !== 'legacyCpu' && !!seamAccelerator && glCtx.floatFBO;
  const exportSeamPlan = useGraphCut ? chooseExportSeamPlan(outW, outH, blockSize) : null;
  const useGraphCutForExport = useGraphCut && exportSeamPlan !== null && !exportSeamPlan.forceFeather;
  const useAcceleratedGraphCutForExport = useGraphCutForExport && useAcceleratedSeam;
  const exportGraphBlockSize = exportSeamPlan?.blockSize ?? blockSize;
  let useDedicatedWebGPUExportSeam = useAcceleratedGraphCutForExport && canUseDedicatedWebGPUSeamBuilder(seamTier);
  let exportWebgpuCompositeState: WebGPUSeamCompositeState | null = null;

  if (useGraphCut && exportSeamPlan) {
    if (exportSeamPlan.forceFeather) {
      setStatus(
        `Export seam graph ${exportSeamPlan.gridW}×${exportSeamPlan.gridH} is too large (${exportSeamPlan.nodes.toLocaleString()} nodes); using feather fallback.`,
      );
    } else if (exportSeamPlan.blockSize !== blockSize) {
      setStatus(
        `Export seam block size adjusted ${blockSize}px → ${exportSeamPlan.blockSize}px to keep graph size manageable.`,
      );
    }
  }

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
  const _expCompPixels = useGraphCutForExport && !useAcceleratedGraphCutForExport
    ? new Uint8Array(outW * outH * 4)
    : null;
  const _expNewPixels = useAcceleratedSeam ? null : new Uint8Array(outW * outH * 4);
  let keyCoverageMask: Uint8Array | null = null;
  let keyCoverageTex: ManagedTexture | null = null;
  const exportFeatherWidth = Math.max(1, Math.round(featherWidth * compositeScale));
  const exportGhostBlockSize = exportGraphBlockSize * 2;

  const blendWithMaskTexture = (
    maskTex: WebGLTexture,
    blendTex: WebGLTexture = newImageTex!.texture,
  ): void => {
    if (useMultiband) {
      pyramidBlender!.blend(currentCompTex.texture, blendTex, maskTex, altCompFBO.fbo, outW, outH, mbLevels);
    } else {
      compositor!.blendWithMask(currentCompTex.texture, blendTex, maskTex, altCompFBO.fbo, outW, outH);
    }
    [currentCompTex, altCompTex] = [altCompTex, currentCompTex];
    [currentCompFBO, altCompFBO] = [altCompFBO, currentCompFBO];
  };

  const blendWithMask = (
    maskPixels: Uint8Array,
    blendTex: WebGLTexture = newImageTex!.texture,
  ): void => {
    const maskTex = createMaskTexture(gl, maskPixels, outW, outH);
    try {
      blendWithMaskTexture(maskTex.texture, blendTex);
    } finally {
      maskTex.dispose();
    }
  };

  const blendGpuFeather = (blendTex: WebGLTexture = newImageTex!.texture): void => {
    if (!seamAccelerator) throw new Error('Seam accelerator unavailable');
    const maskTex = buildGpuFeatherMask(
      'export-feather',
      outW,
      outH,
      exportFeatherWidth,
      currentCompTex.texture,
      blendTex,
      keyCoverageTex?.texture ?? null,
    );
    try {
      blendWithMaskTexture(maskTex.texture, blendTex);
    } finally {
      maskTex.dispose();
    }
  };

  const blendFeatherFromNewPixels = (
    currentImageId: string,
    newPixels: Uint8Array,
    compPixels: Uint8Array | null,
  ): void => {
    const alphaMask = new Uint8Array(outW * outH);
    for (let i = 0; i < outW * outH; i++) alphaMask[i] = newPixels[i * 4 + 3];
    let comp = compPixels;
    if (!comp) {
      comp = new Uint8Array(outW * outH * 4);
      gl.bindFramebuffer(gl.FRAMEBUFFER, currentCompFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, comp);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    }
    const feathered = buildAdaptiveBlendMask(
      alphaMask,
      comp,
      newPixels,
      outW,
      outH,
      exportFeatherWidth,
      { ghostBlockSize: exportGhostBlockSize },
    ).mask;
    enforceKeyForegroundMask(feathered, keyCoverageMask, keyImageId, currentImageId, comp, newPixels);
    blendWithMask(feathered);
  };

  const solveSeamForExport = async (
    imgId: string,
    imgNumber: number,
    gridW: number,
    gridH: number,
    dataCostsBuf: ArrayBuffer,
    edgeWeightsHBuf: ArrayBuffer,
    edgeWeightsVBuf: ArrayBuffer,
    hardBuf: ArrayBuffer,
    forceLegacy = false,
  ): Promise<SeamResultMsg> => {
    if (!wm) throw new Error('Seam worker unavailable');
    const jobId = `export-${imgId}`;
    try {
      return await waitForSeamSolveAdaptive(wm, {
        jobId,
        gridW,
        gridH,
        dataCostsBuf,
        edgeWeightsHBuf,
        edgeWeightsVBuf,
        hardBuf,
        forceLegacy,
        progressEveryMs: EXPORT_SEAM_PROGRESS_INTERVAL_MS,
        onStatus: ({ percent, remainingMs, info }) => {
          setStatus(`Export: ${imgNumber}/${mstOrder.length} — seam ${formatSeamEstimate(percent, remainingMs)} (${info})`);
        },
      });
    } catch (err) {
      if (err instanceof Error && err.message.includes('stalled')) {
        throw new ExportSeamTimeoutError('stall', err.message);
      }
      throw err;
    }
  };

  const activeById = new Map(active.map((img) => [img.id, img]));
  for (const imgId of mstOrder) {
    const img = activeById.get(imgId);
    if (!img) continue;
    const t = transforms.get(imgId);
    if (!t) continue;
    const feat = features.get(imgId);
    const sf = feat?.scaleFactor ?? 1;
    const alignW = Math.round(img.width * sf);
    const alignH = Math.round(img.height * sf);
    const T = t.T;
    const expVigParams = getLastVignette().get(imgId);
    const exportPhotometric = resolveAppliedPhotometricAdjustment(!!settings.sameCameraSettings, gains.get(imgId), expVigParams);
    const gain = exportPhotometric.gain;

    // Build warped mesh with sanity fallback to avoid severe export distortion.
    let mesh: import('./gl').MeshData;
    let globalMesh: import('./gl').MeshData | null = null;
    const ensureGlobalMesh = (): import('./gl').MeshData => {
      if (!globalMesh) {
        globalMesh = buildGlobalWarpMesh(
          alignW, alignH, gridN, T, minX, minY, compositeScale, outW, outH,
        );
      }
      return globalMesh;
    };
    const apap = meshes.get(imgId);
    if (apap) {
      const warpedPos = new Float32Array(apap.vertices.length);
      for (let i = 0; i < apap.vertices.length; i += 2) {
        warpedPos[i] = (apap.vertices[i] - minX) * compositeScale;
        warpedPos[i + 1] = (apap.vertices[i + 1] - minY) * compositeScale;
      }
      const apapMesh: import('./gl').MeshData = {
        positions: warpedPos,
        uvs: apap.uvs,
        indices: apap.indices,
      };
      const apapSanity = meshSanityReport(apapMesh, outW, outH);
      if (apapSanity.tooDistorted) {
        console.warn(
          `Export APAP mesh sanity fallback for ${imgId}: offscreen=${(apapSanity.offscreenFraction * 100).toFixed(1)}% ` +
          `extreme=${(apapSanity.extremeTriangleFraction * 100).toFixed(1)}%`,
        );
        mesh = ensureGlobalMesh();
      } else {
        mesh = apapMesh;
      }
    } else {
      mesh = ensureGlobalMesh();
    }

    const meshSanity = meshSanityReport(mesh, outW, outH);
    if (meshSanity.tooDistorted) {
      clampMeshPositions(mesh, outW, outH);
      const clampedSanity = meshSanityReport(mesh, outW, outH);
      if (clampedSanity.tooDistorted) {
        console.warn(
          `Export mesh remained distorted after clamping for ${imgId}: offscreen=${(clampedSanity.offscreenFraction * 100).toFixed(1)}% ` +
          `extreme=${(clampedSanity.extremeTriangleFraction * 100).toFixed(1)}%`,
        );
      }
    }

    let texW: number, texH: number;
    if (settings.maxResExport) {
      // Full resolution: use original image dimensions (no downscale)
      texW = img.width;
      texH = img.height;
    } else {
      texW = alignW;
      texH = alignH;
    }
    const imgTex = await createScaledImageTexture(gl, img.file, texW, texH);

    // Warp to newImageFBO
    gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
    gl.viewport(0, 0, outW, outH);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.disable(gl.BLEND);
    // Vignetting correction + HDR tone mapping for export
    const expVigA = exportPhotometric.vignette.a;
    const expVigB = exportPhotometric.vignette.b;
    const expVigC = exportPhotometric.vignette.c;
    const expNeedsToneMap = exportPhotometric.toneMap;
    warpRenderer.drawMesh(imgTex.texture, mesh, compViewMat, gain, 1.0, expVigA, expVigB, expVigC, expNeedsToneMap);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    imgTex.dispose();

    if (imgId === keyImageId && !keyCoverageMask && !keyCoverageTex) {
      if (useAcceleratedSeam && seamAccelerator) {
        keyCoverageTex = seamAccelerator.copyTexture(newImageTex.texture, outW, outH);
      } else {
        const keyPixels = _expNewPixels!;
        gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
        gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, keyPixels);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);
        keyCoverageMask = new Uint8Array(outW * outH);
        for (let px = 0; px < outW * outH; px++) {
          keyCoverageMask[px] = keyPixels[px * 4 + 3];
        }
      }
    }

    if (imgIdx === 0) {
      // First image — just copy
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, newImageFBO.fbo);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, currentCompFBO.fbo);
      gl.blitFramebuffer(0, 0, outW, outH, 0, 0, outW, outH, gl.COLOR_BUFFER_BIT, gl.NEAREST);
      gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      if (useDedicatedWebGPUExportSeam) {
        try {
          exportWebgpuCompositeState = captureCompositeStateForDedicatedWebGPUSeam(
            currentCompTex.texture,
            outW,
            outH,
            exportGraphBlockSize,
            seamTier,
          );
          if (!exportWebgpuCompositeState) {
            useDedicatedWebGPUExportSeam = false;
          }
        } catch (err) {
          console.warn(`[seam] Failed to capture export composite seam state for ${imgId}; falling back to WebGL summaries.`, err);
          useDedicatedWebGPUExportSeam = false;
          exportWebgpuCompositeState = null;
        }
      }
    } else if (useGraphCutForExport && useAcceleratedGraphCutForExport) {
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

      const seamBlockSize = exportGraphBlockSize;
      const saliencyGrid = settings?.blurAwareStitching
        ? projectSaliencyToGrid(
          imgId,
          T,
          minX,
          minY,
          compositeScale,
          Math.ceil(outW / seamBlockSize),
          Math.ceil(outH / seamBlockSize),
          seamBlockSize,
          exportSaliencyMaps,
        )
        : null;

      try {
        const seamResult = await buildAndSolveSeam({
          stage: 'export',
          imageId: imgId,
          currentCompTex: currentCompTex.texture,
          newImageTex: newImageTex.texture,
          width: outW,
          height: outH,
          blockSize: seamBlockSize,
          featherRadius: exportFeatherWidth,
          sameCameraSettings: !!settings.sameCameraSettings,
          faceRects: seamFacesExport,
          saliencyGrid,
          keyCoverageTex: keyCoverageTex?.texture ?? null,
          wm,
          progressEveryMs: EXPORT_SEAM_PROGRESS_INTERVAL_MS,
          webgpuSeam: useDedicatedWebGPUExportSeam && webgpuSeamBuilder && exportWebgpuCompositeState
            ? {
              builder: webgpuSeamBuilder,
              compositeState: exportWebgpuCompositeState,
              imageFile: img.file,
              sourceWidth: texW,
              sourceHeight: texH,
              mesh,
              viewMatrix: compViewMat,
              gain,
              vignette: { a: expVigA, b: expVigB, c: expVigC },
              toneMap: expNeedsToneMap,
            }
            : null,
          onStatus: ({ percent, remainingMs, info }) => {
            setStatus(`Export: ${imgIdx + 1}/${mstOrder.length} — seam ${formatSeamEstimate(percent, remainingMs)} (${info})`);
          },
        });
        try {
          blendWithMaskTexture(seamResult.maskTex.texture, seamResult.blendTex);
        } finally {
          seamResult.maskTex.dispose();
          seamResult.correctedTex?.dispose();
        }
        if (useDedicatedWebGPUExportSeam) {
          if (seamResult.graph.backendId !== 'compact-webgpu-grid') {
            useDedicatedWebGPUExportSeam = false;
            exportWebgpuCompositeState = null;
          } else {
            exportWebgpuCompositeState = captureCompositeStateForDedicatedWebGPUSeam(
              currentCompTex.texture,
              outW,
              outH,
              seamResult.graph.resolvedBlockSize,
              seamTier,
            );
            if (!exportWebgpuCompositeState) {
              console.warn(`[seam] Failed to refresh export composite seam state after blending ${imgId}; falling back to WebGL summaries.`);
              useDedicatedWebGPUExportSeam = false;
              exportWebgpuCompositeState = null;
            }
          }
        }
      } catch (err) {
        const message = err instanceof Error ? err.message : String(err);
        if (message.includes('stalled')) {
          console.warn(`Export seam stalled for ${imgId}; falling back to GPU feather blend.`, err);
          setStatus(`Export: ${imgIdx + 1}/${mstOrder.length} — seam stalled, using feather fallback`);
        } else {
          console.warn(`Export seam solve failed for ${imgId}; falling back to GPU feather blend.`, err);
          setStatus(`Export: ${imgIdx + 1}/${mstOrder.length} — seam failed, using feather fallback`);
        }
        useDedicatedWebGPUExportSeam = false;
        exportWebgpuCompositeState = null;
        blendGpuFeather();
      }
    } else if (useGraphCutForExport) {
      // Graph cut seam finding with adaptive block sizing and stall monitoring.
      const compPixels = _expCompPixels!;
      const newPixels = _expNewPixels!;
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

      const seamBlockSize = exportSeamPlan?.blockSize ?? blockSize;
      const costs = computeBlockCosts(compPixels, newPixels, outW, outH, seamBlockSize, 0, seamFacesExport,
        settings?.blurAwareStitching
          ? projectSaliencyToComposite(imgId, T, minX, minY, compositeScale, outW, outH, exportSaliencyMaps)
          : null);
      const dataCostsBuf = costs.dataCosts.buffer.slice(0) as ArrayBuffer;
      const { edgeWeightsHBuf, edgeWeightsVBuf } = splitLegacySeamEdgeWeights(costs.edgeWeights, costs.gridW, costs.gridH);
      const hardBuf = costs.hardConstraints.buffer.slice(0) as ArrayBuffer;

      try {
        const seamResult = await solveSeamForExport(
          imgId,
          imgIdx + 1,
          costs.gridW,
          costs.gridH,
          dataCostsBuf,
          edgeWeightsHBuf,
          edgeWeightsVBuf,
          hardBuf,
          true,
        );
        const blockLabels = new Uint8Array(seamResult.labelsBuffer);
        const pixelMask = labelsToMask(blockLabels, costs.gridW, costs.gridH, seamBlockSize, outW, outH);
        const feathered = buildAdaptiveBlendMask(
          pixelMask,
          compPixels,
          newPixels,
          outW,
          outH,
          exportFeatherWidth,
          { ghostBlockSize: exportGhostBlockSize },
        ).mask;
        enforceKeyForegroundMask(feathered, keyCoverageMask, keyImageId, imgId, compPixels, newPixels);
        blendWithMask(feathered);
      } catch (err) {
        if (err instanceof ExportSeamTimeoutError) {
          console.warn(`Export seam stalled for ${imgId}; falling back to feather blend.`, err.message);
          setStatus(`Export: ${imgIdx + 1}/${mstOrder.length} — seam stalled, using feather fallback`);
        } else {
          console.warn(`Export seam solve failed for ${imgId}; falling back to feather blend.`, err);
          setStatus(`Export: ${imgIdx + 1}/${mstOrder.length} — seam failed, using feather fallback`);
        }
        blendFeatherFromNewPixels(imgId, newPixels, compPixels);
      }
    } else if (useAcceleratedSeam) {
      blendGpuFeather();
    } else {
      // Feather-only fallback — reuse pre-allocated buffer
      const newPixels = _expNewPixels!;
      newPixels.fill(0);
      gl.bindFramebuffer(gl.FRAMEBUFFER, newImageFBO.fbo);
      gl.readPixels(0, 0, outW, outH, gl.RGBA, gl.UNSIGNED_BYTE, newPixels);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      blendFeatherFromNewPixels(imgId, newPixels, null);
    }

    imgIdx++;
    setStatus(`Export: ${imgIdx}/${mstOrder.length}`);
  }

  keyCoverageTex?.dispose();

  // Read back final composite
  setStatus('Finalizing export pixels…');
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

  // Apply manual rotation from the panorama editor before export crop.
  if (Math.abs(editorRotationDeg) > 0.01) {
    const rotated = rotateRgbaImage(flipped, outW, outH, editorRotationDeg * Math.PI / 180);
    flipped.set(rotated);
    console.log(`[editor-rotate] Applied manual export rotation: ${editorRotationDeg.toFixed(2)}°`);
  }

  // Auto-crop: use largest inscribed rectangle (same as preview)
  const alphaThreshExport = 10;
  const expHeights = new Uint16Array(outW * outH);
  for (let x = 0; x < outW; x++) {
    expHeights[x] = flipped[x * 4 + 3] > alphaThreshExport ? 1 : 0;
  }
  for (let y = 1; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      expHeights[y * outW + x] = flipped[(y * outW + x) * 4 + 3] > alphaThreshExport
        ? expHeights[(y - 1) * outW + x] + 1 : 0;
    }
  }

  let expBestArea = 0;
  let expCropL = 0, expCropR = 0, expCropB = 0, expCropT = 0;
  const expStack: number[] = [];
  for (let y = 0; y < outH; y++) {
    expStack.length = 0;
    for (let x = 0; x <= outW; x++) {
      const h = x < outW ? expHeights[y * outW + x] : 0;
      while (expStack.length > 0 && expHeights[y * outW + expStack[expStack.length - 1]] > h) {
        const topH = expHeights[y * outW + expStack.pop()!];
        const width = expStack.length === 0 ? x : x - expStack[expStack.length - 1] - 1;
        const area = topH * width;
        if (area > expBestArea) {
          expBestArea = area;
          expCropR = x - 1;
          expCropL = expStack.length === 0 ? 0 : expStack[expStack.length - 1] + 1;
          expCropT = y;
          expCropB = y - topH + 1;
        }
      }
      expStack.push(x);
    }
  }

  // Bounding box fallback
  let cropMinX = outW, cropMinY = outH, cropMaxX = 0, cropMaxY = 0;
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      if (flipped[(y * outW + x) * 4 + 3] > alphaThreshExport) {
        cropMinX = Math.min(cropMinX, x);
        cropMinY = Math.min(cropMinY, y);
        cropMaxX = Math.max(cropMaxX, x);
        cropMaxY = Math.max(cropMaxY, y);
      }
    }
  }
  const expBBArea = Math.max(1, (cropMaxX - cropMinX + 1) * (cropMaxY - cropMinY + 1));
  if (expBestArea > 0 && expBestArea >= expBBArea * 0.6) {
    cropMinX = expCropL;
    cropMinY = expCropB;
    cropMaxX = expCropR;
    cropMaxY = expCropT;
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
  const encodeStartedAt = performance.now();
  setStatus(`Encoding ${exportFormat.toUpperCase()}… ${formatElapsedSeconds(0)}`);
  const encodingTicker = window.setInterval(() => {
    const elapsedMs = performance.now() - encodeStartedAt;
    setStatus(`Encoding ${exportFormat.toUpperCase()}… ${formatElapsedSeconds(elapsedMs)}`);
  }, EXPORT_ENCODING_STATUS_INTERVAL_MS);

  let blob: Blob;
  try {
    blob = await finalCanvas.convertToBlob({ type: mimeType, quality });
  } finally {
    window.clearInterval(encodingTicker);
  }

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
