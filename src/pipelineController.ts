/**
 * Pipeline controller — orchestrates the image stitching pipeline.
 *
 * Manages the CV worker lifecycle and runs pipeline stages in sequence:
 *  1. Image decoding & grayscale conversion
 *  2. ORB feature extraction (with CLAHE preprocessing)
 *  3. AI saliency map computation (gradient + colour + focus)
 *  4. Vignetting polynomial estimation (PTGui-style)
 *  5. Face detection (Chrome Shape Detection API)
 *  6. Pairwise matching (cross-checked kNN + MAGSAC++ RANSAC)
 *  7. MST construction & global transform propagation
 *  8. Levenberg-Marquardt bundle adjustment
 *  9. Per-channel RGB exposure compensation
 * 10. Depth inference (optional, via ONNX worker)
 * 11. APAP local mesh computation (Tikhonov-regularized weighted DLT)
 *
 * Each stage posts progress updates to the UI for real-time feedback.
 * The pipeline supports adaptive parameters (e.g. minInliers scales with
 * image size) to handle both full-frame photos and small grid tiles.
 */

import { createWorkerManager, type WorkerManager } from './workers/workerManager';
import type { CVFeaturesMsg, CVEdgesMsg, CVEdge, CVMSTMsg, CVTransformsMsg, CVMeshMsg, CVExposureMsg, CVSaliencyMsg, CVVignettingMsg, CVQualityAssessmentOutMsg } from './workers/workerTypes';
import type { DepthResultMsg } from './workers/workerTypes';
import { getState, setState, type ImageEntry } from './appState';
import type { PipelineSettings } from './presets';
import { setStatus, startProgress, endProgress, updateProgress, buildSettingsPanel } from './ui';

let workerManager: WorkerManager | null = null;

/** A detected face rectangle in alignment-scale image coordinates. */
export interface FaceRect {
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
}

/** Per-image feature data received from cv-worker. */
export interface ImageFeatures {
  imageId: string;
  keypoints: Float32Array;  // [x0,y0, x1,y1, ...]
  descriptors: Uint8Array;
  descCols: number;
  /** Scale factor used to resize image for alignment (image coords → alignment coords). */
  scaleFactor: number;
}

/** Feature results from the last run, keyed by imageId. */
let lastFeatures: Map<string, ImageFeatures> = new Map();

/** Edge results from the last matching run. */
export interface MatchEdge {
  i: string;
  j: string;
  H: Float64Array;
  inliers: Float32Array; // [xi, yi, xj, yj, ...]
  rms: number;
  inlierCount: number;
  isDuplicate: boolean;
}
let lastEdges: MatchEdge[] = [];

/** Return the feature descriptors from the last pipeline run. */
export function getLastFeatures(): Map<string, ImageFeatures> {
  return lastFeatures;
}

/** Return the pairwise match edges from the last pipeline run. */
export function getLastEdges(): MatchEdge[] {
  return lastEdges;
}

/** Global transforms from the last pipeline run. */
export interface GlobalTransform {
  imageId: string;
  T: Float64Array; // 3x3 row-major
}
let lastTransforms: Map<string, GlobalTransform> = new Map();
let lastRefId: string | null = null;
let lastMstOrder: string[] = [];
let lastMstParent: Record<string, string | null> = {};

/** Return the global homography transforms from the last pipeline run. */
export function getLastTransforms(): Map<string, GlobalTransform> {
  return lastTransforms;
}
/** Return the reference image ID chosen as the MST root. */
export function getLastRefId(): string | null {
  return lastRefId;
}
/** Return the MST traversal order (image IDs, root first). */
export function getLastMstOrder(): string[] {
  return lastMstOrder;
}
/** Return the MST parent mapping: imageId → parentId (null for root). */
export function getLastMstParent(): Record<string, string | null> {
  return lastMstParent;
}

/** Per-image depth map data. */
export interface DepthMap {
  imageId: string;
  depth: Uint16Array;
  width: number;
  height: number;
  nearIsOne: boolean;
}
let lastDepthMaps: Map<string, DepthMap> = new Map();

/** Return depth maps from the last pipeline run (empty if depth disabled). */
export function getLastDepthMaps(): Map<string, DepthMap> {
  return lastDepthMaps;
}

/** APAP mesh data per image. */
export interface APAPMesh {
  imageId: string;
  vertices: Float32Array;
  uvs: Float32Array;
  indices: Uint32Array;
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
}
let lastMeshes: Map<string, APAPMesh> = new Map();

/** Return APAP mesh data per image from the last pipeline run. */
export function getLastMeshes(): Map<string, APAPMesh> {
  return lastMeshes;
}

/** Per-image exposure gains — scalar fallback plus optional per-channel RGB. */
export interface ExposureGain {
  gain: number;
  gainR: number;
  gainG: number;
  gainB: number;
}
let lastGains: Map<string, ExposureGain> = new Map();

/** Return per-image exposure gains (scalar + RGB) from the last pipeline run. */
export function getLastGains(): Map<string, ExposureGain> {
  return lastGains;
}

/** Per-image saliency maps at alignment scale. */
export interface SaliencyData {
  imageId: string;
  saliency: Float32Array;
  width: number;
  height: number;
  blurScore: number;
}
let lastSaliency: Map<string, SaliencyData> = new Map();

/** Return per-image saliency maps at alignment scale. */
export function getLastSaliency(): Map<string, SaliencyData> {
  return lastSaliency;
}

/** Per-image vignetting polynomial coefficients. */
export interface VignetteParams {
  imageId: string;
  a: number;
  b: number;
}
let lastVignette: Map<string, VignetteParams> = new Map();

/** Return per-image vignetting polynomial coefficients. */
export function getLastVignette(): Map<string, VignetteParams> {
  return lastVignette;
}

/** Per-image face detection results. */
let lastFaces: Map<string, FaceRect[]> = new Map();

/** Return per-image face detection results (bounding boxes). */
export function getLastFaces(): Map<string, FaceRect[]> {
  return lastFaces;
}

/** Return the shared WorkerManager instance (null until first pipeline run). */
export function getWorkerManager(): WorkerManager | null {
  return workerManager;
}

/** Initialize all workers. Returns readiness status. */
export async function initWorkers(opts: { enableDepth?: boolean; enableSeam?: boolean; depthInitTimeoutMs?: number } = {}): Promise<{ cv: boolean; depth: boolean; seam: boolean }> {
  if (workerManager) {
    workerManager.dispose();
  }
  workerManager = createWorkerManager();
  setStatus('Initializing workers…');

  const result = await workerManager.initAll(opts);

  const parts: string[] = [];
  if (result.cv) parts.push('CV ✓');
  else parts.push('CV ✗');
  if (opts.enableDepth === false) parts.push('Depth off');
  else if (result.depth) parts.push('Depth ✓');
  else parts.push('Depth ✗');
  if (opts.enableSeam === false) parts.push('Seam off');
  else if (result.seam) parts.push('Seam ✓');
  else parts.push('Seam ✗');

  setStatus(`Workers: ${parts.join(' | ')}`);
  return result;
}

/**
 * Decode an image file to grayscale (and small RGB) at the alignment scale.
 * Returns { gray, rgbSmall, width, height, scaleFactor }.
 */
async function imageToGray(
  entry: ImageEntry,
  alignScale: number,
): Promise<{ gray: Uint8ClampedArray; rgbSmall: Uint8ClampedArray; width: number; height: number; scaleFactor: number }> {
  const bmp = await createImageBitmap(entry.file);
  const origW = bmp.width;
  const origH = bmp.height;

  // Compute scale so max(w,h) <= alignScale
  const maxDim = Math.max(origW, origH);
  const scaleFactor = maxDim > alignScale ? alignScale / maxDim : 1;
  const w = Math.round(origW * scaleFactor);
  const h = Math.round(origH * scaleFactor);

  // Draw onto offscreen canvas
  const offscreen = new OffscreenCanvas(w, h);
  const ctx = offscreen.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0, w, h);
  bmp.close();

  const imgData = ctx.getImageData(0, 0, w, h);
  const rgba = imgData.data;

  // Extract RGB (3-channel) for per-channel exposure compensation
  const rgbSmall = new Uint8ClampedArray(w * h * 3);
  // Convert to grayscale (luminance)
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    rgbSmall[i * 3] = r;
    rgbSmall[i * 3 + 1] = g;
    rgbSmall[i * 3 + 2] = b;
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  return { gray, rgbSmall, width: w, height: h, scaleFactor };
}

/**
 * Detect faces in an image at alignment scale using the Shape Detection API
 * (Chrome 70+). Falls back gracefully to empty array on unsupported browsers.
 */
async function detectFaces(
  entry: ImageEntry,
  scaleFactor: number,
): Promise<FaceRect[]> {
  try {
    // @ts-ignore — FaceDetector is a Chrome Shape Detection API not in TS lib
    if (typeof FaceDetector === 'undefined') return [];
    // @ts-ignore — same; TS lacks FaceDetector constructor typings
    const detector = new FaceDetector({ fastMode: true, maxDetectedFaces: 20 });
    const bmp = await createImageBitmap(entry.file);
    const origW = bmp.width;
    const origH = bmp.height;
    const w = Math.round(origW * scaleFactor);
    const h = Math.round(origH * scaleFactor);
    // Resize for faster detection
    const offscreen = new OffscreenCanvas(w, h);
    const ctx = offscreen.getContext('2d')!;
    ctx.drawImage(bmp, 0, 0, w, h);
    bmp.close();
    // Race detect against a timeout — FaceDetector can hang in headless/SW-rendered Chrome
    const timeoutMs = 10000;
    const faces = await Promise.race([
      detector.detect(offscreen),
      new Promise<never>((_, rej) => setTimeout(() => rej(new Error('FaceDetector timeout')), timeoutMs)),
    ]);
    return faces.map((f: any) => ({
      x: f.boundingBox.x,
      y: f.boundingBox.y,
      width: f.boundingBox.width,
      height: f.boundingBox.height,
      confidence: 1.0,
    }));
  } catch (e) {
    // Non-fatal: face detection is best-effort. Log for diagnostics.
    console.warn('Face detection failed for', entry.name, ':', e);
    return [];
  }
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function roundToStep(value: number, step: number): number {
  return Math.round(value / step) * step;
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  if (sorted.length % 2 === 0) return (sorted[mid - 1] + sorted[mid]) / 2;
  return sorted[mid];
}

function countConnectedComponents(ids: string[], edges: Array<{ i: string; j: string }>): number {
  if (ids.length === 0) return 0;

  const adj = new Map<string, Set<string>>();
  for (const id of ids) adj.set(id, new Set());
  for (const e of edges) {
    if (!adj.has(e.i) || !adj.has(e.j)) continue;
    adj.get(e.i)!.add(e.j);
    adj.get(e.j)!.add(e.i);
  }

  const visited = new Set<string>();
  let components = 0;
  for (const id of ids) {
    if (visited.has(id)) continue;
    components++;
    const stack = [id];
    visited.add(id);
    while (stack.length > 0) {
      const cur = stack.pop()!;
      for (const next of adj.get(cur) || []) {
        if (!visited.has(next)) {
          visited.add(next);
          stack.push(next);
        }
      }
    }
  }
  return components;
}

interface FeaturePassOptions {
  stage: string;
  statusPrefix: string;
  progressStart: number;
  progressEnd: number;
}

async function runFeaturePass(
  wm: WorkerManager,
  active: ImageEntry[],
  orbFeatures: number,
  options: FeaturePassOptions,
): Promise<Map<string, CVFeaturesMsg>> {
  const out = new Map<string, CVFeaturesMsg>();
  if (active.length === 0) return out;

  const span = Math.max(0, options.progressEnd - options.progressStart);
  updateProgress(options.stage, options.progressStart);

  const featureResolvers = new Map<string, {
    resolve: (m: CVFeaturesMsg) => void;
    reject: (e: Error) => void;
    timer: ReturnType<typeof setTimeout>;
  }>();
  const featurePromises = new Map<string, Promise<CVFeaturesMsg>>();
  for (const img of active) {
    featurePromises.set(
      img.id,
      new Promise<CVFeaturesMsg>((resolve, reject) => {
        const timer = setTimeout(() => {
          featureResolvers.delete(img.id);
          reject(new Error(`Timeout waiting for features of ${img.name}`));
        }, 60000);
        featureResolvers.set(img.id, { resolve, reject, timer });
      }),
    );
  }

  const featUnsub = wm.onCV((msg) => {
    if (msg.type === 'features' && featureResolvers.has(msg.imageId)) {
      const r = featureResolvers.get(msg.imageId)!;
      clearTimeout(r.timer);
      featureResolvers.delete(msg.imageId);
      r.resolve(msg as CVFeaturesMsg);
    } else if (msg.type === 'error') {
      for (const [, r] of featureResolvers) {
        clearTimeout(r.timer);
        r.reject(new Error((msg as any).message || 'Feature extraction failed'));
      }
      featureResolvers.clear();
    }
  });

  wm.sendCV({
    type: 'computeFeatures',
    orbParams: { nFeatures: orbFeatures },
  });

  try {
    let done = 0;
    for (const img of active) {
      const featMsg = await featurePromises.get(img.id)!;
      out.set(img.id, featMsg);
      done++;
      const numKp = featMsg.keypointsBuffer.byteLength / 8;
      setStatus(`${options.statusPrefix}: ${img.name} — ${numKp} keypoints (${done}/${active.length})`);
      updateProgress(options.stage, options.progressStart + span * (done / active.length));
    }
  } finally {
    featUnsub();
  }

  return out;
}

interface PrepareImagesOptions {
  stage: string;
  statusPrefix: string;
}

async function clearWorkerImages(wm: WorkerManager): Promise<void> {
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => { unsub(); reject(new Error('Timeout waiting for clearImages')); }, 15000);
    const unsub = wm.onCV((msg) => {
      if (msg.type === 'progress' && msg.stage === 'clearImages') {
        clearTimeout(timer);
        unsub();
        resolve();
      } else if (msg.type === 'error') {
        clearTimeout(timer);
        unsub();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    });
    wm.sendCV({ type: 'clearImages' });
  });
}

async function prepareImagesForCV(
  wm: WorkerManager,
  active: ImageEntry[],
  alignScale: number,
  options: PrepareImagesOptions,
): Promise<Record<string, number>> {
  const scaleFactors: Record<string, number> = {};
  updateProgress(options.stage, 0);
  for (let i = 0; i < active.length; i++) {
    const img = active[i];
    setStatus(`${options.statusPrefix} ${i + 1}/${active.length}: ${img.name}`);
    const { gray, rgbSmall, width, height, scaleFactor } = await imageToGray(img, alignScale);
    scaleFactors[img.id] = scaleFactor;
    const buf = gray.buffer as ArrayBuffer;
    const rgbBuf = rgbSmall.buffer as ArrayBuffer;
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => { unsub(); reject(new Error(`Timeout waiting for addImage ack: ${img.name}`)); }, 30000);
      const unsub = wm.onCV((msg) => {
        if (msg.type === 'progress' && msg.stage === 'addImage' && msg.info?.includes(img.id)) {
          clearTimeout(timer);
          unsub();
          resolve();
        } else if (msg.type === 'error') {
          clearTimeout(timer);
          unsub();
          reject(new Error((msg as any).message || 'Worker error'));
        }
      });
      wm.sendCV(
        {
          type: 'addImage',
          imageId: img.id,
          grayBuffer: buf,
          rgbSmallBuffer: rgbBuf,
          width,
          height,
        },
        [buf, rgbBuf],
      );
    });
    updateProgress(options.stage, (i + 1) / active.length);
  }
  return scaleFactors;
}

function countCandidatePairs(imageCount: number, windowW: number, matchAllPairs: boolean): number {
  if (imageCount <= 1) return 0;
  if (matchAllPairs) return (imageCount * (imageCount - 1)) / 2;
  let pairs = 0;
  for (let i = 0; i < imageCount; i++) {
    pairs += Math.max(0, Math.min(imageCount - i - 1, windowW));
  }
  return pairs;
}

interface MatchGraphPlan {
  windowW: number;
  matchAllPairs: boolean;
  candidatePairs: number;
  workloadScore: number;
  changed: boolean;
  summary: string;
}

function optimizeMatchGraphPlan(
  activeCount: number,
  baseWindowW: number,
  baseMatchAllPairs: boolean,
  avgKeypoints: number,
): MatchGraphPlan {
  const baseW = clamp(Math.round(baseWindowW), 2, 20);
  const basePairs = countCandidatePairs(activeCount, baseW, baseMatchAllPairs);
  const keypointFactor = clamp(avgKeypoints / 1400, 0.8, 4.0);
  const workloadScore = basePairs * keypointFactor;

  let windowW = baseW;
  let matchAllPairs = baseMatchAllPairs;

  // Runtime optimization pass for large/high-detail sets:
  // keep quality-oriented behavior for small sets, but limit pair explosion
  // when expected matching work is very high.
  if (activeCount > 12 && workloadScore > 220 && matchAllPairs) {
    matchAllPairs = false;
    windowW = Math.min(windowW, Math.max(4, Math.ceil(activeCount / 4)));
  }
  if (!matchAllPairs && activeCount >= 18 && workloadScore > 300) {
    windowW = Math.min(windowW, Math.max(4, Math.ceil(activeCount / 5)));
  }
  if (!matchAllPairs && activeCount >= 26 && workloadScore > 420) {
    windowW = Math.min(windowW, 4);
  }

  const candidatePairs = countCandidatePairs(activeCount, windowW, matchAllPairs);
  const changed = windowW !== baseW || matchAllPairs !== baseMatchAllPairs;
  const summary = changed
    ? `matchGraph runtime optimization: window ${baseW}→${windowW}, allPairs ${baseMatchAllPairs}→${matchAllPairs}, pairs ${basePairs}→${candidatePairs}`
    : `matchGraph runtime optimization kept defaults: window ${baseW}, allPairs ${baseMatchAllPairs}, pairs ${candidatePairs}`;

  return {
    windowW,
    matchAllPairs,
    candidatePairs,
    workloadScore,
    changed,
    summary,
  };
}

interface MatchGraphTimeoutPlan {
  absoluteTimeoutMs: number;
  initialStallTimeoutMs: number;
}

function estimateMatchGraphTimeoutPlan(
  candidatePairs: number,
  activeCount: number,
  avgKeypoints: number,
): MatchGraphTimeoutPlan {
  const keypointFactor = clamp(avgKeypoints / 1200, 0.8, 4.5);
  const estimatedAbsoluteMs = 60000 + candidatePairs * 900 * keypointFactor + activeCount * 1500;
  const absoluteTimeoutMs = clamp(Math.round(estimatedAbsoluteMs), 120000, 900000);
  const initialStallTimeoutMs = clamp(Math.round(45000 + keypointFactor * 15000), 45000, 240000);
  return { absoluteTimeoutMs, initialStallTimeoutMs };
}

interface WaitForMatchGraphEdgesOptions {
  wm: WorkerManager;
  candidatePairs: number;
  activeCount: number;
  avgKeypoints: number;
  timeoutLabel: string;
  onProgress?: (percent: number) => void;
}

async function waitForMatchGraphEdges(options: WaitForMatchGraphEdgesOptions): Promise<CVEdgesMsg> {
  const {
    wm,
    candidatePairs,
    activeCount,
    avgKeypoints,
    timeoutLabel,
    onProgress,
  } = options;
  const timePlan = estimateMatchGraphTimeoutPlan(candidatePairs, activeCount, avgKeypoints);
  const startedAt = performance.now();
  let lastProgressAt = startedAt;
  let heartbeatSeen = false;
  let heartbeatEwmaMs = 1000;
  let bestPercent = 0;

  return await new Promise<CVEdgesMsg>((resolve, reject) => {
    let monitorTimer = 0;
    let unsub: (() => void) | null = null;

    const cleanup = () => {
      if (monitorTimer) window.clearInterval(monitorTimer);
      if (unsub) unsub();
    };

    const rejectWith = (message: string): void => {
      cleanup();
      reject(new Error(message));
    };

    const tick = () => {
      const now = performance.now();
      const elapsedMs = now - startedAt;
      const staleMs = now - lastProgressAt;
      const stallLimitMs = heartbeatSeen
        ? clamp(
            Math.round(Math.max(timePlan.initialStallTimeoutMs, heartbeatEwmaMs * 8)),
            45000,
            300000,
          )
        : timePlan.initialStallTimeoutMs;

      if (elapsedMs > timePlan.absoluteTimeoutMs) {
        rejectWith(
          `Timeout waiting for ${timeoutLabel} edges (${Math.round(elapsedMs / 1000)}s elapsed, ${Math.round(bestPercent)}% complete, budget ${Math.round(timePlan.absoluteTimeoutMs / 1000)}s, ${candidatePairs} pairs).`,
        );
        return;
      }
      if (staleMs > stallLimitMs) {
        rejectWith(
          `Timeout waiting for ${timeoutLabel} edges: matching progress stalled for ${Math.round(staleMs / 1000)}s (limit ${Math.round(stallLimitMs / 1000)}s, ${Math.round(bestPercent)}% complete).`,
        );
      }
    };

    monitorTimer = window.setInterval(tick, 1000);

    unsub = wm.onCV((msg) => {
      if (msg.type === 'progress' && msg.stage === 'matching' && msg.percent !== undefined) {
        const now = performance.now();
        if (heartbeatSeen) {
          const intervalMs = Math.max(1, now - lastProgressAt);
          heartbeatEwmaMs = heartbeatEwmaMs * 0.8 + intervalMs * 0.2;
        } else {
          heartbeatSeen = true;
          heartbeatEwmaMs = 1000;
        }
        lastProgressAt = now;
        bestPercent = Math.max(bestPercent, clamp(msg.percent, 0, 100));
        if (onProgress) onProgress(bestPercent);
        return;
      }
      if (msg.type === 'edges') {
        cleanup();
        resolve(msg as CVEdgesMsg);
        return;
      }
      if (msg.type === 'error') {
        cleanup();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    });
  });
}

interface MatchingProbeMetrics {
  probeOrbFeatures: number;
  probeMinInliers: number;
  avgAlignedDim: number;
  avgKeypoints: number;
  minKeypoints: number;
  maxKeypoints: number;
  candidatePairCount: number;
  usableEdgeCount: number;
  usableEdgeDensity: number;
  componentCount: number;
  medianInliers: number;
  medianRms: number;
  duplicatePairCount: number;
}

interface MatchingProbeResult {
  tunedSettings: PipelineSettings;
  recommendedMinInliers: number;
  summary: string;
  metrics: MatchingProbeMetrics;
}

interface DepthSampleMetrics {
  imageId: string;
  stdNorm: number;
  gradientNorm: number;
}

interface DepthProbeResult {
  attempted: number;
  succeeded: number;
  avgStdNorm: number;
  avgGradientNorm: number;
}

interface DepthTuneResult {
  tunedSettings: PipelineSettings;
  summary: string;
  depthProbe: DepthProbeResult | null;
}

function pickProbeIndices(total: number, maxSamples: number): number[] {
  if (total <= 0 || maxSamples <= 0) return [];
  if (total <= maxSamples) {
    const all: number[] = [];
    for (let i = 0; i < total; i++) all.push(i);
    return all;
  }
  const out = new Set<number>();
  out.add(0);
  out.add(total - 1);
  while (out.size < maxSamples) {
    const t = out.size / (maxSamples - 1);
    const idx = Math.round(t * (total - 1));
    out.add(clamp(idx, 0, total - 1));
  }
  return Array.from(out).sort((a, b) => a - b);
}

function computeDepthProbeStats(depth: Uint16Array, depthW: number, depthH: number): { stdNorm: number; gradientNorm: number } {
  if (depth.length === 0 || depthW <= 1 || depthH <= 1) {
    return { stdNorm: 0, gradientNorm: 0 };
  }

  let mean = 0;
  let m2 = 0;
  let n = 0;
  for (let i = 0; i < depth.length; i++) {
    const x = depth[i] / 65535;
    n++;
    const delta = x - mean;
    mean += delta / n;
    m2 += delta * (x - mean);
  }
  const variance = n > 1 ? m2 / (n - 1) : 0;
  const stdNorm = Math.sqrt(Math.max(0, variance));

  let gradSum = 0;
  let gradCount = 0;
  for (let y = 0; y < depthH - 1; y++) {
    const row = y * depthW;
    const nextRow = (y + 1) * depthW;
    for (let x = 0; x < depthW - 1; x++) {
      const idx = row + x;
      const dx = Math.abs(depth[idx + 1] - depth[idx]) / 65535;
      const dy = Math.abs(depth[nextRow + x] - depth[idx]) / 65535;
      gradSum += dx + dy;
      gradCount += 2;
    }
  }
  const gradientNorm = gradCount > 0 ? gradSum / gradCount : 0;
  return { stdNorm, gradientNorm };
}

async function waitDepthResultForImage(
  wm: WorkerManager,
  imageId: string,
  timeoutMs: number,
): Promise<DepthResultMsg> {
  return await new Promise<DepthResultMsg>((resolve, reject) => {
    let unsub: (() => void) | null = null;
    const timer = setTimeout(() => {
      if (unsub) unsub();
      reject(new Error(`Timeout waiting for depth result: ${imageId}`));
    }, timeoutMs);
    const handler = (msg: import('./workers/workerTypes').DepthOutMsg) => {
      if (msg.type === 'result') {
        if (msg.imageId !== imageId) return;
        clearTimeout(timer);
        if (unsub) unsub();
        resolve(msg as DepthResultMsg);
      } else if (msg.type === 'error') {
        if (msg.imageId && msg.imageId !== imageId) return;
        clearTimeout(timer);
        if (unsub) unsub();
        reject(new Error(msg.message || `Depth worker error for ${imageId}`));
      }
    };
    unsub = wm.onDepth(handler);
  });
}

async function runDepthProbe(
  wm: WorkerManager,
  active: ImageEntry[],
  stage: string,
): Promise<DepthProbeResult> {
  const indices = pickProbeIndices(active.length, 3);
  if (indices.length === 0) {
    updateProgress(stage, 1);
    return {
      attempted: 0,
      succeeded: 0,
      avgStdNorm: 0,
      avgGradientNorm: 0,
    };
  }

  const probeInputSize = 128;
  const samples: DepthSampleMetrics[] = [];
  updateProgress(stage, 0);

  for (let i = 0; i < indices.length; i++) {
    const img = active[indices[i]];
    setStatus(`First pass: probing depth ${i + 1}/${indices.length} — ${img.name}`);
    try {
      const bmp = await createImageBitmap(img.file);
      const offscreen = new OffscreenCanvas(probeInputSize, probeInputSize);
      const ctx = offscreen.getContext('2d')!;
      ctx.drawImage(bmp, 0, 0, probeInputSize, probeInputSize);
      bmp.close();

      const imgData = ctx.getImageData(0, 0, probeInputSize, probeInputSize);
      const rgbaBuf = imgData.data.buffer as ArrayBuffer;
      wm.sendDepth(
        {
          type: 'infer',
          imageId: img.id,
          rgbaBuffer: rgbaBuf,
          width: probeInputSize,
          height: probeInputSize,
        },
        [rgbaBuf],
      );
      const result = await waitDepthResultForImage(wm, img.id, 20000);
      const depth = new Uint16Array(result.depthUint16Buffer);
      const stats = computeDepthProbeStats(depth, result.depthW, result.depthH);
      samples.push({
        imageId: img.id,
        stdNorm: stats.stdNorm,
        gradientNorm: stats.gradientNorm,
      });
    } catch (e) {
      console.warn(`Depth probe failed for ${img.name}:`, e);
    }
    updateProgress(stage, (i + 1) / indices.length);
  }

  const succeeded = samples.length;
  const avgStdNorm = succeeded > 0
    ? samples.reduce((s, m) => s + m.stdNorm, 0) / succeeded
    : 0;
  const avgGradientNorm = succeeded > 0
    ? samples.reduce((s, m) => s + m.gradientNorm, 0) / succeeded
    : 0;

  return {
    attempted: indices.length,
    succeeded,
    avgStdNorm,
    avgGradientNorm,
  };
}

function optimizeDepthSettings(
  tunedFromMatching: PipelineSettings,
  matchingMetrics: MatchingProbeMetrics,
  activeCount: number,
  depthWorkerReady: boolean,
  depthProbe: DepthProbeResult | null,
  isMobile: boolean,
): DepthTuneResult {
  const tuned: PipelineSettings = { ...tunedFromMatching };

  if (!depthWorkerReady) {
    const beforeEnabled = tuned.depthEnabled;
    const beforeInput = tuned.depthInputSize;
    tuned.depthEnabled = false;
    tuned.depthInputSize = clamp(roundToStep(Math.min(tuned.depthInputSize, 192), 64), 64, 512);
    const changes: string[] = [];
    if (beforeEnabled !== tuned.depthEnabled) changes.push(`Depth ${beforeEnabled ? 'on' : 'off'}→off`);
    if (beforeInput !== tuned.depthInputSize) changes.push(`DepthSize ${beforeInput}→${tuned.depthInputSize}`);
    const summary = changes.length > 0
      ? `Depth tuning applied: ${changes.join(', ')} (depth worker unavailable).`
      : 'Depth tuning kept current settings (depth worker unavailable).';
    return {
      tunedSettings: tuned,
      summary,
      depthProbe,
    };
  }

  // Depth currently affects APAP local mesh weighting only, so keep it off when meshing is off.
  if (tuned.meshGrid <= 0) {
    const changed = tuned.depthEnabled;
    tuned.depthEnabled = false;
    return {
      tunedSettings: tuned,
      summary: changed ? 'Depth disabled because APAP grid is off.' : 'Depth kept off (APAP grid disabled).',
      depthProbe,
    };
  }

  let signal = 0;
  if (matchingMetrics.medianRms >= 2.0) signal += 1;
  if (matchingMetrics.medianRms >= 2.8) signal += 1;
  if (matchingMetrics.componentCount === 1 && matchingMetrics.usableEdgeDensity >= 0.10) signal += 1;
  if (matchingMetrics.avgAlignedDim >= 900) signal += 1;
  if (matchingMetrics.avgKeypoints < matchingMetrics.probeOrbFeatures * 0.35) signal -= 1;
  if (activeCount > 22) signal -= 1;

  if (depthProbe && depthProbe.succeeded > 0) {
    if (depthProbe.avgGradientNorm >= 0.045) signal += 2;
    else if (depthProbe.avgGradientNorm >= 0.03) signal += 1;
    if (depthProbe.avgStdNorm >= 0.20) signal += 1;
  } else if (depthProbe) {
    signal -= 1;
  }

  const enableThreshold = tunedFromMatching.depthEnabled ? 2 : 3;
  const recommendedDepthEnabled = signal >= enableThreshold;
  tuned.depthEnabled = recommendedDepthEnabled;

  if (recommendedDepthEnabled) {
    let targetSize = 128;
    if (matchingMetrics.avgAlignedDim >= 1400 && activeCount <= 16 && signal >= 4) {
      targetSize = 256;
    } else if (matchingMetrics.avgAlignedDim >= 900 && activeCount <= 24) {
      targetSize = 192;
    }
    if (isMobile) targetSize = Math.min(targetSize, 192);
    tuned.depthInputSize = clamp(roundToStep(targetSize, 64), 64, 512);
  } else {
    // Keep disabled runs lighter; no need to hold a high input size.
    tuned.depthInputSize = clamp(roundToStep(Math.min(tuned.depthInputSize, 192), 64), 64, 512);
  }

  const changes: string[] = [];
  if (tunedFromMatching.depthEnabled !== tuned.depthEnabled) {
    changes.push(`Depth ${tunedFromMatching.depthEnabled ? 'on' : 'off'}→${tuned.depthEnabled ? 'on' : 'off'}`);
  }
  if (tunedFromMatching.depthInputSize !== tuned.depthInputSize) {
    changes.push(`DepthSize ${tunedFromMatching.depthInputSize}→${tuned.depthInputSize}`);
  }

  const probeText = depthProbe
    ? `depth probe ${depthProbe.succeeded}/${depthProbe.attempted}, grad ${depthProbe.avgGradientNorm.toFixed(3)}, std ${depthProbe.avgStdNorm.toFixed(3)}`
    : 'depth probe unavailable';
  const summary = changes.length > 0
    ? `Depth tuning applied: ${changes.join(', ')} (${probeText}, signal ${signal}).`
    : `Depth tuning kept current settings (${probeText}, signal ${signal}).`;

  return {
    tunedSettings: tuned,
    summary,
    depthProbe,
  };
}

function estimateMatchingSettings(
  baseSettings: PipelineSettings,
  activeCount: number,
  metrics: MatchingProbeMetrics,
): MatchingProbeResult {
  const tuned: PipelineSettings = { ...baseSettings };

  let orbScale = 1;
  if (metrics.avgKeypoints < metrics.probeOrbFeatures * 0.45) orbScale += 0.2;
  if (metrics.minKeypoints < metrics.probeOrbFeatures * 0.25) orbScale += 0.2;
  if (metrics.componentCount > 1) orbScale += 0.2;
  if (metrics.usableEdgeDensity < 0.12) orbScale += 0.2;
  if (metrics.componentCount === 1 && metrics.usableEdgeDensity > 0.35 && metrics.avgKeypoints > metrics.probeOrbFeatures * 0.75) orbScale -= 0.1;
  tuned.orbFeatures = clamp(roundToStep(baseSettings.orbFeatures * orbScale, 500), 500, 10000);

  let ratioDelta = 0;
  if (metrics.componentCount > 1 || metrics.medianInliers < 18) ratioDelta += 0.04;
  if (metrics.duplicatePairCount > 0) ratioDelta -= 0.03;
  if (metrics.usableEdgeDensity > 0.40 && metrics.medianInliers > 35) ratioDelta -= 0.02;
  tuned.ratioTest = clamp(Number((baseSettings.ratioTest + ratioDelta).toFixed(2)), 0.55, 0.90);

  let ransac = baseSettings.ransacThreshPx;
  if (metrics.componentCount > 1 || metrics.medianInliers < 12) ransac += 0.5;
  if (metrics.medianInliers > 50 && metrics.usableEdgeDensity > 0.25) ransac -= 0.5;
  tuned.ransacThreshPx = clamp(roundToStep(ransac, 0.5), 1, 10);

  let pairWindow = baseSettings.pairWindowW;
  if (metrics.componentCount > 1 && !baseSettings.matchAllPairs) {
    pairWindow = Math.max(pairWindow, Math.min(20, Math.ceil(activeCount / 2)));
  } else if (metrics.componentCount === 1 && metrics.usableEdgeDensity > 0.25 && activeCount > 16) {
    pairWindow = Math.max(3, Math.min(pairWindow, Math.ceil(activeCount / 4)));
  }
  tuned.pairWindowW = clamp(Math.round(pairWindow), 2, 20);

  let matchAllPairs = baseSettings.matchAllPairs;
  if (activeCount <= 12) {
    matchAllPairs = true;
  } else if (metrics.componentCount > 1 && activeCount <= 25) {
    matchAllPairs = true;
  } else if (metrics.componentCount === 1 && metrics.usableEdgeDensity > 0.22 && activeCount > 18) {
    matchAllPairs = false;
  }
  tuned.matchAllPairs = matchAllPairs;

  let refineIters = baseSettings.refineIters;
  if (metrics.componentCount > 1 || metrics.medianInliers < 20) refineIters += 10;
  if (metrics.medianInliers > 60 && metrics.usableEdgeDensity > 0.25) refineIters -= 5;
  tuned.refineIters = clamp(roundToStep(refineIters, 5), 0, 100);

  let recommendedMinInliers = Math.max(4, Math.min(15, Math.round(metrics.avgAlignedDim / 70)));
  if (metrics.componentCount > 1 || metrics.medianInliers < 16) recommendedMinInliers = Math.max(4, recommendedMinInliers - 2);
  if (metrics.usableEdgeDensity > 0.35 && metrics.medianInliers > 40) recommendedMinInliers = Math.min(18, recommendedMinInliers + 1);
  tuned.minInliers = clamp(Math.round(recommendedMinInliers), 0, 18);

  const complexityScore = activeCount * (metrics.avgAlignedDim / 1024);
  if (complexityScore >= 18 && metrics.componentCount === 1 && metrics.avgKeypoints > metrics.probeOrbFeatures * 0.60) {
    tuned.alignScale = clamp(roundToStep(baseSettings.alignScale - 128, 128), 512, 3072);
    if (baseSettings.meshGrid > 0) tuned.meshGrid = clamp(baseSettings.meshGrid - 2, 0, 24);
    if (baseSettings.multibandLevels > 0) tuned.multibandLevels = clamp(baseSettings.multibandLevels - 1, 2, 7);
    if (baseSettings.seamMethod === 'graphcut') tuned.seamBlockSize = clamp(baseSettings.seamBlockSize + 4, 4, 64);
  } else if (metrics.componentCount > 1 && metrics.avgKeypoints < metrics.probeOrbFeatures * 0.50) {
    tuned.alignScale = clamp(roundToStep(baseSettings.alignScale + 128, 128), 512, 3072);
    if (baseSettings.meshGrid > 0) tuned.meshGrid = clamp(baseSettings.meshGrid + 2, 0, 24);
  }

  const changed: string[] = [];
  const maybeAdd = (label: string, before: number | boolean, after: number | boolean, fmt: (v: number | boolean) => string = v => String(v)) => {
    if (before !== after) changed.push(`${label} ${fmt(before)}→${fmt(after)}`);
  };
  maybeAdd('ORB', baseSettings.orbFeatures, tuned.orbFeatures);
  maybeAdd('PairWin', baseSettings.pairWindowW, tuned.pairWindowW);
  maybeAdd('AllPairs', baseSettings.matchAllPairs, tuned.matchAllPairs);
  maybeAdd('Ratio', baseSettings.ratioTest, tuned.ratioTest, v => Number(v).toFixed(2));
  maybeAdd('RANSAC', baseSettings.ransacThreshPx, tuned.ransacThreshPx, v => Number(v).toFixed(1));
  maybeAdd('Refine', baseSettings.refineIters, tuned.refineIters);
  maybeAdd('MinInliers', baseSettings.minInliers, tuned.minInliers);
  maybeAdd('AlignScale', baseSettings.alignScale, tuned.alignScale);
  maybeAdd('MeshGrid', baseSettings.meshGrid, tuned.meshGrid);
  maybeAdd('Block', baseSettings.seamBlockSize, tuned.seamBlockSize);
  maybeAdd('Bands', baseSettings.multibandLevels, tuned.multibandLevels);

  const probeDetail = `probe edges ${metrics.usableEdgeCount}/${metrics.candidatePairCount}, components ${metrics.componentCount}, median inliers ${metrics.medianInliers.toFixed(0)}, median rms ${metrics.medianRms.toFixed(2)}, avg kp ${metrics.avgKeypoints.toFixed(0)}`;
  const summary = changed.length > 0
    ? `First pass tuning applied: ${changed.join(', ')} (${probeDetail}).`
    : `First pass tuning kept current matching settings (${probeDetail}).`;

  return {
    tunedSettings: tuned,
    recommendedMinInliers,
    summary,
    metrics,
  };
}

async function runMatchingProbe(
  wm: WorkerManager,
  active: ImageEntry[],
  baseSettings: PipelineSettings,
  scaleFactors: Record<string, number>,
): Promise<MatchingProbeResult> {
  const probeOrbFeatures = clamp(
    roundToStep(baseSettings.orbFeatures * 0.35, 500),
    500,
    Math.min(4000, baseSettings.orbFeatures),
  );

  setStatus(`First pass: extracting probe features (0/${active.length})…`);
  const probeFeatures = await runFeaturePass(
    wm,
    active,
    probeOrbFeatures,
    {
      stage: 'prepass',
      statusPrefix: 'First pass features',
      progressStart: 0,
      progressEnd: 0.45,
    },
  );

  const keypointCounts = active.map((img) => {
    const feat = probeFeatures.get(img.id);
    if (!feat) return 0;
    return feat.keypointsBuffer.byteLength / 8;
  });
  const avgKeypoints = keypointCounts.reduce((s, n) => s + n, 0) / Math.max(1, keypointCounts.length);
  const minKeypoints = keypointCounts.length > 0 ? Math.min(...keypointCounts) : 0;
  const maxKeypoints = keypointCounts.length > 0 ? Math.max(...keypointCounts) : 0;

  const avgAlignedDim = active.reduce((sum, img) => {
    const sf = scaleFactors[img.id] ?? 1;
    return sum + Math.max(img.width * sf, img.height * sf);
  }, 0) / active.length;
  const probeMinInliers = Math.max(4, Math.min(12, Math.round(avgAlignedDim / 85)));
  const probeMatchAllPairs = active.length <= 10 || baseSettings.matchAllPairs;
  const matchPlan = optimizeMatchGraphPlan(
    active.length,
    baseSettings.pairWindowW,
    probeMatchAllPairs,
    avgKeypoints,
  );
  const candidatePairCount = matchPlan.candidatePairs;
  console.info('[prepass-match-plan]', matchPlan.summary, {
    workloadScore: Number(matchPlan.workloadScore.toFixed(1)),
    avgKeypoints: Number(avgKeypoints.toFixed(0)),
  });

  setStatus('First pass: probing pair matches (0%)…');
  updateProgress('prepass', 0.45);
  const edgesPromise = waitForMatchGraphEdges({
    wm,
    candidatePairs: candidatePairCount,
    activeCount: active.length,
    avgKeypoints,
    timeoutLabel: 'first-pass matchGraph',
    onProgress: (pct) => {
      setStatus(`First pass: probing pair matches (${pct}%)…`);
      updateProgress('prepass', 0.45 + (pct / 100) * 0.45);
    },
  });

  wm.sendCV({
    type: 'matchGraph',
    windowW: matchPlan.windowW,
    ratio: baseSettings.ratioTest,
    ransacThreshPx: baseSettings.ransacThreshPx,
    minInliers: probeMinInliers,
    matchAllPairs: matchPlan.matchAllPairs,
  });

  const edgesMsg = await edgesPromise;
  updateProgress('prepass', 0.90);

  const usableEdges: CVEdge[] = [];
  let duplicatePairCount = edgesMsg.duplicatePairs?.length ?? 0;
  for (const e of edgesMsg.edges) {
    if (e.isDuplicate) {
      if (!edgesMsg.duplicatePairs) duplicatePairCount++;
      continue;
    }
    if (e.inlierCount >= probeMinInliers) usableEdges.push(e);
  }
  const usableEdgeDensity = candidatePairCount > 0 ? usableEdges.length / candidatePairCount : 0;
  const componentCount = countConnectedComponents(active.map((i) => i.id), usableEdges.map((e) => ({ i: e.i, j: e.j })));
  const medianInliers = median(usableEdges.map((e) => e.inlierCount));
  const medianRms = median(usableEdges.map((e) => e.rms));

  return estimateMatchingSettings(baseSettings, active.length, {
    probeOrbFeatures,
    probeMinInliers,
    avgAlignedDim,
    avgKeypoints,
    minKeypoints,
    maxKeypoints,
    candidatePairCount,
    usableEdgeCount: usableEdges.length,
    usableEdgeDensity,
    componentCount,
    medianInliers,
    medianRms,
    duplicatePairCount,
  });
}

/** Run only the first-pass analysis and apply optimized settings. */
export async function runFirstPassOptimization(): Promise<void> {
  const { images, settings, capabilities } = getState();
  const active = images.filter(i => !i.excluded);
  const wantDepthProbe = settings?.meshGrid !== undefined && settings.meshGrid > 0;

  if (active.length < 2) {
    setStatus('Need at least 2 images to optimize settings.');
    return;
  }
  if (!settings) {
    setStatus('Settings not loaded.');
    return;
  }
  if (getState().pipelineStatus === 'running') {
    setStatus('Pipeline already running.');
    return;
  }

  setState({ pipelineStatus: 'running' });
  setStatus('Starting first-pass optimization…');

  const prepassWeight = wantDepthProbe ? 40 : 50;
  const depthProbeWeight = wantDepthProbe ? 10 : 0;
  const stages = [
    { name: 'init', weight: 15 },
    { name: 'sendImages', weight: 35 },
    { name: 'prepass', weight: prepassWeight },
  ];
  if (depthProbeWeight > 0) {
    stages.push({ name: 'depthProbe', weight: depthProbeWeight });
  }
  startProgress(stages);

  try {
    updateProgress('init', 0);
    const ready = await initWorkers({
      enableDepth: wantDepthProbe,
      enableSeam: false,
      depthInitTimeoutMs: 15000,
    });
    updateProgress('init', 1);
    if (!ready.cv) throw new Error('CV worker failed to initialize. Cannot optimize settings.');

    setStatus('Preparing images for first pass…');
    await clearWorkerImages(workerManager!);
    const scaleFactors = await prepareImagesForCV(workerManager!, active, settings.alignScale, {
      stage: 'sendImages',
      statusPrefix: 'First pass: preparing image',
    });

    const probe = await runMatchingProbe(workerManager!, active, settings, scaleFactors);
    let tuned = probe.tunedSettings;
    let depthSummary = 'Depth tuning skipped.';
    let depthProbeMetrics: DepthProbeResult | null = null;

    if (wantDepthProbe) {
      if (ready.depth) {
        depthProbeMetrics = await runDepthProbe(workerManager!, active, 'depthProbe');
      } else {
        updateProgress('depthProbe', 1);
        depthProbeMetrics = null;
      }

      const depthResult = optimizeDepthSettings(
        tuned,
        probe.metrics,
        active.length,
        ready.depth,
        depthProbeMetrics,
        !!capabilities?.isMobile,
      );
      tuned = depthResult.tunedSettings;
      depthSummary = depthResult.summary;
    }

    setState({ settings: tuned });
    buildSettingsPanel();
    updateProgress('prepass', 1);
    setStatus(`Optimization complete. ${probe.summary} ${depthSummary}`);
    console.info('[prepass]', probe.summary, probe.metrics, depthSummary, depthProbeMetrics);
  } catch (err) {
    console.error('First-pass optimization error:', err);
    setStatus(`First-pass optimization error: ${err instanceof Error ? err.message : String(err)}`);
    setState({ pipelineStatus: 'error' });
  } finally {
    endProgress();
    if (getState().pipelineStatus === 'running') {
      setState({ pipelineStatus: 'idle' });
    }
  }
}

/** Run the full stitch preview pipeline. */
export async function runStitchPreview(): Promise<void> {
  const { images, settings } = getState();
  let active = images.filter(i => !i.excluded);

  if (active.length < 2) {
    setStatus('Need at least 2 images to stitch.');
    return;
  }

  if (!settings) {
    setStatus('Settings not loaded.');
    return;
  }
  let effectiveSettings: PipelineSettings = { ...settings };

  // Prevent re-entry while pipeline is already running
  if (getState().pipelineStatus === 'running') {
    setStatus('Pipeline already running.');
    return;
  }

  setState({ pipelineStatus: 'running' });
  setStatus('Starting pipeline…');

  // Set up progress tracking with weighted stages
  startProgress([
    { name: 'init', weight: 5 },
    { name: 'sendImages', weight: 2 },
    { name: 'features', weight: 15 },
    { name: 'saliency', weight: 5 },
    { name: 'vignetting', weight: 3 },
    { name: 'faces', weight: 3 },
    { name: 'matching', weight: 25 },
    { name: 'mst', weight: 2 },
    { name: 'refine', weight: 5 },
    { name: 'quality', weight: 2 },
    { name: 'exposure', weight: 3 },
    { name: 'apap', weight: 10 },
  ]);

  try {

  // Step 1: Init workers
  setStatus('Initializing OpenCV worker…');
  updateProgress('init', 0);
  const ready = await initWorkers({
    enableDepth: effectiveSettings.depthEnabled,
    enableSeam: effectiveSettings.seamMethod === 'graphcut',
  });
  updateProgress('init', 1);
  if (!ready.cv) {
    setStatus('CV worker failed to initialize. Cannot stitch.');
    setState({ pipelineStatus: 'error' });
    endProgress();
    return;
  }

  setStatus('Preparing images…');
  await clearWorkerImages(workerManager!);
  const scaleFactors = await prepareImagesForCV(workerManager!, active, effectiveSettings.alignScale, {
    stage: 'sendImages',
    statusPrefix: 'Preparing image',
  });

  // Step 3: Compute features (ORB) with tuned settings
  setStatus(`Extracting features (0/${active.length})…`);
  const featureMsgs = await runFeaturePass(
    workerManager!,
    active,
    effectiveSettings.orbFeatures,
    {
      stage: 'features',
      statusPrefix: 'Features',
      progressStart: 0,
      progressEnd: 1,
    },
  );

  // Build cached feature store for downstream stages and rendering
  lastFeatures = new Map();
  let totalKp = 0;
  for (const img of active) {
    const featMsg = featureMsgs.get(img.id);
    if (!featMsg) throw new Error(`Missing feature result for ${img.name}`);
    const keypoints = new Float32Array(featMsg.keypointsBuffer);
    lastFeatures.set(img.id, {
      imageId: img.id,
      keypoints,
      descriptors: new Uint8Array(featMsg.descriptorsBuffer),
      descCols: featMsg.descCols,
      scaleFactor: scaleFactors[img.id],
    });
    totalKp += keypoints.length / 2;
  }
  setStatus(`Feature extraction complete — ${totalKp} keypoints across ${active.length} images.`);

  // Dispatch custom event so main.ts can draw keypoint overlay
  window.dispatchEvent(new CustomEvent('features-ready'));

  // Step 3a: Saliency computation (AI object/texture/blur detection)
  lastSaliency = new Map();
  if (effectiveSettings.saliencyEnabled) {
    setStatus('Computing saliency maps…');
    updateProgress('saliency', 0);

    // Single handler for saliency results (same pattern as features)
    const salResolvers = new Map<string, {
      resolve: (m: CVSaliencyMsg) => void;
      reject: (e: Error) => void;
      timer: ReturnType<typeof setTimeout>;
    }>();
    const saliencyPromises = new Map<string, Promise<CVSaliencyMsg>>();
    for (const img of active) {
      saliencyPromises.set(
        img.id,
        new Promise<CVSaliencyMsg>((resolve, reject) => {
          const timer = setTimeout(() => {
            salResolvers.delete(img.id);
            reject(new Error(`Timeout waiting for saliency of ${img.name}`));
          }, 60000);
          salResolvers.set(img.id, { resolve, reject, timer });
        }),
      );
    }
    const salUnsub = workerManager!.onCV((msg) => {
      if (msg.type === 'saliency' && salResolvers.has((msg as CVSaliencyMsg).imageId)) {
        const m = msg as CVSaliencyMsg;
        const r = salResolvers.get(m.imageId)!;
        clearTimeout(r.timer);
        salResolvers.delete(m.imageId);
        r.resolve(m);
      } else if (msg.type === 'error') {
        for (const [, r] of salResolvers) {
          clearTimeout(r.timer);
          r.reject(new Error((msg as any).message || 'Saliency computation failed'));
        }
        salResolvers.clear();
      }
    });

    workerManager!.sendCV({ type: 'computeSaliency' });

    let salIdx = 0;
    for (const img of active) {
      try {
        const salMsg = await saliencyPromises.get(img.id)!;
        lastSaliency.set(img.id, {
          imageId: img.id,
          saliency: new Float32Array(salMsg.saliencyBuffer),
          width: salMsg.width,
          height: salMsg.height,
          blurScore: salMsg.blurScore,
        });
      } catch {
        // Non-fatal: continue without saliency for this image
      }
      salIdx++;
      updateProgress('saliency', salIdx / active.length);
    }
    setStatus(`Saliency maps computed for ${lastSaliency.size}/${active.length} images.`);
    salUnsub();
  }
  updateProgress('saliency', 1);

  // Step 3b-pre: Vignetting estimation (PTGui-style polynomial radial model)
  lastVignette = new Map();
  if (effectiveSettings.vignetteCorrection) {
    setStatus('Estimating vignetting…');
    updateProgress('vignetting', 0);

    const vignetteUnsub = workerManager!.onCV((msg) => {
      if (msg.type === 'vignetting') {
        const vmsg = msg as CVVignettingMsg;
        lastVignette.set(vmsg.imageId, {
          imageId: vmsg.imageId,
          a: vmsg.vignetteParams.a,
          b: vmsg.vignetteParams.b,
        });
      }
    });

    workerManager!.sendCV({ type: 'computeVignetting', pooled: effectiveSettings.sameCameraSettings });

    // Wait for vignetting progress complete
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => resolve(), 30000);
      const unsub = workerManager!.onCV((msg) => {
        if (msg.type === 'progress' && msg.stage === 'vignetting' && msg.percent >= 100) {
          clearTimeout(timer);
          unsub();
          resolve();
        }
      });
    });
    vignetteUnsub();
    setStatus(`Vignetting estimated for ${lastVignette.size} images.`);
  }
  updateProgress('vignetting', 1);

  // Step 3b: Face detection (browser Shape Detection API)
  lastFaces = new Map();
  setStatus('Detecting faces…');
  updateProgress('faces', 0);
  for (let i = 0; i < active.length; i++) {
    const img = active[i];
    const sf = scaleFactors[img.id];
    const faces = await detectFaces(img, sf);
    if (faces.length > 0) {
      lastFaces.set(img.id, faces);
    }
    updateProgress('faces', (i + 1) / active.length);
  }
  const totalFaces = Array.from(lastFaces.values()).reduce((s, f) => s + f.length, 0);
  if (totalFaces > 0) {
    setStatus(`Detected ${totalFaces} face(s) across ${lastFaces.size} image(s).`);
  }
  updateProgress('faces', 1);

  // Step 4: Match pairs — knnMatch + ratio test + RANSAC homography
  setStatus('Matching image pairs (0%)…');
  updateProgress('matching', 0);

  // ── Adaptive min-inliers threshold ─────────────────────────────
  // For small tiles (≤ 512px), the overlap zone contains fewer features
  // and ORB has less room for matches (especially in sky/featureless areas
  // where CLAHE-enhanced ORB still only finds sparse corners).
  // Scale minInliers linearly based on average alignment dimension:
  //   256px → 4 inliers,  512px → 8,  1024+px → 15
  // This prevents grid-layout tiles from being disconnected.
  const avgDim = active.reduce((s, img) => {
    const feat = lastFeatures.get(img.id);
    const sf = feat?.scaleFactor ?? 1;
    return s + Math.max(img.width * sf, img.height * sf);
  }, 0) / active.length;
  const baselineMinInliers = Math.max(4, Math.min(15, Math.round(avgDim / 70)));
  const adaptiveMinInliers = effectiveSettings.minInliers > 0
    ? clamp(Math.round(effectiveSettings.minInliers), 4, 18)
    : baselineMinInliers;
  console.log(`Adaptive minInliers: ${adaptiveMinInliers} (avg dimension ${avgDim.toFixed(0)}px)`);

  // ── Force all-pairs matching for small image sets ──────────────
  // With ≤ 12 images, the O(n²) matching cost is low and ensuring
  // full connectivity (especially in grid-layout tiles) is critical.
  const forceAllPairs = active.length <= 12 || effectiveSettings.matchAllPairs;
  const avgMatchKeypoints = active.reduce((sum, img) => {
    const feat = lastFeatures.get(img.id);
    const kpCount = feat ? feat.keypoints.length / 2 : 0;
    return sum + kpCount;
  }, 0) / active.length;
  const matchPlan = optimizeMatchGraphPlan(
    active.length,
    effectiveSettings.pairWindowW,
    forceAllPairs,
    avgMatchKeypoints,
  );
  if (matchPlan.changed) {
    console.info('[matchGraph-plan]', matchPlan.summary, {
      workloadScore: Number(matchPlan.workloadScore.toFixed(1)),
      avgKeypoints: Number(avgMatchKeypoints.toFixed(0)),
    });
  }

  const edgesPromise = waitForMatchGraphEdges({
    wm: workerManager!,
    candidatePairs: matchPlan.candidatePairs,
    activeCount: active.length,
    avgKeypoints: avgMatchKeypoints,
    timeoutLabel: 'matchGraph',
    onProgress: (pct) => {
      setStatus(`Matching image pairs (${pct}%)…`);
      updateProgress('matching', pct / 100);
    },
  });

  workerManager!.sendCV({
    type: 'matchGraph',
    windowW: matchPlan.windowW,
    ratio: effectiveSettings.ratioTest,
    ransacThreshPx: effectiveSettings.ransacThreshPx,
    minInliers: adaptiveMinInliers,
    matchAllPairs: matchPlan.matchAllPairs,
  });

  const edgesMsg = await edgesPromise;
  updateProgress('matching', 1);
  lastEdges = edgesMsg.edges.map((e: CVEdge) => ({
    i: e.i,
    j: e.j,
    H: new Float64Array(e.HBuffer),
    inliers: new Float32Array(e.inliersBuffer),
    rms: e.rms,
    inlierCount: e.inlierCount,
    isDuplicate: e.isDuplicate || false,
  }));

  // Handle near-duplicates: mark one image in each duplicate pair as excluded
  if (edgesMsg.duplicatePairs && edgesMsg.duplicatePairs.length > 0) {
    const toExclude = new Set<string>();
    const activeIndexById = new Map(active.map((img, idx) => [img.id, idx]));
    for (const [idA, idB] of edgesMsg.duplicatePairs) {
      // Prefer to keep the image that appears earlier in the list
      const idxA = activeIndexById.get(idA);
      const idxB = activeIndexById.get(idB);
      if (idxA !== undefined && idxB !== undefined) {
        const excludeId = idxA < idxB ? idB : idA;
        toExclude.add(excludeId);
      }
    }
    
    if (toExclude.size > 0) {
      const allImages = getState().images;
      const updated = allImages.map(img => 
        toExclude.has(img.id) ? { ...img, excluded: true } : img
      );
      setState({ images: updated });
      // Recompute active array to exclude duplicates from subsequent pipeline stages
      active = updated.filter(i => !i.excluded);
      // Also remove excluded edges
      lastEdges = lastEdges.filter(e => !toExclude.has(e.i) && !toExclude.has(e.j));
      setStatus(`Excluded ${toExclude.size} near-duplicate image(s). ${active.length} remain.`);
    }
  }

  if (lastEdges.length === 0) {
    setStatus('No matching pairs found. Try adding more overlapping images.');
    setState({ pipelineStatus: 'idle' });
    endProgress();
    return;
  }

  const totalInliers = lastEdges.reduce((s, e) => s + e.inlierCount, 0);
  setStatus(`Matching complete — ${lastEdges.length} edges, ${totalInliers} total inliers.`);

  // Dispatch event for match heatmap overlay
  window.dispatchEvent(new CustomEvent('edges-ready'));

  // Step 5: Build MST and compute initial global transforms
  setStatus('Building alignment graph…');
  updateProgress('mst', 0);

  const mstPromise = workerManager!.waitCV('mst', 15000);
  // We also need the follow-up transforms message
  const transformsPromise = new Promise<CVTransformsMsg>((resolve, reject) => {
    let unsub: (() => void) | null = null;
    const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for initial transforms')); }, 30000);
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'transforms') {
        clearTimeout(timer);
        if (unsub) unsub();
        resolve(msg as CVTransformsMsg);
      } else if (msg.type === 'error') {
        clearTimeout(timer);
        if (unsub) unsub();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    };
    unsub = workerManager!.onCV(handler);
  });

  workerManager!.sendCV({ type: 'buildMST' });

  const mstMsg = await mstPromise as CVMSTMsg;
  lastRefId = mstMsg.refId;
  lastMstOrder = mstMsg.order;
  lastMstParent = mstMsg.parent;
  const activeNameById = new Map(active.map((img) => [img.id, img.name]));
  setStatus(`Alignment graph built — ref: ${activeNameById.get(lastRefId || '') ?? lastRefId}, ${lastMstOrder.length} images`);
  updateProgress('mst', 0.5);

  const transformsMsg = await transformsPromise;
  lastTransforms = new Map();
  for (const t of transformsMsg.transforms) {
    lastTransforms.set(t.imageId, {
      imageId: t.imageId,
      T: new Float64Array(t.TBuffer),
    });
  }
  updateProgress('mst', 1);

  // Step 6: Refine transforms via Levenberg-Marquardt bundle adjustment
  setStatus('Refining transforms…');
  updateProgress('refine', 0);

  const refineTransformPromise = new Promise<CVTransformsMsg>((resolve, reject) => {
    let unsub: (() => void) | null = null;
    const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for refine transforms')); }, 60000);
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'transforms') {
        clearTimeout(timer);
        if (unsub) unsub();
        resolve(msg as CVTransformsMsg);
      } else if (msg.type === 'error') {
        clearTimeout(timer);
        if (unsub) unsub();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    };
    unsub = workerManager!.onCV(handler);
  });

  workerManager!.sendCV({
    type: 'refine',
    maxIters: effectiveSettings.refineIters,
    huberDeltaPx: 2.0,
    lambdaInit: 0.01,
    sameCameraSettings: effectiveSettings.sameCameraSettings,
  });

  const refinedMsg = await refineTransformPromise;
  for (const t of refinedMsg.transforms) {
    lastTransforms.set(t.imageId, {
      imageId: t.imageId,
      T: new Float64Array(t.TBuffer),
    });
  }
  updateProgress('refine', 1);

  // Step 6a: Post-BA quality assessment — auto-exclude badly aligned images
  updateProgress('quality', 0);
  setStatus('Assessing alignment quality…');

  const qualityPromise = new Promise<CVQualityAssessmentOutMsg>((resolve, reject) => {
    let unsub: (() => void) | null = null;
    const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for quality assessment')); }, 30000);
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'qualityAssessment') {
        clearTimeout(timer);
        if (unsub) unsub();
        resolve(msg as CVQualityAssessmentOutMsg);
      } else if (msg.type === 'error') {
        clearTimeout(timer);
        if (unsub) unsub();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    };
    unsub = workerManager!.onCV(handler);
  });

  workerManager!.sendCV({ type: 'qualityAssessment', threshold: 5.0 });

  const qualityMsg = await qualityPromise;

  // Auto-exclude recommended images
  if (qualityMsg.excludeIds.length > 0 && qualityMsg.excludeIds.length < active.length - 1) {
    const excludeSet = new Set(qualityMsg.excludeIds);
    const reasons = qualityMsg.quality
      .filter(q => q.isOutlier)
      .map(q => `${q.imageId}: ${q.reason}`)
      .join('; ');
    setStatus(`Auto-excluding ${excludeSet.size} image(s): ${reasons}`);
    console.warn('[quality] Excluding images:', reasons);

    // Mark excluded in appState
    const current = getState();
    const updated = current.images.map(img =>
      excludeSet.has(img.id) ? { ...img, excluded: true } : img
    );
    setState({ images: updated });

    // Filter active list
    active = active.filter(img => !excludeSet.has(img.id));

    // Re-run BA without excluded images
    setStatus('Re-running bundle adjustment without excluded images…');
    const reRefinePromise = new Promise<CVTransformsMsg>((resolve, reject) => {
      let unsub: (() => void) | null = null;
      const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for re-refine transforms')); }, 60000);
      const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
        if (msg.type === 'transforms') {
          clearTimeout(timer);
          if (unsub) unsub();
          resolve(msg as CVTransformsMsg);
        } else if (msg.type === 'error') {
          clearTimeout(timer);
          if (unsub) unsub();
          reject(new Error((msg as any).message || 'Worker error'));
        }
      };
      unsub = workerManager!.onCV(handler);
    });

    workerManager!.sendCV({
      type: 'refine',
      maxIters: effectiveSettings.refineIters,
      huberDeltaPx: 2.0,
      lambdaInit: 0.01,
      sameCameraSettings: effectiveSettings.sameCameraSettings,
    });

    const reRefinedMsg = await reRefinePromise;
    lastTransforms = new Map();
    for (const t of reRefinedMsg.transforms) {
      lastTransforms.set(t.imageId, {
        imageId: t.imageId,
        T: new Float64Array(t.TBuffer),
      });
    }

    // Rebuild MST order for the reduced set
    const reMSTPromise = new Promise<CVMSTMsg>((resolve, reject) => {
      let unsub: (() => void) | null = null;
      const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for MST rebuild')); }, 15000);
      const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
        if (msg.type === 'mst') {
          clearTimeout(timer);
          if (unsub) unsub();
          resolve(msg as CVMSTMsg);
        } else if (msg.type === 'error') {
          clearTimeout(timer);
          if (unsub) unsub();
          reject(new Error((msg as any).message || 'Worker error'));
        }
      };
      unsub = workerManager!.onCV(handler);
    });

    workerManager!.sendCV({ type: 'buildMST' });
    const reMstMsg = await reMSTPromise;
    lastMstOrder = reMstMsg.order;
    lastMstParent = reMstMsg.parent;
    lastRefId = reMstMsg.refId;
    setStatus(`Rebuilt MST for ${active.length} images (ref=${reMstMsg.refId})`);
  } else if (qualityMsg.excludeIds.length >= active.length - 1) {
    console.warn('[quality] Skipping exclusion — would remove all images');
  }

  updateProgress('quality', 1);

  // Step 6b: Exposure compensation (per-image scalar gain)
  lastGains = new Map();
  if (effectiveSettings.exposureComp) {
    setStatus('Computing exposure gains…');
    updateProgress('exposure', 0);

    const exposurePromise = new Promise<CVExposureMsg>((resolve, reject) => {
      let unsub: (() => void) | null = null;
      const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for exposure gains')); }, 30000);
      const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
        if (msg.type === 'exposure') {
          clearTimeout(timer);
          if (unsub) unsub();
          resolve(msg as CVExposureMsg);
        } else if (msg.type === 'error') {
          clearTimeout(timer);
          if (unsub) unsub();
          reject(new Error((msg as any).message || 'Worker error'));
        }
      };
      unsub = workerManager!.onCV(handler);
    });

    workerManager!.sendCV({ type: 'computeExposure', sameCameraSettings: effectiveSettings.sameCameraSettings });

    const exposureMsg = await exposurePromise;
    for (const g of exposureMsg.gains) {
      lastGains.set(g.imageId, {
        gain: g.gain,
        gainR: g.gainR ?? g.gain,
        gainG: g.gainG ?? g.gain,
        gainB: g.gainB ?? g.gain,
      });
    }

    const gainStr = exposureMsg.gains.map(g => `${g.gain.toFixed(3)}`).join(', ');
    setStatus(`Exposure gains: [${gainStr}]`);
  }
  updateProgress('exposure', 1);

  // Step 7: Depth inference (optional, best-effort)
  lastDepthMaps = new Map();
  if (effectiveSettings.depthEnabled && ready.depth) {
    setStatus('Running depth estimation…');
    for (let idx = 0; idx < active.length; idx++) {
      const img = active[idx];
      try {
        // Decode image to RGBA at depth input size
        const bmp = await createImageBitmap(img.file);
        const depthSize = effectiveSettings.depthInputSize;
        const offscreen = new OffscreenCanvas(depthSize, depthSize);
        const ctx = offscreen.getContext('2d')!;
        ctx.drawImage(bmp, 0, 0, depthSize, depthSize);
        bmp.close();
        const imgData = ctx.getImageData(0, 0, depthSize, depthSize);
        const rgbaBuf = imgData.data.buffer as ArrayBuffer;

        workerManager!.sendDepth(
          {
            type: 'infer',
            imageId: img.id,
            rgbaBuffer: rgbaBuf,
            width: depthSize,
            height: depthSize,
          },
          [rgbaBuf],
        );

        const result = await workerManager!.waitDepth('result', 30000) as DepthResultMsg;
        lastDepthMaps.set(img.id, {
          imageId: img.id,
          depth: new Uint16Array(result.depthUint16Buffer),
          width: result.depthW,
          height: result.depthH,
          nearIsOne: result.nearIsOne,
        });
        setStatus(`Depth: ${img.name} (${idx + 1}/${active.length})`);
      } catch (e) {
        console.warn(`Depth inference failed for ${img.name}:`, e);
        // Continue without depth for this image
      }
    }
    setStatus(`Depth estimation complete — ${lastDepthMaps.size}/${active.length} images.`);
  }

  // Step 8: APAP local mesh computation (if meshGrid > 0)
  lastMeshes = new Map();
  if (effectiveSettings.meshGrid > 0 && lastMstOrder.length > 0) {
    setStatus('Computing adaptive meshes (0%)…');
    updateProgress('apap', 0);

    // Send depth data to cv-worker for mesh weighting (via addImage with depth)
    // The depth data was already ingested can be passed as part of computeLocalMesh msg depth field
    // Actually, depth is stored in the cv-worker images[] if we sent it via addImage.
    // We need to re-send addImage with depth if we have it.
    // For now, depth weighting requires depth maps to be part of the image in cv-worker.
    // We'll resend the depth data for images that have it.
    if (lastDepthMaps.size > 0) {
      const imageById = new Map(images.map(img => [img.id, img]));
      for (const [id, dm] of lastDepthMaps) {
        const img = imageById.get(id);
        if (!img) continue;
        const feat = lastFeatures.get(id);
        if (!feat) continue;
        try {
          // Re-add image with depth buffer
          const depthBuf = dm.depth.buffer.slice(0) as ArrayBuffer;
          workerManager!.sendCV({
            type: 'addImage',
            imageId: id,
            grayBuffer: new Uint8ClampedArray(1).buffer as ArrayBuffer, // dummy, already have gray
            width: feat.keypoints.length > 0 ? Math.round(img.width * feat.scaleFactor) : img.width,
            height: feat.keypoints.length > 0 ? Math.round(img.height * feat.scaleFactor) : img.height,
            depth: depthBuf,
          }, [depthBuf]);
          await workerManager!.waitCV('progress', 5000);
        } catch (e) {
          console.warn(`Failed to re-send depth for ${id}, continuing without depth weighting:`, e);
        }
      }
    }

    let meshDone = 0; // separate counter excluding skipped reference image
    for (let idx = 0; idx < lastMstOrder.length; idx++) {
      const nodeId = lastMstOrder[idx];
      const parentId = lastMstParent[nodeId];
      if (!parentId) continue; // skip reference image

      const meshPromise = new Promise<CVMeshMsg>((resolve, reject) => {
        let unsub: (() => void) | null = null;
        const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error(`Timeout waiting for mesh: ${nodeId}`)); }, 30000);
        const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
          if (msg.type === 'mesh' && msg.imageId === nodeId) {
            clearTimeout(timer);
            if (unsub) unsub();
            resolve(msg as CVMeshMsg);
          } else if (msg.type === 'error') {
            clearTimeout(timer);
            if (unsub) unsub();
            reject(new Error((msg as any).message || 'Worker error'));
          }
        };
        unsub = workerManager!.onCV(handler);
      });

      workerManager!.sendCV({
        type: 'computeLocalMesh',
        imageId: nodeId,
        parentId: parentId,
        meshGrid: effectiveSettings.meshGrid,
        sigma: 100,       // spatial sigma in alignment pixels
        depthSigma: 0.1,  // depth sigma (normalized)
        minSupport: 4,
        faceRects: lastFaces.get(nodeId) || [],
        sameCameraSettings: effectiveSettings.sameCameraSettings,
      });

      const meshMsg = await meshPromise;
      lastMeshes.set(nodeId, {
        imageId: nodeId,
        vertices: new Float32Array(meshMsg.verticesBuffer),
        uvs: new Float32Array(meshMsg.uvsBuffer),
        indices: new Uint32Array(meshMsg.indicesBuffer),
        bounds: meshMsg.bounds,
      });

      meshDone++;
      const meshTotal = lastMstOrder.length - 1;
      const meshPct = Math.min(100, Math.round((meshDone / meshTotal) * 100));
      setStatus(`Computing adaptive meshes (${meshPct}%) — ${meshDone}/${meshTotal}`);
      updateProgress('apap', Math.min(1, meshDone / meshTotal));
    }

    setStatus(`Adaptive meshes computed for ${lastMeshes.size} images.`);
  }
  updateProgress('apap', 1);

  endProgress();
  setStatus(`Pipeline complete — ${active.length} images aligned.`);

  // Dispatch event so main.ts can render warped preview
  window.dispatchEvent(new CustomEvent('transforms-ready'));

  } catch (err) {
    console.error('Pipeline error:', err);
    setStatus(`Pipeline error: ${err instanceof Error ? err.message : String(err)}`);
    setState({ pipelineStatus: 'error' });
    endProgress();
  } finally {
    // pipelineStatus is set to 'idle' by the transforms-ready handler
    // in main.ts after compositing completes, avoiding the race where
    // the pipeline is marked idle before renderWarpedPreview finishes.
    // If status is 'error', leave it as-is for user visibility.
  }
}
