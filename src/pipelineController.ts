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
import { setStatus, startProgress, endProgress, updateProgress } from './ui';

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
export async function initWorkers(opts: { enableDepth?: boolean; enableSeam?: boolean } = {}): Promise<{ cv: boolean; depth: boolean; seam: boolean }> {
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
    enableDepth: settings.depthEnabled,
    enableSeam: settings.seamMethod === 'graphcut',
  });
  updateProgress('init', 1);
  if (!ready.cv) {
    setStatus('CV worker failed to initialize. Cannot stitch.');
    setState({ pipelineStatus: 'error' });
    endProgress();
    return;
  }

  setStatus('Preparing images…');
  updateProgress('sendImages', 0);

  // Clear old images from worker state
  await new Promise<void>((resolve, reject) => {
    const timer = setTimeout(() => { unsub(); reject(new Error('Timeout waiting for clearImages')); }, 15000);
    const unsub = workerManager!.onCV((msg) => {
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
    workerManager!.sendCV({ type: 'clearImages' });
  });

  // Step 2: Convert images to grayscale and send to cv-worker
  const scaleFactors: Record<string, number> = {};
  for (let i = 0; i < active.length; i++) {
    const img = active[i];
    setStatus(`Preparing image ${i + 1}/${active.length}: ${img.name}`);
    const { gray, rgbSmall, width, height, scaleFactor } = await imageToGray(img, settings.alignScale);
    scaleFactors[img.id] = scaleFactor;
    const buf = gray.buffer as ArrayBuffer;
    const rgbBuf = rgbSmall.buffer as ArrayBuffer;
    // Send and wait for ack
    await new Promise<void>((resolve, reject) => {
      const timer = setTimeout(() => { unsub(); reject(new Error(`Timeout waiting for addImage ack: ${img.name}`)); }, 30000);
      const unsub = workerManager!.onCV((msg) => {
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
      workerManager!.sendCV(
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
    updateProgress('sendImages', (i + 1) / active.length);
  }

  // Step 3: Compute features (ORB)
  setStatus(`Extracting features (0/${active.length})…`);
  updateProgress('features', 0);

  // Collect features messages as they arrive.
  // Uses a single CV handler to dispatch per-image results, avoiding
  // the broadcast-error problem where one error rejects all promises.
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
  const featUnsub = workerManager!.onCV((msg) => {
    if (msg.type === 'features' && featureResolvers.has(msg.imageId)) {
      const r = featureResolvers.get(msg.imageId)!;
      clearTimeout(r.timer);
      featureResolvers.delete(msg.imageId);
      r.resolve(msg as CVFeaturesMsg);
    } else if (msg.type === 'error') {
      // Only reject remaining promises on a fatal worker error
      for (const [id, r] of featureResolvers) {
        clearTimeout(r.timer);
        r.reject(new Error((msg as any).message || 'Feature extraction failed'));
      }
      featureResolvers.clear();
    }
  });

  workerManager!.sendCV({
    type: 'computeFeatures',
    orbParams: { nFeatures: settings.orbFeatures },
  });

  // Wait for all features
  lastFeatures = new Map();
  let featIdx = 0;
  for (const img of active) {
    const featMsg = await featurePromises.get(img.id)!;
    lastFeatures.set(img.id, {
      imageId: img.id,
      keypoints: new Float32Array(featMsg.keypointsBuffer),
      descriptors: new Uint8Array(featMsg.descriptorsBuffer),
      descCols: featMsg.descCols,
      scaleFactor: scaleFactors[img.id],
    });
    featIdx++;
    const numKp = new Float32Array(featMsg.keypointsBuffer).length / 2;
    setStatus(`Features: ${img.name} — ${numKp} keypoints (${featIdx}/${active.length})`);
    updateProgress('features', featIdx / active.length);
  }

  const totalKp = Array.from(lastFeatures.values()).reduce(
    (s, f) => s + f.keypoints.length / 2, 0,
  );
  featUnsub(); // unsubscribe the feature collection handler
  setStatus(`Feature extraction complete — ${totalKp} keypoints across ${active.length} images.`);

  // Dispatch custom event so main.ts can draw keypoint overlay
  window.dispatchEvent(new CustomEvent('features-ready'));

  // Step 3a: Saliency computation (AI object/texture/blur detection)
  lastSaliency = new Map();
  if (settings.saliencyEnabled) {
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
  if (settings.vignetteCorrection) {
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

    workerManager!.sendCV({ type: 'computeVignetting', pooled: settings.sameCameraSettings });

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

  // Listen for matching progress from worker
  const matchProgressUnsub = workerManager!.onCV((msg) => {
    if (msg.type === 'progress' && msg.stage === 'matching' && msg.percent !== undefined) {
      const pct = msg.percent;
      setStatus(`Matching image pairs (${pct}%)…`);
      updateProgress('matching', pct / 100);
    }
  });

  const edgesPromise = new Promise<CVEdgesMsg>((resolve, reject) => {
    let unsub: (() => void) | null = null;
    const timer = setTimeout(() => { if (unsub) unsub(); reject(new Error('Timeout waiting for matchGraph edges')); }, 120000);
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'edges') {
        clearTimeout(timer);
        if (unsub) unsub();
        resolve(msg as CVEdgesMsg);
      } else if (msg.type === 'error') {
        clearTimeout(timer);
        if (unsub) unsub();
        reject(new Error((msg as any).message || 'Worker error'));
      }
    };
    unsub = workerManager!.onCV(handler);
  });

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
  const adaptiveMinInliers = Math.max(4, Math.min(15, Math.round(avgDim / 70)));
  console.log(`Adaptive minInliers: ${adaptiveMinInliers} (avg dimension ${avgDim.toFixed(0)}px)`);

  // ── Force all-pairs matching for small image sets ──────────────
  // With ≤ 12 images, the O(n²) matching cost is low and ensuring
  // full connectivity (especially in grid-layout tiles) is critical.
  const forceAllPairs = active.length <= 12 || settings.matchAllPairs;

  workerManager!.sendCV({
    type: 'matchGraph',
    windowW: settings.pairWindowW,
    ratio: settings.ratioTest,
    ransacThreshPx: settings.ransacThreshPx,
    minInliers: adaptiveMinInliers,
    matchAllPairs: forceAllPairs,
  });

  let edgesMsg: CVEdgesMsg;
  try {
    edgesMsg = await edgesPromise;
  } finally {
    matchProgressUnsub(); // always clean up the progress listener
  }
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
    for (const [idA, idB] of edgesMsg.duplicatePairs) {
      // Prefer to keep the image that appears earlier in the list
      const imgA = active.find(i => i.id === idA);
      const imgB = active.find(i => i.id === idB);
      if (imgA && imgB) {
        const idxA = active.indexOf(imgA);
        const idxB = active.indexOf(imgB);
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
  setStatus(`Alignment graph built — ref: ${active.find(i => i.id === lastRefId)?.name ?? lastRefId}, ${lastMstOrder.length} images`);
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
    maxIters: settings.refineIters,
    huberDeltaPx: 2.0,
    lambdaInit: 0.01,
    sameCameraSettings: settings.sameCameraSettings,
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
      maxIters: settings.refineIters,
      huberDeltaPx: 2.0,
      lambdaInit: 0.01,
      sameCameraSettings: settings.sameCameraSettings,
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
  if (settings.exposureComp) {
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

    workerManager!.sendCV({ type: 'computeExposure', sameCameraSettings: settings.sameCameraSettings });

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
  if (settings.depthEnabled && ready.depth) {
    setStatus('Running depth estimation…');
    for (let idx = 0; idx < active.length; idx++) {
      const img = active[idx];
      try {
        // Decode image to RGBA at depth input size
        const bmp = await createImageBitmap(img.file);
        const depthSize = settings.depthInputSize;
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
  if (settings.meshGrid > 0 && lastMstOrder.length > 0) {
    setStatus('Computing adaptive meshes (0%)…');
    updateProgress('apap', 0);

    // Send depth data to cv-worker for mesh weighting (via addImage with depth)
    // The depth data was already ingested can be passed as part of computeLocalMesh msg depth field
    // Actually, depth is stored in the cv-worker images[] if we sent it via addImage.
    // We need to re-send addImage with depth if we have it.
    // For now, depth weighting requires depth maps to be part of the image in cv-worker.
    // We'll resend the depth data for images that have it.
    if (lastDepthMaps.size > 0) {
      for (const [id, dm] of lastDepthMaps) {
        const img = images.find(i => i.id === id);
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
        meshGrid: settings.meshGrid,
        sigma: 100,       // spatial sigma in alignment pixels
        depthSigma: 0.1,  // depth sigma (normalized)
        minSupport: 4,
        faceRects: lastFaces.get(nodeId) || [],
        sameCameraSettings: settings.sameCameraSettings,
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
