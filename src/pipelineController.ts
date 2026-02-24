/**
 * Pipeline controller — orchestrates the image stitching pipeline.
 * Manages worker lifecycle, runs pipeline stages in order.
 */

import { createWorkerManager, type WorkerManager } from './workers/workerManager';
import type { CVFeaturesMsg, CVEdgesMsg, CVEdge } from './workers/workerTypes';
import { getState, setState, type ImageEntry } from './appState';
import { setStatus } from './ui';

let workerManager: WorkerManager | null = null;

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
}
let lastEdges: MatchEdge[] = [];

export function getLastFeatures(): Map<string, ImageFeatures> {
  return lastFeatures;
}

export function getLastEdges(): MatchEdge[] {
  return lastEdges;
}

export function getWorkerManager(): WorkerManager | null {
  return workerManager;
}

/** Initialize all workers. Returns readiness status. */
export async function initWorkers(): Promise<{ cv: boolean; depth: boolean; seam: boolean }> {
  if (workerManager) {
    workerManager.dispose();
  }
  workerManager = createWorkerManager();
  setStatus('Initializing workers…');

  const result = await workerManager.initAll();

  const parts: string[] = [];
  if (result.cv) parts.push('CV ✓');
  else parts.push('CV ✗');
  if (result.depth) parts.push('Depth ✓');
  else parts.push('Depth ✗');
  if (result.seam) parts.push('Seam ✓');
  else parts.push('Seam ✗');

  setStatus(`Workers: ${parts.join(' | ')}`);
  return result;
}

/**
 * Decode an image file to grayscale at the alignment scale.
 * Returns { gray, width, height, scaleFactor }.
 */
async function imageToGray(
  entry: ImageEntry,
  alignScale: number,
): Promise<{ gray: Uint8ClampedArray; width: number; height: number; scaleFactor: number }> {
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

  // Convert to grayscale (luminance)
  const gray = new Uint8ClampedArray(w * h);
  for (let i = 0; i < w * h; i++) {
    const r = rgba[i * 4];
    const g = rgba[i * 4 + 1];
    const b = rgba[i * 4 + 2];
    gray[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
  }

  return { gray, width: w, height: h, scaleFactor };
}

/** Run the full stitch preview pipeline. */
export async function runStitchPreview(): Promise<void> {
  const { images, settings } = getState();
  const active = images.filter(i => !i.excluded);

  if (active.length < 2) {
    setStatus('Need at least 2 images to stitch.');
    return;
  }

  if (!settings) {
    setStatus('Settings not loaded.');
    return;
  }

  setState({ pipelineStatus: 'running' });
  setStatus('Starting pipeline…');

  // Step 1: Init workers
  const ready = await initWorkers();
  if (!ready.cv) {
    setStatus('CV worker failed to initialize. Cannot stitch.');
    setState({ pipelineStatus: 'error' });
    return;
  }

  setStatus('Extracting features…');

  // Step 2: Convert images to grayscale and send to cv-worker
  const scaleFactors: Record<string, number> = {};
  for (const img of active) {
    const { gray, width, height, scaleFactor } = await imageToGray(img, settings.alignScale);
    scaleFactors[img.id] = scaleFactor;
    const buf = gray.buffer as ArrayBuffer;
    workerManager!.sendCV(
      {
        type: 'addImage',
        imageId: img.id,
        grayBuffer: buf,
        width,
        height,
      },
      [buf],
    );
    // Wait for ack
    await workerManager!.waitCV('progress', 5000);
  }

  // Step 3: Compute features (ORB)
  // Collect features messages as they arrive
  const featurePromises = new Map<string, Promise<CVFeaturesMsg>>();
  for (const img of active) {
    featurePromises.set(
      img.id,
      new Promise<CVFeaturesMsg>((resolve) => {
        const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
          if (msg.type === 'features' && msg.imageId === img.id) {
            resolve(msg as CVFeaturesMsg);
          }
        };
        workerManager!.onCV(handler);
      }),
    );
  }

  workerManager!.sendCV({
    type: 'computeFeatures',
    orbParams: { nFeatures: settings.orbFeatures },
  });

  // Wait for all features
  lastFeatures = new Map();
  for (const img of active) {
    const featMsg = await featurePromises.get(img.id)!;
    lastFeatures.set(img.id, {
      imageId: img.id,
      keypoints: new Float32Array(featMsg.keypointsBuffer),
      descriptors: new Uint8Array(featMsg.descriptorsBuffer),
      descCols: featMsg.descCols,
      scaleFactor: scaleFactors[img.id],
    });
    const numKp = new Float32Array(featMsg.keypointsBuffer).length / 2;
    setStatus(`Features: ${img.name} — ${numKp} keypoints`);
  }

  const totalKp = Array.from(lastFeatures.values()).reduce(
    (s, f) => s + f.keypoints.length / 2, 0,
  );
  setStatus(`Feature extraction complete — ${totalKp} keypoints across ${active.length} images.`);

  // Dispatch custom event so main.ts can draw keypoint overlay
  window.dispatchEvent(new CustomEvent('features-ready'));

  // Step 4: Match pairs — knnMatch + ratio test + RANSAC homography
  setStatus('Matching image pairs…');

  const edgesPromise = new Promise<CVEdgesMsg>((resolve) => {
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'edges') {
        resolve(msg as CVEdgesMsg);
      }
    };
    workerManager!.onCV(handler);
  });

  workerManager!.sendCV({
    type: 'matchGraph',
    windowW: settings.pairWindowW,
    ratio: settings.ratioTest,
    ransacThreshPx: settings.ransacThreshPx,
    minInliers: 15,
    matchAllPairs: settings.matchAllPairs,
  });

  const edgesMsg = await edgesPromise;
  lastEdges = edgesMsg.edges.map((e: CVEdge) => ({
    i: e.i,
    j: e.j,
    H: new Float64Array(e.HBuffer),
    inliers: new Float32Array(e.inliersBuffer),
    rms: e.rms,
    inlierCount: e.inlierCount,
  }));

  if (lastEdges.length === 0) {
    setStatus('No matching pairs found. Try adding more overlapping images.');
    setState({ pipelineStatus: 'idle' });
    return;
  }

  const totalInliers = lastEdges.reduce((s, e) => s + e.inlierCount, 0);
  setStatus(`Matching complete — ${lastEdges.length} edges, ${totalInliers} total inliers.`);

  // Dispatch event for match heatmap overlay
  window.dispatchEvent(new CustomEvent('edges-ready'));

  setState({ pipelineStatus: 'idle' });
}
