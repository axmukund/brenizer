/**
 * Pipeline controller — orchestrates the image stitching pipeline.
 * Manages worker lifecycle, runs pipeline stages in order.
 */

import { createWorkerManager, type WorkerManager } from './workers/workerManager';
import type { CVFeaturesMsg, CVEdgesMsg, CVEdge, CVMSTMsg, CVTransformsMsg, CVMeshMsg, CVExposureMsg } from './workers/workerTypes';
import type { DepthResultMsg } from './workers/workerTypes';
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

/** Global transforms from the last pipeline run. */
export interface GlobalTransform {
  imageId: string;
  T: Float64Array; // 3x3 row-major
}
let lastTransforms: Map<string, GlobalTransform> = new Map();
let lastRefId: string | null = null;
let lastMstOrder: string[] = [];
let lastMstParent: Record<string, string | null> = {};

export function getLastTransforms(): Map<string, GlobalTransform> {
  return lastTransforms;
}
export function getLastRefId(): string | null {
  return lastRefId;
}
export function getLastMstOrder(): string[] {
  return lastMstOrder;
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

export function getLastMeshes(): Map<string, APAPMesh> {
  return lastMeshes;
}

/** Per-image exposure gains (scalar multiplier applied to RGB). */
let lastGains: Map<string, number> = new Map();

export function getLastGains(): Map<string, number> {
  return lastGains;
}

/** Per-image seam masks (Uint8 alpha mask at alignment scale). */
export interface SeamMask {
  imageId: string;
  mask: Uint8Array;     // alpha mask [0..255] at alignment scale
  width: number;
  height: number;
  offsetX: number;      // offset in global composite coords
  offsetY: number;
}
let lastSeamMasks: Map<string, SeamMask> = new Map();

export function getLastSeamMasks(): Map<string, SeamMask> {
  return lastSeamMasks;
}

/** Computed global bounding box of the composite. */
export interface CompositeBounds {
  minX: number; minY: number;
  maxX: number; maxY: number;
  width: number; height: number;
}
let lastCompositeBounds: CompositeBounds | null = null;

export function getLastCompositeBounds(): CompositeBounds | null {
  return lastCompositeBounds;
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

  // Prevent re-entry while pipeline is already running
  if (getState().pipelineStatus === 'running') {
    setStatus('Pipeline already running.');
    return;
  }

  setState({ pipelineStatus: 'running' });
  setStatus('Starting pipeline…');

  try {

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
        let unsub: (() => void) | null = null;
        const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
          if (msg.type === 'features' && msg.imageId === img.id) {
            if (unsub) unsub();
            resolve(msg as CVFeaturesMsg);
          }
        };
        unsub = workerManager!.onCV(handler);
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
    let unsub: (() => void) | null = null;
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'edges') {
        if (unsub) unsub();
        resolve(msg as CVEdgesMsg);
      }
    };
    unsub = workerManager!.onCV(handler);
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
      setStatus(`Excluded ${toExclude.size} near-duplicate image(s). ${active.length - toExclude.size} remain.`);
    }
  }

  if (lastEdges.length === 0) {
    setStatus('No matching pairs found. Try adding more overlapping images.');
    setState({ pipelineStatus: 'idle' });
    return;
  }

  const totalInliers = lastEdges.reduce((s, e) => s + e.inlierCount, 0);
  setStatus(`Matching complete — ${lastEdges.length} edges, ${totalInliers} total inliers.`);

  // Dispatch event for match heatmap overlay
  window.dispatchEvent(new CustomEvent('edges-ready'));

  // Step 5: Build MST and compute initial global transforms
  setStatus('Building MST and initial transforms…');

  const mstPromise = workerManager!.waitCV('mst', 15000);
  // We also need the follow-up transforms message
  const transformsPromise = new Promise<CVTransformsMsg>((resolve) => {
    let unsub: (() => void) | null = null;
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'transforms') {
        if (unsub) unsub();
        resolve(msg as CVTransformsMsg);
      }
    };
    unsub = workerManager!.onCV(handler);
  });

  workerManager!.sendCV({ type: 'buildMST' });

  const mstMsg = await mstPromise as CVMSTMsg;
  lastRefId = mstMsg.refId;
  lastMstOrder = mstMsg.order;
  lastMstParent = mstMsg.parent;
  setStatus(`MST built — ref: ${active.find(i => i.id === lastRefId)?.name ?? lastRefId}, order: ${lastMstOrder.length} images`);

  const transformsMsg = await transformsPromise;
  lastTransforms = new Map();
  for (const t of transformsMsg.transforms) {
    lastTransforms.set(t.imageId, {
      imageId: t.imageId,
      T: new Float64Array(t.TBuffer),
    });
  }

  // Step 6: Refine transforms (LM — placeholder for now, sends back same transforms)
  setStatus('Refining transforms…');

  const refineTransformPromise = new Promise<CVTransformsMsg>((resolve) => {
    let unsub: (() => void) | null = null;
    const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
      if (msg.type === 'transforms') {
        if (unsub) unsub();
        resolve(msg as CVTransformsMsg);
      }
    };
    unsub = workerManager!.onCV(handler);
  });

  workerManager!.sendCV({
    type: 'refine',
    maxIters: settings.refineIters,
    huberDeltaPx: 2.0,
    lambdaInit: 0.01,
  });

  const refinedMsg = await refineTransformPromise;
  for (const t of refinedMsg.transforms) {
    lastTransforms.set(t.imageId, {
      imageId: t.imageId,
      T: new Float64Array(t.TBuffer),
    });
  }

  // Step 6b: Exposure compensation (per-image scalar gain)
  lastGains = new Map();
  if (settings.exposureComp) {
    setStatus('Computing exposure gains…');

    const exposurePromise = new Promise<CVExposureMsg>((resolve) => {
      let unsub: (() => void) | null = null;
      const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
        if (msg.type === 'exposure') {
          if (unsub) unsub();
          resolve(msg as CVExposureMsg);
        }
      };
      unsub = workerManager!.onCV(handler);
    });

    workerManager!.sendCV({ type: 'computeExposure' });

    const exposureMsg = await exposurePromise;
    for (const g of exposureMsg.gains) {
      lastGains.set(g.imageId, g.gain);
    }

    const gainStr = exposureMsg.gains.map(g => `${g.gain.toFixed(3)}`).join(', ');
    setStatus(`Exposure gains: [${gainStr}]`);
  }

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
    setStatus('Computing APAP local meshes…');

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
      }
    }

    for (let idx = 0; idx < lastMstOrder.length; idx++) {
      const nodeId = lastMstOrder[idx];
      const parentId = lastMstParent[nodeId];
      if (!parentId) continue; // skip reference image

      const meshPromise = new Promise<CVMeshMsg>((resolve) => {
        let unsub: (() => void) | null = null;
        const handler = (msg: import('./workers/workerTypes').CVOutMsg) => {
          if (msg.type === 'mesh' && msg.imageId === nodeId) {
            if (unsub) unsub();
            resolve(msg as CVMeshMsg);
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
      });

      const meshMsg = await meshPromise;
      lastMeshes.set(nodeId, {
        imageId: nodeId,
        vertices: new Float32Array(meshMsg.verticesBuffer),
        uvs: new Float32Array(meshMsg.uvsBuffer),
        indices: new Uint32Array(meshMsg.indicesBuffer),
        bounds: meshMsg.bounds,
      });

      setStatus(`APAP mesh: ${idx + 1}/${lastMstOrder.length - 1}`);
    }

    setStatus(`APAP meshes computed for ${lastMeshes.size} images.`);
  }

  setStatus(`Pipeline complete — ${active.length} images aligned.`);

  // Dispatch event so main.ts can render warped preview
  window.dispatchEvent(new CustomEvent('transforms-ready'));

  } finally {
    setState({ pipelineStatus: 'idle' });
  }
}
