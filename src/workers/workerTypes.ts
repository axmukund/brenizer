/**
 * Shared message type definitions for all workers.
 * These match the contracts defined in SPEC.md §10.
 */

// ── depth.worker messages ────────────────────────────────

export interface DepthInitMsg {
  type: 'init';
  baseUrl: string;
  modelPath: string;
  preferWebGPU: boolean;
  targetSize: number;
  epPreference?: string;
}

export interface DepthInferMsg {
  type: 'infer';
  imageId: string;
  rgbaBuffer: ArrayBuffer;
  width: number;
  height: number;
}

export type DepthInMsg = DepthInitMsg | DepthInferMsg;

export interface DepthProgressMsg {
  type: 'progress';
  stage: string;
  done: number;
  total: number;
  info?: string;
}

export interface DepthResultMsg {
  type: 'result';
  imageId: string;
  depthUint16Buffer: ArrayBuffer;
  depthW: number;
  depthH: number;
  nearIsOne: boolean;
  epUsed: string;
}

export interface DepthErrorMsg {
  type: 'error';
  imageId?: string;
  message: string;
}

export type DepthOutMsg = DepthProgressMsg | DepthResultMsg | DepthErrorMsg;

// ── cv-worker messages ───────────────────────────────────

export interface CVInitMsg {
  type: 'init';
  baseUrl: string;
  opencvPath?: string;
}

export interface CVClearImagesMsg {
  type: 'clearImages';
}

export interface CVAddImageMsg {
  type: 'addImage';
  imageId: string;
  grayBuffer: ArrayBuffer;
  width: number;
  height: number;
  rgbSmallBuffer?: ArrayBuffer;
  depth?: ArrayBuffer;
}

export interface CVComputeFeaturesMsg {
  type: 'computeFeatures';
  orbParams: { nFeatures?: number };
}

export interface CVComputeSaliencyMsg {
  type: 'computeSaliency';
}

export interface CVComputeVignettingMsg {
  type: 'computeVignetting';
}

export interface CVMatchGraphMsg {
  type: 'matchGraph';
  windowW: number;
  ratio: number;
  ransacThreshPx: number;
  minInliers: number;
  matchAllPairs: boolean;
}

export interface CVBuildGraphMsg {
  type: 'buildGraph';
}

export interface CVRefineMsg {
  type: 'refine';
  maxIters: number;
  huberDeltaPx: number;
  lambdaInit: number;
}

export interface CVComputeExposureMsg {
  type: 'computeExposure';
}

export interface CVBuildMSTMsg {
  type: 'buildMST';
}

export interface CVComputeLocalMeshMsg {
  type: 'computeLocalMesh';
  imageId: string;
  parentId: string;
  meshGrid: number;
  sigma: number;
  depthSigma: number;
  minSupport: number;
  faceRects?: Array<{ x: number; y: number; width: number; height: number; confidence: number }>;
}

export type CVInMsg =
  | CVInitMsg
  | CVClearImagesMsg
  | CVAddImageMsg
  | CVComputeFeaturesMsg
  | CVComputeSaliencyMsg
  | CVComputeVignettingMsg
  | CVMatchGraphMsg
  | CVBuildGraphMsg
  | CVRefineMsg
  | CVComputeExposureMsg
  | CVBuildMSTMsg
  | CVComputeLocalMeshMsg;

export interface CVProgressMsg {
  type: 'progress';
  stage: string;
  percent: number;
  info?: string;
}

export interface CVFeaturesMsg {
  type: 'features';
  imageId: string;
  keypointsBuffer: ArrayBuffer;
  descriptorsBuffer: ArrayBuffer;
  descCols: number;
}

export interface CVSaliencyMsg {
  type: 'saliency';
  imageId: string;
  saliencyBuffer: ArrayBuffer;
  width: number;
  height: number;
  blurScore: number;
}

export interface CVVignettingMsg {
  type: 'vignetting';
  imageId: string;
  vignetteParams: { a: number; b: number };
}

export interface CVEdge {
  i: string;
  j: string;
  HBuffer: ArrayBuffer;
  inliersBuffer: ArrayBuffer;
  rms: number;
  inlierCount: number;
  isDuplicate?: boolean;
}

export interface CVEdgesMsg {
  type: 'edges';
  edges: CVEdge[];
  duplicatePairs?: [string, string][];
}

export interface CVTransformsMsg {
  type: 'transforms';
  refId: string;
  transforms: { imageId: string; TBuffer: ArrayBuffer }[];
}

export interface CVExposureMsg {
  type: 'exposure';
  gains: { imageId: string; gain: number; gainR?: number; gainG?: number; gainB?: number }[];
}

export interface CVMSTMsg {
  type: 'mst';
  refId: string | null;
  order: string[];
  parent: Record<string, string | null>;
}

export interface CVMeshMsg {
  type: 'mesh';
  imageId: string;
  verticesBuffer: ArrayBuffer;
  uvsBuffer: ArrayBuffer;
  indicesBuffer: ArrayBuffer;
  bounds: { minX: number; minY: number; maxX: number; maxY: number };
}

export interface CVErrorMsg {
  type: 'error';
  message: string;
}

export type CVOutMsg =
  | CVProgressMsg
  | CVFeaturesMsg
  | CVSaliencyMsg
  | CVVignettingMsg
  | CVEdgesMsg
  | CVTransformsMsg
  | CVExposureMsg
  | CVMSTMsg
  | CVMeshMsg
  | CVErrorMsg;

// ── seam-worker messages ─────────────────────────────────

export interface SeamInitMsg {
  type: 'init';
  baseUrl: string;
  maxflowPath: string;
}

export interface SeamSolveMsg {
  type: 'solve';
  jobId: string;
  gridW: number;
  gridH: number;
  dataCostsBuffer: ArrayBuffer;
  edgeWeightsBuffer?: ArrayBuffer;
  hardConstraintsBuffer: ArrayBuffer;
  params: Record<string, number>;
}

export type SeamInMsg = SeamInitMsg | SeamSolveMsg;

export interface SeamResultMsg {
  type: 'result';
  jobId: string;
  labelsBuffer: ArrayBuffer;
}

export interface SeamErrorMsg {
  type: 'error';
  jobId?: string;
  message: string;
}

export type SeamOutMsg = SeamResultMsg | SeamErrorMsg | CVProgressMsg;
