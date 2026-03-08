import { createProgram } from './programs';
import { createEmptyTexture, createTextureFromData, type ManagedTexture } from './textures';
import { createFBO, type ManagedFBO } from './framebuffers';
import type { FaceRectComposite } from './composition';

export type SeamAccelerationTier = 'desktopTurbo' | 'webgpu' | 'webglGrid' | 'legacyCpu';

export interface SeamColorTransferStats {
  apply: boolean;
  sampleWeight: number;
  gain: [number, number, number];
  offset: [number, number, number];
  meanComp: [number, number, number];
  meanNew: [number, number, number];
  stdComp: [number, number, number];
  stdNew: [number, number, number];
}

export interface CompactSeamGraphBuildResult {
  gridW: number;
  gridH: number;
  dataCosts: Float32Array;
  edgeWeightsH: Float32Array;
  edgeWeightsV: Float32Array;
  hardConstraints: Uint8Array;
  ghostPenalty: Float32Array;
  colorTransferStats: SeamColorTransferStats;
  colorTransferStatsBuffer: Float32Array;
  ghostPenaltyBuffer: Float32Array;
  readbackBytes: number;
  summaryMs: number;
  buildMs: number;
  backendId: string;
  resolvedBlockSize: number;
  sampleGrid: number;
  ghostMedianDiff: number;
  ghostThreshold: number;
  lightingSoftStart: number;
  lightingSoftEnd: number;
  summaryBuffers: CompactSeamSummaryBuffers;
}

export interface CompactSeamSummaryBuffers {
  gridW: number;
  gridH: number;
  blockSize: number;
  sampleGrid: number;
  compMean: Float32Array;
  compSq: Float32Array;
  newMean: Float32Array;
  newSq: Float32Array;
}

export interface BuildCompactSeamGraphArgs {
  compositeTex: WebGLTexture;
  newTex: WebGLTexture;
  width: number;
  height: number;
  blockSize: number;
  faceRects?: FaceRectComposite[];
  saliencyGrid?: Float32Array | null;
  tier: SeamAccelerationTier;
}

export interface BuildCompactGraphFromSummariesArgs {
  width: number;
  height: number;
  blockSize: number;
  sampleGrid: number;
  compMean: Float32Array;
  compSq: Float32Array;
  newMean: Float32Array;
  newSq: Float32Array;
  faceRects?: FaceRectComposite[];
  saliencyGrid?: Float32Array | null;
  summaryMs: number;
  readbackBytes: number;
  backendId: string;
}

export interface SummarizeTextureArgs {
  sourceTex: WebGLTexture;
  width: number;
  height: number;
  blockSize: number;
  tier: SeamAccelerationTier;
}

export interface CompactTextureSummaryResult {
  gridW: number;
  gridH: number;
  blockSize: number;
  sampleGrid: number;
  mean: Float32Array;
  sq: Float32Array;
  readbackBytes: number;
  summaryMs: number;
  backendId: string;
}

export interface BuildMaskTextureArgs {
  labels: Uint8Array;
  gridW: number;
  gridH: number;
  width: number;
  height: number;
  blockSize: number;
  featherRadius: number;
  ghostPenalty: Float32Array;
  ghostThreshold: number;
  lightingSoftStart: number;
  lightingSoftEnd: number;
  compositeTex: WebGLTexture;
  newTex: WebGLTexture;
  keyCoverageTex?: WebGLTexture | null;
  sameCameraSettings?: boolean;
}

export interface SeamAccelerator {
  buildCompactGraph(args: BuildCompactSeamGraphArgs): CompactSeamGraphBuildResult | null;
  summarizeTexture(args: SummarizeTextureArgs): CompactTextureSummaryResult | null;
  applyColorTransfer(
    sourceTex: WebGLTexture,
    width: number,
    height: number,
    stats: SeamColorTransferStats,
  ): ManagedTexture;
  copyTexture(
    sourceTex: WebGLTexture,
    width: number,
    height: number,
  ): ManagedTexture;
  buildMaskTexture(args: BuildMaskTextureArgs): ManagedTexture;
  dispose(): void;
}

const CONTENT_ALPHA_THRESHOLD = 10 / 255;
const MASK_HARD_LOW = 8 / 255;
const MASK_HARD_HIGH = 247 / 255;

const FULLSCREEN_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const BLOCK_SUMMARY_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec2 u_sourceSize;
uniform float u_blockSize;
uniform int u_samples;
uniform int u_mode;
uniform float u_alphaThreshold;
out vec4 fragColor;
void main() {
  vec2 cell = floor(gl_FragCoord.xy - vec2(0.5));
  vec2 blockOrigin = cell * u_blockSize;
  vec3 sum = vec3(0.0);
  vec3 sumSq = vec3(0.0);
  float count = 0.0;
  for (int sy = 0; sy < 6; sy++) {
    if (sy >= u_samples) break;
    for (int sx = 0; sx < 6; sx++) {
      if (sx >= u_samples) break;
      vec2 offset = (vec2(float(sx) + 0.5, float(sy) + 0.5) / float(u_samples)) * u_blockSize;
      vec2 px = min(blockOrigin + offset, u_sourceSize - vec2(1.0));
      vec4 sampleColor = texture(u_texture, (px + vec2(0.5)) / u_sourceSize);
      if (sampleColor.a <= u_alphaThreshold) continue;
      sum += sampleColor.rgb;
      sumSq += sampleColor.rgb * sampleColor.rgb;
      count += 1.0;
    }
  }
  float totalSamples = float(u_samples * u_samples);
  if (count <= 0.0) {
    fragColor = vec4(0.0);
    return;
  }
  if (u_mode == 0) {
    fragColor = vec4(sum / count, count / totalSamples);
  } else {
    fragColor = vec4(sumSq / count, count / totalSamples);
  }
}
`;

const COLOR_TRANSFER_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec3 u_gain;
uniform vec3 u_offset;
out vec4 fragColor;
void main() {
  vec4 c = texture(u_texture, v_uv);
  c.rgb = clamp(c.rgb * u_gain + u_offset, 0.0, 1.0);
  fragColor = c;
}
`;

const MASK_EXPAND_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_labelTex;
uniform sampler2D u_composite;
uniform sampler2D u_newImage;
uniform sampler2D u_keyCoverage;
uniform float u_useKey;
uniform vec2 u_gridSize;
uniform vec2 u_outputSize;
uniform float u_blockSize;
uniform float u_alphaThreshold;
out vec4 fragColor;
void main() {
  vec2 px = floor(gl_FragCoord.xy - vec2(0.5));
  vec2 cell = floor(px / u_blockSize);
  vec2 labelUv = (cell + vec2(0.5)) / u_gridSize;
  float m = texture(u_labelTex, labelUv).r;
  vec2 uv = (px + vec2(0.5)) / u_outputSize;
  float compA = texture(u_composite, uv).a;
  float newA = texture(u_newImage, uv).a;
  if (newA <= u_alphaThreshold) {
    m = 0.0;
  } else if (compA <= u_alphaThreshold) {
    m = 1.0;
  } else if (u_useKey > 0.5) {
    float keyA = texture(u_keyCoverage, uv).a;
    if (keyA > u_alphaThreshold && compA > u_alphaThreshold && newA > u_alphaThreshold) {
      m = 0.0;
    }
  }
  fragColor = vec4(m, m, m, 1.0);
}
`;

const MASK_REFINE_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_mask;
uniform sampler2D u_labelTex;
uniform sampler2D u_penaltyTex;
uniform sampler2D u_composite;
uniform sampler2D u_newImage;
uniform sampler2D u_keyCoverage;
uniform float u_useKey;
uniform vec2 u_gridSize;
uniform vec2 u_outputSize;
uniform float u_blockSize;
uniform float u_alphaThreshold;
uniform float u_hardThreshold;
uniform float u_softStart;
uniform float u_softEnd;
out vec4 fragColor;
void main() {
  vec2 px = floor(gl_FragCoord.xy - vec2(0.5));
  vec2 cell = floor(px / u_blockSize);
  vec2 gridUv = (cell + vec2(0.5)) / u_gridSize;
  vec2 uv = (px + vec2(0.5)) / u_outputSize;
  float label = texture(u_labelTex, gridUv).r;
  float penalty = texture(u_penaltyTex, gridUv).r;
  float m = texture(u_mask, uv).r;
  float compA = texture(u_composite, uv).a;
  float newA = texture(u_newImage, uv).a;
  if (newA <= u_alphaThreshold) {
    m = 0.0;
  } else if (compA <= u_alphaThreshold) {
    m = 1.0;
  } else {
    if (u_useKey > 0.5) {
      float keyA = texture(u_keyCoverage, uv).a;
      if (keyA > u_alphaThreshold && compA > u_alphaThreshold && newA > u_alphaThreshold) {
        m = 0.0;
      }
    }
    if (m > ${MASK_HARD_LOW.toFixed(8)} && m < ${MASK_HARD_HIGH.toFixed(8)}) {
      if (penalty >= u_hardThreshold) {
        m = label >= 0.5 ? 1.0 : 0.0;
      } else if (penalty > u_softStart) {
        float t = clamp((penalty - u_softStart) / max(1e-5, u_softEnd - u_softStart), 0.0, 1.0);
        float blend = 0.08 + 0.26 * t;
        m = mix(m, 0.5, blend);
      }
    }
  }
  fragColor = vec4(m, m, m, 1.0);
}
`;

const BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec2 u_direction;
uniform float u_radius;
out vec4 fragColor;
void main() {
  vec4 sum = vec4(0.0);
  float totalW = 0.0;
  float sigma = max(0.5, u_radius * 0.5);
  float invSigma2 = 1.0 / (2.0 * sigma * sigma);
  int iRadius = int(u_radius);
  for (int i = -128; i <= 128; i++) {
    if (i < -iRadius) continue;
    if (i > iRadius) break;
    float fi = float(i);
    float w = exp(-fi * fi * invSigma2);
    sum += texture(u_texture, v_uv + u_direction * fi) * w;
    totalW += w;
  }
  fragColor = sum / max(totalW, 1e-5);
}
`;

function clampUnit(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function colorMismatchNormalized(
  comp: Float32Array,
  compIdx: number,
  nextComp: Float32Array,
  nextIdx: number,
): number {
  const compR = comp[compIdx];
  const compG = comp[compIdx + 1];
  const compB = comp[compIdx + 2];
  const newR = nextComp[nextIdx];
  const newG = nextComp[nextIdx + 1];
  const newB = nextComp[nextIdx + 2];
  const compLum = 0.2126 * compR + 0.7152 * compG + 0.0722 * compB;
  const newLum = 0.2126 * newR + 0.7152 * newG + 0.0722 * newB;
  const lumDiff = Math.abs(compLum - newLum);
  const chromaDiff = (Math.abs(compR - newR) + Math.abs(compG - newG) + Math.abs(compB - newB)) / 3;
  return lumDiff * 0.6 + chromaDiff * 0.4;
}

function computeBlockDistanceField(hasData: Uint8Array, gridW: number, gridH: number): Float32Array {
  const dist = new Float32Array(gridW * gridH);
  const inf = gridW + gridH;
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const idx = gy * gridW + gx;
      if (!hasData[idx]) {
        dist[idx] = 0;
        continue;
      }
      const touchesGridEdge = gx === 0 || gy === 0 || gx === gridW - 1 || gy === gridH - 1;
      const touchesVoid =
        (gx > 0 && !hasData[idx - 1])
        || (gx + 1 < gridW && !hasData[idx + 1])
        || (gy > 0 && !hasData[idx - gridW])
        || (gy + 1 < gridH && !hasData[idx + gridW]);
      dist[idx] = touchesGridEdge || touchesVoid ? 0 : inf;
    }
  }
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const idx = gy * gridW + gx;
      if (gx > 0) dist[idx] = Math.min(dist[idx], dist[idx - 1] + 1);
      if (gy > 0) dist[idx] = Math.min(dist[idx], dist[idx - gridW] + 1);
    }
  }
  for (let gy = gridH - 1; gy >= 0; gy--) {
    for (let gx = gridW - 1; gx >= 0; gx--) {
      const idx = gy * gridW + gx;
      if (gx + 1 < gridW) dist[idx] = Math.min(dist[idx], dist[idx + 1] + 1);
      if (gy + 1 < gridH) dist[idx] = Math.min(dist[idx], dist[idx + gridW] + 1);
    }
  }
  return dist;
}

function createExpandedFaces(faceRects: FaceRectComposite[]): Array<{ imageLabel: 0 | 1; left: number; top: number; right: number; bottom: number }> {
  return faceRects.map((face) => {
    const margin = Math.max(face.width, face.height) * 0.5;
    return {
      imageLabel: face.imageLabel,
      left: face.x - margin,
      top: face.y - margin,
      right: face.x + face.width + margin,
      bottom: face.y + face.height + margin,
    };
  });
}

function computeColorTransferStats(
  compMean: Float32Array,
  compSq: Float32Array,
  newMean: Float32Array,
  newSq: Float32Array,
  gridW: number,
  gridH: number,
): { stats: SeamColorTransferStats; buffer: Float32Array } {
  let weightSum = 0;
  const sumC = [0, 0, 0];
  const sumN = [0, 0, 0];
  const sumC2 = [0, 0, 0];
  const sumN2 = [0, 0, 0];
  for (let i = 0; i < gridW * gridH; i++) {
    const meanOff = i * 4;
    const compCoverage = compMean[meanOff + 3];
    const newCoverage = newMean[meanOff + 3];
    const overlapWeight = Math.min(compCoverage, newCoverage);
    if (overlapWeight < 0.25) continue;
    weightSum += overlapWeight;
    for (let ch = 0; ch < 3; ch++) {
      sumC[ch] += compMean[meanOff + ch] * overlapWeight;
      sumN[ch] += newMean[meanOff + ch] * overlapWeight;
      sumC2[ch] += compSq[meanOff + ch] * overlapWeight;
      sumN2[ch] += newSq[meanOff + ch] * overlapWeight;
    }
  }

  const meanComp: [number, number, number] = [0, 0, 0];
  const meanNew: [number, number, number] = [0, 0, 0];
  const stdComp: [number, number, number] = [0, 0, 0];
  const stdNew: [number, number, number] = [0, 0, 0];
  const gain: [number, number, number] = [1, 1, 1];
  const offset: [number, number, number] = [0, 0, 0];
  let apply = false;

  if (weightSum > 1) {
    for (let ch = 0; ch < 3; ch++) {
      meanComp[ch] = sumC[ch] / weightSum;
      meanNew[ch] = sumN[ch] / weightSum;
      stdComp[ch] = Math.sqrt(Math.max(0, sumC2[ch] / weightSum - meanComp[ch] * meanComp[ch]));
      stdNew[ch] = Math.sqrt(Math.max(0, sumN2[ch] / weightSum - meanNew[ch] * meanNew[ch]));
      if (stdNew[ch] > 1e-3) {
        gain[ch] = Math.max(0.5, Math.min(2.0, stdComp[ch] / stdNew[ch]));
        offset[ch] = meanComp[ch] - gain[ch] * meanNew[ch];
        if (Math.abs(gain[ch] - 1.0) > 0.03 || Math.abs(offset[ch]) > (3 / 255)) {
          apply = true;
        }
      }
    }
  }

  return {
    stats: {
      apply,
      sampleWeight: weightSum,
      gain,
      offset,
      meanComp,
      meanNew,
      stdComp,
      stdNew,
    },
    buffer: new Float32Array([
      apply ? 1 : 0,
      weightSum,
      ...meanComp,
      ...meanNew,
      ...stdComp,
      ...stdNew,
      ...gain,
      ...offset,
    ]),
  };
}

export function resolveCompactSeamGrid(
  width: number,
  height: number,
  blockSize: number,
): { gridW: number; gridH: number } {
  return {
    gridW: Math.max(1, Math.ceil(width / blockSize)),
    gridH: Math.max(1, Math.ceil(height / blockSize)),
  };
}

export function resolveCompactSummarySampleGrid(tier: SeamAccelerationTier): number {
  return tier === 'desktopTurbo' ? 5 : 4;
}

export function buildCompactGraphFromSummaries(
  args: BuildCompactGraphFromSummariesArgs,
): CompactSeamGraphBuildResult {
  const { gridW, gridH } = resolveCompactSeamGrid(args.width, args.height, args.blockSize);
  const buildStart = performance.now();
  const nNodes = gridW * gridH;
  const dataCosts = new Float32Array(nNodes * 2);
  const hardConstraints = new Uint8Array(nNodes);
  const compHas = new Uint8Array(nNodes);
  const newHas = new Uint8Array(nNodes);
  const ghostPenalty = new Float32Array(nNodes);
  const expandedFaces = createExpandedFaces(args.faceRects ?? []);
  const saliencyGrid = args.saliencyGrid && args.saliencyGrid.length >= nNodes ? args.saliencyGrid : null;

  for (let i = 0; i < nNodes; i++) {
    const off = i * 4;
    compHas[i] = args.compMean[off + 3] >= 0.45 ? 1 : 0;
    newHas[i] = args.newMean[off + 3] >= 0.45 ? 1 : 0;
    ghostPenalty[i] = colorMismatchNormalized(args.compMean, off, args.newMean, off);
  }

  const compDist = computeBlockDistanceField(compHas, gridW, gridH);
  const newDist = computeBlockDistanceField(newHas, gridW, gridH);
  let maxCompDist = 1;
  let maxNewDist = 1;
  for (let i = 0; i < nNodes; i++) {
    if (compDist[i] > maxCompDist) maxCompDist = compDist[i];
    if (newDist[i] > maxNewDist) maxNewDist = newDist[i];
  }

  const means: number[] = [];
  for (let i = 0; i < nNodes; i++) {
    if (compHas[i] && newHas[i]) means.push(ghostPenalty[i]);
  }
  means.sort((a, b) => a - b);
  const ghostMedianDiff = means.length > 0 ? means[(means.length - 1) >> 1] : 0;
  const ghostThreshold = Math.max(ghostMedianDiff * 3, 30 / 255);
  const lightingSoftStart = Math.max(6 / 255, ghostMedianDiff * 1.15);
  const lightingSoftEnd = Math.max(lightingSoftStart + 12 / 255, ghostMedianDiff * 3.2);

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const nodeIdx = gy * gridW + gx;
      const cHas = compHas[nodeIdx];
      const nHas = newHas[nodeIdx];
      if (!cHas && !nHas) {
        dataCosts[nodeIdx * 2] = 0;
        dataCosts[nodeIdx * 2 + 1] = 0;
        continue;
      }
      if (cHas && !nHas) {
        hardConstraints[nodeIdx] = 1;
        continue;
      }
      if (!cHas && nHas) {
        hardConstraints[nodeIdx] = 2;
        continue;
      }
      const cD = compDist[nodeIdx] / maxCompDist;
      const nD = newDist[nodeIdx] / maxNewDist;
      const colorDiff = ghostPenalty[nodeIdx];
      dataCosts[nodeIdx * 2] = 0.8 * (1 - cD) + 0.2 * colorDiff;
      dataCosts[nodeIdx * 2 + 1] = 0.8 * (1 - nD) + 0.2 * colorDiff;
      if (saliencyGrid) {
        const salPenalty = saliencyGrid[nodeIdx] * 5.0;
        dataCosts[nodeIdx * 2] += salPenalty;
        dataCosts[nodeIdx * 2 + 1] += salPenalty;
      }
      const blockLeft = gx * args.blockSize;
      const blockTop = gy * args.blockSize;
      const blockRight = blockLeft + args.blockSize;
      const blockBottom = blockTop + args.blockSize;
      for (const face of expandedFaces) {
        if (
          blockLeft < face.right
          && blockRight > face.left
          && blockTop < face.bottom
          && blockBottom > face.top
        ) {
          const facePenalty = 10.0;
          if (face.imageLabel === 0) dataCosts[nodeIdx * 2 + 1] += facePenalty;
          else dataCosts[nodeIdx * 2] += facePenalty;
        }
      }
    }
  }

  const edgeWeightsH = new Float32Array(Math.max(0, (gridW - 1) * gridH));
  const edgeWeightsV = new Float32Array(Math.max(0, gridW * (gridH - 1)));
  const getSal = (idxA: number, idxB: number): number => {
    if (!saliencyGrid) return 1;
    return (saliencyGrid[idxA] + saliencyGrid[idxB]) * 0.5;
  };

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW - 1; gx++) {
      const a = gy * gridW + gx;
      const b = a + 1;
      const offA = a * 4;
      const offB = b * 4;
      const compGrad =
        (Math.abs(args.compMean[offB] - args.compMean[offA])
        + Math.abs(args.compMean[offB + 1] - args.compMean[offA + 1])
        + Math.abs(args.compMean[offB + 2] - args.compMean[offA + 2])) / 3;
      const newGrad =
        (Math.abs(args.newMean[offB] - args.newMean[offA])
        + Math.abs(args.newMean[offB + 1] - args.newMean[offA + 1])
        + Math.abs(args.newMean[offB + 2] - args.newMean[offA + 2])) / 3;
      const crossGrad = colorMismatchNormalized(args.compMean, offB, args.newMean, offB);
      const gradientAgreement = Math.max(
        0.01,
        1
          - colorMismatchNormalized(args.compMean, offA, args.newMean, offA) * 0.5
          - colorMismatchNormalized(args.compMean, offB, args.newMean, offB) * 0.5,
      );
      const edgeStrength = Math.max(0.01, 1 - Math.max(compGrad, newGrad, crossGrad));
      const blurDiscount = 0.2 + 0.8 * getSal(a, b);
      edgeWeightsH[gy * (gridW - 1) + gx] = (0.4 * edgeStrength + 0.6 * gradientAgreement) * blurDiscount;
    }
  }
  for (let gy = 0; gy < gridH - 1; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const a = gy * gridW + gx;
      const b = a + gridW;
      const offA = a * 4;
      const offB = b * 4;
      const compGrad =
        (Math.abs(args.compMean[offB] - args.compMean[offA])
        + Math.abs(args.compMean[offB + 1] - args.compMean[offA + 1])
        + Math.abs(args.compMean[offB + 2] - args.compMean[offA + 2])) / 3;
      const newGrad =
        (Math.abs(args.newMean[offB] - args.newMean[offA])
        + Math.abs(args.newMean[offB + 1] - args.newMean[offA + 1])
        + Math.abs(args.newMean[offB + 2] - args.newMean[offA + 2])) / 3;
      const crossGrad = colorMismatchNormalized(args.compMean, offB, args.newMean, offB);
      const gradientAgreement = Math.max(
        0.01,
        1
          - colorMismatchNormalized(args.compMean, offA, args.newMean, offA) * 0.5
          - colorMismatchNormalized(args.compMean, offB, args.newMean, offB) * 0.5,
      );
      const edgeStrength = Math.max(0.01, 1 - Math.max(compGrad, newGrad, crossGrad));
      const blurDiscount = 0.2 + 0.8 * getSal(a, b);
      edgeWeightsV[gy * gridW + gx] = (0.4 * edgeStrength + 0.6 * gradientAgreement) * blurDiscount;
    }
  }

  const colorTransfer = computeColorTransferStats(args.compMean, args.compSq, args.newMean, args.newSq, gridW, gridH);
  const buildMs = performance.now() - buildStart;

  return {
    gridW,
    gridH,
    dataCosts,
    edgeWeightsH,
    edgeWeightsV,
    hardConstraints,
    ghostPenalty,
    colorTransferStats: colorTransfer.stats,
    colorTransferStatsBuffer: colorTransfer.buffer,
    ghostPenaltyBuffer: ghostPenalty,
    readbackBytes: args.readbackBytes,
    summaryMs: args.summaryMs,
    buildMs,
    backendId: args.backendId,
    resolvedBlockSize: args.blockSize,
    sampleGrid: args.sampleGrid,
    ghostMedianDiff,
    ghostThreshold,
    lightingSoftStart,
    lightingSoftEnd,
    summaryBuffers: {
      gridW,
      gridH,
      blockSize: args.blockSize,
      sampleGrid: args.sampleGrid,
      compMean: args.compMean,
      compSq: args.compSq,
      newMean: args.newMean,
      newSq: args.newSq,
    },
  };
}

export function createSeamAccelerator(gl: WebGL2RenderingContext, floatFBO: boolean): SeamAccelerator {
  const summaryProg = createProgram(gl, FULLSCREEN_VERT, BLOCK_SUMMARY_FRAG);
  const colorProg = createProgram(gl, FULLSCREEN_VERT, COLOR_TRANSFER_FRAG);
  const expandProg = createProgram(gl, FULLSCREEN_VERT, MASK_EXPAND_FRAG);
  const refineProg = createProgram(gl, FULLSCREEN_VERT, MASK_REFINE_FRAG);
  const blurProg = createProgram(gl, FULLSCREEN_VERT, BLUR_FRAG);

  const quadVAO = gl.createVertexArray()!;
  const quadVBO = gl.createBuffer()!;
  gl.bindVertexArray(quadVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(0);
  gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  let maskTexA: ManagedTexture | null = null;
  let maskTexB: ManagedTexture | null = null;
  let maskFboA: ManagedFBO | null = null;
  let maskFboB: ManagedFBO | null = null;
  let cachedMaskW = 0;
  let cachedMaskH = 0;

  function ensureMaskTargets(width: number, height: number): void {
    if (maskTexA && cachedMaskW === width && cachedMaskH === height) return;
    maskFboA?.dispose();
    maskFboB?.dispose();
    maskTexA?.dispose();
    maskTexB?.dispose();
    maskTexA = createEmptyTexture(gl, width, height);
    maskTexB = createEmptyTexture(gl, width, height);
    maskFboA = createFBO(gl, maskTexA.texture);
    maskFboB = createFBO(gl, maskTexB.texture);
    cachedMaskW = width;
    cachedMaskH = height;
  }

  function drawFullscreen(targetFbo: WebGLFramebuffer | null, width: number, height: number, program: WebGLProgram): void {
    gl.bindFramebuffer(gl.FRAMEBUFFER, targetFbo);
    gl.viewport(0, 0, width, height);
    gl.useProgram(program);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  function renderBlockSummary(
    sourceTex: WebGLTexture,
    sourceW: number,
    sourceH: number,
    gridW: number,
    gridH: number,
    blockSize: number,
    sampleGrid: number,
    mode: 0 | 1,
  ): Float32Array {
    const tex = createEmptyTexture(gl, gridW, gridH, gl.RGBA32F, gl.RGBA, gl.FLOAT);
    const fbo = createFBO(gl, tex.texture);
    gl.useProgram(summaryProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTex);
    gl.uniform1i(gl.getUniformLocation(summaryProg, 'u_texture'), 0);
    gl.uniform2f(gl.getUniformLocation(summaryProg, 'u_sourceSize'), sourceW, sourceH);
    gl.uniform1f(gl.getUniformLocation(summaryProg, 'u_blockSize'), blockSize);
    gl.uniform1i(gl.getUniformLocation(summaryProg, 'u_samples'), sampleGrid);
    gl.uniform1i(gl.getUniformLocation(summaryProg, 'u_mode'), mode);
    gl.uniform1f(gl.getUniformLocation(summaryProg, 'u_alphaThreshold'), CONTENT_ALPHA_THRESHOLD);
    drawFullscreen(fbo.fbo, gridW, gridH, summaryProg);
    const out = new Float32Array(gridW * gridH * 4);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo.fbo);
    gl.readPixels(0, 0, gridW, gridH, gl.RGBA, gl.FLOAT, out);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    fbo.dispose();
    tex.dispose();
    return out;
  }

  function buildCompactGraph(args: BuildCompactSeamGraphArgs): CompactSeamGraphBuildResult | null {
    if (!floatFBO) return null;
    const summaryStart = performance.now();
    const { gridW, gridH } = resolveCompactSeamGrid(args.width, args.height, args.blockSize);
    const sampleGrid = resolveCompactSummarySampleGrid(args.tier);
    const compMean = renderBlockSummary(args.compositeTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 0);
    const compSq = renderBlockSummary(args.compositeTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 1);
    const newMean = renderBlockSummary(args.newTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 0);
    const newSq = renderBlockSummary(args.newTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 1);
    const summaryMs = performance.now() - summaryStart;
    return buildCompactGraphFromSummaries({
      width: args.width,
      height: args.height,
      blockSize: args.blockSize,
      sampleGrid,
      compMean,
      compSq,
      newMean,
      newSq,
      faceRects: args.faceRects,
      saliencyGrid: args.saliencyGrid,
      summaryMs,
      readbackBytes: compMean.byteLength + compSq.byteLength + newMean.byteLength + newSq.byteLength,
      backendId: 'compact-webgl-grid',
    });
  }

  function summarizeTexture(args: SummarizeTextureArgs): CompactTextureSummaryResult | null {
    if (!floatFBO) return null;
    const summaryStart = performance.now();
    const { gridW, gridH } = resolveCompactSeamGrid(args.width, args.height, args.blockSize);
    const sampleGrid = resolveCompactSummarySampleGrid(args.tier);
    const mean = renderBlockSummary(args.sourceTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 0);
    const sq = renderBlockSummary(args.sourceTex, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid, 1);
    return {
      gridW,
      gridH,
      blockSize: args.blockSize,
      sampleGrid,
      mean,
      sq,
      readbackBytes: mean.byteLength + sq.byteLength,
      summaryMs: performance.now() - summaryStart,
      backendId: 'compact-webgl-grid',
    };
  }

  function renderColorAdjustedTexture(
    sourceTex: WebGLTexture,
    width: number,
    height: number,
    gain: [number, number, number],
    offset: [number, number, number],
  ): ManagedTexture {
    const outTex = createEmptyTexture(gl, width, height);
    const outFbo = createFBO(gl, outTex.texture);
    gl.useProgram(colorProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTex);
    gl.uniform1i(gl.getUniformLocation(colorProg, 'u_texture'), 0);
    gl.uniform3f(gl.getUniformLocation(colorProg, 'u_gain'), gain[0], gain[1], gain[2]);
    gl.uniform3f(gl.getUniformLocation(colorProg, 'u_offset'), offset[0], offset[1], offset[2]);
    drawFullscreen(outFbo.fbo, width, height, colorProg);
    outFbo.dispose();
    return outTex;
  }

  function copyTexture(sourceTex: WebGLTexture, width: number, height: number): ManagedTexture {
    return renderColorAdjustedTexture(sourceTex, width, height, [1, 1, 1], [0, 0, 0]);
  }

  function applyColorTransfer(sourceTex: WebGLTexture, width: number, height: number, stats: SeamColorTransferStats): ManagedTexture {
    return renderColorAdjustedTexture(sourceTex, width, height, stats.gain, stats.offset);
  }

  function blurInto(
    sourceTex: WebGLTexture,
    tempFbo: WebGLFramebuffer,
    outputFbo: WebGLFramebuffer,
    width: number,
    height: number,
    radius: number,
  ): void {
    const tempDir = [1 / Math.max(1, width), 0];
    const outDir = [0, 1 / Math.max(1, height)];
    gl.useProgram(blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, sourceTex);
    gl.uniform1i(gl.getUniformLocation(blurProg, 'u_texture'), 0);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'u_direction'), tempDir[0], tempDir[1]);
    gl.uniform1f(gl.getUniformLocation(blurProg, 'u_radius'), radius);
    drawFullscreen(tempFbo, width, height, blurProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, maskTexB!.texture);
    gl.uniform1i(gl.getUniformLocation(blurProg, 'u_texture'), 0);
    gl.uniform2f(gl.getUniformLocation(blurProg, 'u_direction'), outDir[0], outDir[1]);
    gl.uniform1f(gl.getUniformLocation(blurProg, 'u_radius'), radius);
    drawFullscreen(outputFbo, width, height, blurProg);
  }

  function buildMaskTexture(args: BuildMaskTextureArgs): ManagedTexture {
    ensureMaskTargets(args.width, args.height);
    const normalizedLabels = new Uint8Array(args.labels.length);
    for (let i = 0; i < args.labels.length; i++) {
      normalizedLabels[i] = args.labels[i] ? 255 : 0;
    }
    const labelTex = createTextureFromData(
      gl,
      args.gridW,
      args.gridH,
      gl.R8,
      gl.RED,
      gl.UNSIGNED_BYTE,
      normalizedLabels,
      { unpackAlignment: 1 },
    );
    const penaltyTex = createTextureFromData(
      gl,
      args.gridW,
      args.gridH,
      gl.R32F,
      gl.RED,
      gl.FLOAT,
      args.ghostPenalty,
    );

    gl.useProgram(expandProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, labelTex.texture);
    gl.uniform1i(gl.getUniformLocation(expandProg, 'u_labelTex'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, args.compositeTex);
    gl.uniform1i(gl.getUniformLocation(expandProg, 'u_composite'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, args.newTex);
    gl.uniform1i(gl.getUniformLocation(expandProg, 'u_newImage'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, args.keyCoverageTex ?? null);
    gl.uniform1i(gl.getUniformLocation(expandProg, 'u_keyCoverage'), 3);
    gl.uniform1f(gl.getUniformLocation(expandProg, 'u_useKey'), args.keyCoverageTex ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(expandProg, 'u_gridSize'), args.gridW, args.gridH);
    gl.uniform2f(gl.getUniformLocation(expandProg, 'u_outputSize'), args.width, args.height);
    gl.uniform1f(gl.getUniformLocation(expandProg, 'u_blockSize'), args.blockSize);
    gl.uniform1f(gl.getUniformLocation(expandProg, 'u_alphaThreshold'), CONTENT_ALPHA_THRESHOLD);
    drawFullscreen(maskFboA!.fbo, args.width, args.height, expandProg);

    const sameCameraSettings = !!args.sameCameraSettings;
    const featherRadius = sameCameraSettings
      ? Math.max(
        Math.max(6, Math.round(args.featherRadius * 0.85)),
        Math.round(Math.max(1, args.featherRadius) * 1.35),
      )
      : Math.max(1, Math.round(args.featherRadius));
    const hardThreshold = sameCameraSettings
      ? clampUnit(Math.min(1, args.ghostThreshold * 1.8))
      : clampUnit(args.ghostThreshold);
    const softStart = sameCameraSettings ? 1 : clampUnit(args.lightingSoftStart);
    const softEnd = sameCameraSettings ? 1 : clampUnit(args.lightingSoftEnd);
    blurInto(maskTexA!.texture, maskFboB!.fbo, maskFboA!.fbo, args.width, args.height, featherRadius);

    gl.useProgram(refineProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, maskTexA!.texture);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_mask'), 0);
    gl.activeTexture(gl.TEXTURE1);
    gl.bindTexture(gl.TEXTURE_2D, labelTex.texture);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_labelTex'), 1);
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, penaltyTex.texture);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_penaltyTex'), 2);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, args.compositeTex);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_composite'), 3);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, args.newTex);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_newImage'), 4);
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, args.keyCoverageTex ?? null);
    gl.uniform1i(gl.getUniformLocation(refineProg, 'u_keyCoverage'), 5);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_useKey'), args.keyCoverageTex ? 1 : 0);
    gl.uniform2f(gl.getUniformLocation(refineProg, 'u_gridSize'), args.gridW, args.gridH);
    gl.uniform2f(gl.getUniformLocation(refineProg, 'u_outputSize'), args.width, args.height);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_blockSize'), args.blockSize);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_alphaThreshold'), CONTENT_ALPHA_THRESHOLD);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_hardThreshold'), hardThreshold);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_softStart'), softStart);
    gl.uniform1f(gl.getUniformLocation(refineProg, 'u_softEnd'), softEnd);
    drawFullscreen(maskFboB!.fbo, args.width, args.height, refineProg);

    const outTex = createEmptyTexture(gl, args.width, args.height);
    const outFbo = createFBO(gl, outTex.texture);
    gl.useProgram(colorProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, maskTexB!.texture);
    gl.uniform1i(gl.getUniformLocation(colorProg, 'u_texture'), 0);
    gl.uniform3f(gl.getUniformLocation(colorProg, 'u_gain'), 1, 1, 1);
    gl.uniform3f(gl.getUniformLocation(colorProg, 'u_offset'), 0, 0, 0);
    drawFullscreen(outFbo.fbo, args.width, args.height, colorProg);
    outFbo.dispose();
    labelTex.dispose();
    penaltyTex.dispose();
    return outTex;
  }

  return {
    buildCompactGraph,
    summarizeTexture,
    applyColorTransfer,
    copyTexture,
    buildMaskTexture,
    dispose() {
      maskFboA?.dispose();
      maskFboB?.dispose();
      maskTexA?.dispose();
      maskTexB?.dispose();
      gl.deleteProgram(summaryProg);
      gl.deleteProgram(colorProg);
      gl.deleteProgram(expandProg);
      gl.deleteProgram(refineProg);
      gl.deleteProgram(blurProg);
      gl.deleteVertexArray(quadVAO);
      gl.deleteBuffer(quadVBO);
    },
  };
}
