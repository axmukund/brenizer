export type SeamPostprocessMode = 'fast' | 'standard' | 'highQuality';
export type SeamOrientation = 'vertical' | 'horizontal';
export type SeamCandidateSource = 'metadata' | 'detected' | 'hybrid';

export interface SeamFootprintBounds {
  left: number;
  top: number;
  right: number;
  bottom: number;
  confidence?: number;
}

export interface SeamCandidate {
  orientation: SeamOrientation;
  position: number;
  nominalWidth: number;
  confidence: number;
  source: SeamCandidateSource;
}

export interface SeamPostprocessMetadata {
  seams?: SeamCandidate[];
  bounds?: SeamFootprintBounds[];
}

export interface SeamPostprocessOptions {
  enabled: boolean;
  mode: SeamPostprocessMode;
  bandBaseWidth: number;
  bandScale: number;
  chromaCorrectionWeight: number;
  edgeGateStrength: number;
  maxCorrectionClamp: number;
  autoDetect?: boolean;
  debug?: boolean;
  metadata?: SeamPostprocessMetadata | null;
}

export interface AppliedSeamCorrection {
  orientation: SeamOrientation;
  position: number;
  nominalWidth: number;
  radius: number;
  confidence: number;
  source: SeamCandidateSource;
  meanDeltaY: number;
  meanDeltaCb: number;
  meanDeltaCr: number;
  consistency: number;
}

export interface SeamPostprocessDebug {
  verticalProfile: number[];
  horizontalProfile: number[];
  seams: AppliedSeamCorrection[];
}

export interface SeamPostprocessResult {
  image: Uint8ClampedArray;
  seams: AppliedSeamCorrection[];
  debug?: SeamPostprocessDebug;
}

interface ProfileConfig {
  detectionStride: number;
  sampleStride: number;
  detectionRadius: number;
  detectionInner: number;
  patchHalfSpan: number;
  smoothingRadius: number;
  peakSpacing: number;
  maxSeamsPerOrientation: number;
}

interface BiasFieldResult {
  denseY: Float32Array;
  denseCb: Float32Array;
  denseCr: Float32Array;
  meanDeltaY: number;
  meanDeltaCb: number;
  meanDeltaCr: number;
  consistency: number;
}

interface InternalCandidate extends SeamCandidate {
  score: number;
}

const WR = 0.2126;
const WG = 0.7152;
const WB = 0.0722;
const ALPHA_THRESHOLD = 8;

function clampNumber(v: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, v));
}

function smoothstep01(t: number): number {
  const x = clampNumber(t, 0, 1);
  return x * x * (3 - 2 * x);
}

function medianOf(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = values.slice().sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[mid - 1] + sorted[mid]) * 0.5
    : sorted[mid];
}

function makeCopy(data: Uint8Array | Uint8ClampedArray): Uint8ClampedArray {
  const out = new Uint8ClampedArray(data.length);
  out.set(data);
  return out;
}

function computeY(r: number, g: number, b: number): number {
  return WR * r + WG * g + WB * b;
}

function computeCb(y: number, b: number): number {
  return b - y;
}

function computeCr(y: number, r: number): number {
  return r - y;
}

function estimateEdgeMagnitude(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  x: number,
  y: number,
): number {
  const xm = Math.max(0, x - 1);
  const xp = Math.min(width - 1, x + 1);
  const ym = Math.max(0, y - 1);
  const yp = Math.min(height - 1, y + 1);
  const leftIdx = (y * width + xm) * 4;
  const rightIdx = (y * width + xp) * 4;
  const upIdx = (ym * width + x) * 4;
  const downIdx = (yp * width + x) * 4;
  const leftY = computeY(data[leftIdx], data[leftIdx + 1], data[leftIdx + 2]);
  const rightY = computeY(data[rightIdx], data[rightIdx + 1], data[rightIdx + 2]);
  const upY = computeY(data[upIdx], data[upIdx + 1], data[upIdx + 2]);
  const downY = computeY(data[downIdx], data[downIdx + 1], data[downIdx + 2]);
  const dx = Math.abs(rightY - leftY);
  const dy = Math.abs(downY - upY);
  return Math.max(dx, dy) + 0.5 * Math.min(dx, dy);
}

function writeWorkingColor(
  data: Uint8ClampedArray,
  offset: number,
  y: number,
  cb: number,
  cr: number,
): void {
  const r = y + cr;
  const b = y + cb;
  const g = (y - WR * r - WB * b) / WG;
  data[offset] = Math.round(clampNumber(r, 0, 255));
  data[offset + 1] = Math.round(clampNumber(g, 0, 255));
  data[offset + 2] = Math.round(clampNumber(b, 0, 255));
}

function computeProfileStats(profile: Float32Array): { mean: number; std: number; max: number } {
  let sum = 0;
  let count = 0;
  let max = 0;
  for (let i = 0; i < profile.length; i++) {
    const v = profile[i];
    if (v <= 0) continue;
    sum += v;
    count++;
    if (v > max) max = v;
  }
  if (count === 0) return { mean: 0, std: 0, max: 0 };
  const mean = sum / count;
  let varSum = 0;
  for (let i = 0; i < profile.length; i++) {
    const v = profile[i];
    if (v <= 0) continue;
    const d = v - mean;
    varSum += d * d;
  }
  const std = Math.sqrt(varSum / count);
  return { mean, std, max };
}

function smoothProfile(profile: Float32Array, radius: number): Float32Array {
  if (radius <= 0) return profile.slice();
  const out = new Float32Array(profile.length);
  for (let i = 0; i < profile.length; i++) {
    let sum = 0;
    let weight = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(profile.length - 1, i + radius); j++) {
      const kernel = radius + 1 - Math.abs(i - j);
      sum += profile[j] * kernel;
      weight += kernel;
    }
    out[i] = weight > 0 ? sum / weight : 0;
  }
  return out;
}

function smoothWeighted(values: Float32Array, weights: Float32Array, radius: number): Float32Array {
  if (radius <= 0) return values.slice();
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i++) {
    let sum = 0;
    let totalWeight = 0;
    for (let j = Math.max(0, i - radius); j <= Math.min(values.length - 1, i + radius); j++) {
      const kernel = radius + 1 - Math.abs(i - j);
      const weight = weights[j] * kernel;
      if (weight <= 0) continue;
      sum += values[j] * weight;
      totalWeight += weight;
    }
    out[i] = totalWeight > 0 ? sum / totalWeight : 0;
  }
  return out;
}

function resampleToDense(length: number, sampleStride: number, values: Float32Array): Float32Array {
  if (values.length === 0) return new Float32Array(length);
  const out = new Float32Array(length);
  const lastIndex = values.length - 1;
  for (let i = 0; i < length; i++) {
    const samplePos = i / Math.max(1, sampleStride);
    const lo = Math.floor(samplePos);
    const hi = Math.min(lastIndex, lo + 1);
    const t = clampNumber(samplePos - lo, 0, 1);
    const a = values[Math.min(lastIndex, lo)];
    const b = values[hi];
    out[i] = a + (b - a) * t;
  }
  return out;
}

function resolveProfileConfig(mode: SeamPostprocessMode, width: number, height: number, baseWidth: number): ProfileConfig {
  const shortEdge = Math.max(1, Math.min(width, height));
  if (mode === 'fast') {
    return {
      detectionStride: 8,
      sampleStride: 6,
      detectionRadius: clampNumber(Math.round(baseWidth * 0.9), 6, 20),
      detectionInner: clampNumber(Math.round(baseWidth * 0.2), 1, 6),
      patchHalfSpan: 1,
      smoothingRadius: 2,
      peakSpacing: clampNumber(Math.round(shortEdge * 0.06), Math.max(48, Math.round(baseWidth * 4)), 512),
      maxSeamsPerOrientation: 8,
    };
  }
  if (mode === 'highQuality') {
    return {
      detectionStride: 2,
      sampleStride: 2,
      detectionRadius: clampNumber(Math.round(baseWidth * 1.5), 8, 32),
      detectionInner: clampNumber(Math.round(baseWidth * 0.3), 1, 10),
      patchHalfSpan: 2,
      smoothingRadius: 5,
      peakSpacing: clampNumber(Math.round(shortEdge * 0.07), Math.max(64, Math.round(baseWidth * 5)), 512),
      maxSeamsPerOrientation: 12,
    };
  }
  return {
    detectionStride: 4,
    sampleStride: 4,
    detectionRadius: clampNumber(Math.round(baseWidth * 1.2), 8, 24),
    detectionInner: clampNumber(Math.round(baseWidth * 0.25), 1, 8),
    patchHalfSpan: 1,
    smoothingRadius: 3,
    peakSpacing: clampNumber(Math.round(shortEdge * 0.065), Math.max(56, Math.round(baseWidth * 4.5)), 512),
    maxSeamsPerOrientation: 10,
  };
}

function clusterCandidates(candidates: InternalCandidate[], tolerance: number): InternalCandidate[] {
  if (candidates.length === 0) return [];
  const sorted = candidates.slice().sort((a, b) => a.position - b.position);
  const clusters: InternalCandidate[] = [];
  for (const candidate of sorted) {
    const last = clusters[clusters.length - 1];
    if (!last || Math.abs(candidate.position - last.position) > tolerance) {
      clusters.push({ ...candidate });
      continue;
    }
    const lastWeight = Math.max(1e-3, last.score);
    const nextWeight = Math.max(1e-3, candidate.score);
    const totalWeight = lastWeight + nextWeight;
    last.position = (last.position * lastWeight + candidate.position * nextWeight) / totalWeight;
    last.nominalWidth = (last.nominalWidth * lastWeight + candidate.nominalWidth * nextWeight) / totalWeight;
    last.confidence = Math.max(last.confidence, candidate.confidence);
    last.score = Math.max(last.score, candidate.score);
    last.source = last.source === candidate.source ? last.source : 'hybrid';
  }
  return clusters;
}

function getCandidateBiasThreshold(candidate: InternalCandidate): number {
  if (candidate.source === 'metadata') return 0.35;
  if (candidate.source === 'hybrid') return 0.5;
  return 0.75;
}

function getCandidateStrengthFloor(candidate: InternalCandidate): number {
  if (candidate.source === 'metadata') return 0.3;
  if (candidate.source === 'hybrid') return 0.24;
  return 0.2;
}

function getCandidateShoulderFloor(candidate: InternalCandidate): number {
  if (candidate.source === 'metadata') return 0.06;
  if (candidate.source === 'hybrid') return 0.04;
  return 0.02;
}

function resolveRadiusLimit(mode: SeamPostprocessMode, width: number, height: number): number {
  const shortEdge = Math.max(1, Math.min(width, height));
  const modeLimit = mode === 'highQuality' ? 224 : mode === 'standard' ? 192 : 144;
  return clampNumber(Math.round(shortEdge * 0.16), 96, modeLimit);
}

function computeCoreRadius(
  candidate: InternalCandidate,
  options: SeamPostprocessOptions,
  width: number,
  height: number,
): number {
  const sourceBoost = candidate.source === 'metadata' ? 1.25 : candidate.source === 'hybrid' ? 1.12 : 1;
  const scaleBoost = 1 + options.bandScale * (0.75 + candidate.confidence * 0.5);
  const rawRadius = Math.max(candidate.nominalWidth, options.bandBaseWidth) * scaleBoost * sourceBoost;
  return clampNumber(Math.round(rawRadius), 8, resolveRadiusLimit(options.mode, width, height));
}

function computeBandRadius(
  coreRadius: number,
  candidate: InternalCandidate,
  options: SeamPostprocessOptions,
  width: number,
  height: number,
): number {
  const shoulderBoost = candidate.source === 'metadata' ? 1.35 : candidate.source === 'hybrid' ? 1.22 : 1.12;
  return clampNumber(
    Math.round(coreRadius * shoulderBoost),
    coreRadius,
    resolveRadiusLimit(options.mode, width, height),
  );
}

function computeCorrectionStrength(candidate: InternalCandidate, consistency: number): number {
  return clampNumber(
    Math.max(getCandidateStrengthFloor(candidate), candidate.confidence * consistency),
    0.2,
    1,
  );
}

function computeEdgeGateTau(candidate: InternalCandidate, options: SeamPostprocessOptions): number {
  const sourceBoost = candidate.source === 'metadata' ? 1.08 : candidate.source === 'hybrid' ? 1.04 : 1;
  return clampNumber(options.edgeGateStrength * sourceBoost, 4, 128);
}

function computeSeamBlend(distance: number, coreRadius: number, bandRadius: number, candidate: InternalCandidate): number {
  const local = 1 - smoothstep01(distance / Math.max(1, coreRadius));
  const shoulder = getCandidateShoulderFloor(candidate) * (1 - smoothstep01(distance / Math.max(coreRadius + 1, bandRadius)));
  return clampNumber(Math.max(local, shoulder), 0, 1);
}

export function buildSeamMetadataFromBounds(
  bounds: SeamFootprintBounds[],
  width: number,
  height: number,
  nominalWidth = 16,
): SeamPostprocessMetadata {
  if (bounds.length < 2) return { seams: [] };
  const rawCandidates: InternalCandidate[] = [];
  const minOverlapX = Math.max(8, width * 0.01);
  const minOverlapY = Math.max(8, height * 0.01);

  for (let i = 0; i < bounds.length; i++) {
    const a = bounds[i];
    const aWidth = Math.max(1, a.right - a.left);
    const aHeight = Math.max(1, a.bottom - a.top);
    const aCenterX = (a.left + a.right) * 0.5;
    const aCenterY = (a.top + a.bottom) * 0.5;
    for (let j = i + 1; j < bounds.length; j++) {
      const b = bounds[j];
      const bWidth = Math.max(1, b.right - b.left);
      const bHeight = Math.max(1, b.bottom - b.top);
      const bCenterX = (b.left + b.right) * 0.5;
      const bCenterY = (b.top + b.bottom) * 0.5;
      const overlapX = Math.min(a.right, b.right) - Math.max(a.left, b.left);
      const overlapY = Math.min(a.bottom, b.bottom) - Math.max(a.top, b.top);
      if (overlapX <= 0 || overlapY <= 0) continue;

      const dx = Math.abs(aCenterX - bCenterX);
      const dy = Math.abs(aCenterY - bCenterY);
      const overlapXRatio = overlapX / Math.max(1, Math.min(aWidth, bWidth));
      const overlapYRatio = overlapY / Math.max(1, Math.min(aHeight, bHeight));
      const pairConfidence = Math.max(0.2, Math.min(a.confidence ?? 0.75, b.confidence ?? 0.75));

      if (dx >= dy * 1.1 && overlapX >= minOverlapX) {
        rawCandidates.push({
          orientation: 'vertical',
          position: Math.max(a.left, b.left) + overlapX * 0.5,
          nominalWidth: clampNumber(Math.max(nominalWidth, overlapX * 0.7), nominalWidth * 0.9, 192),
          confidence: clampNumber(pairConfidence * (0.35 + overlapYRatio * 0.65), 0.15, 1),
          source: 'metadata',
          score: clampNumber(0.3 + overlapYRatio * 0.7, 0.3, 1),
        });
      }

      if (dy >= dx * 1.1 && overlapY >= minOverlapY) {
        rawCandidates.push({
          orientation: 'horizontal',
          position: Math.max(a.top, b.top) + overlapY * 0.5,
          nominalWidth: clampNumber(Math.max(nominalWidth, overlapY * 0.7), nominalWidth * 0.9, 192),
          confidence: clampNumber(pairConfidence * (0.35 + overlapXRatio * 0.65), 0.15, 1),
          source: 'metadata',
          score: clampNumber(0.3 + overlapXRatio * 0.7, 0.3, 1),
        });
      }
    }
  }

  const tolerance = Math.max(6, nominalWidth * 1.5);
  const seams = clusterCandidates(
    rawCandidates.filter((candidate) => candidate.confidence >= 0.15),
    tolerance,
  ).map(({ score: _score, ...candidate }) => candidate);

  return { seams };
}

function buildVerticalProfile(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  config: ProfileConfig,
  chromaWeight: number,
): Float32Array {
  const profile = new Float32Array(width);
  const counts = new Uint16Array(width);
  const yRow = new Float32Array(width);
  const cbRow = new Float32Array(width);
  const crRow = new Float32Array(width);
  const prefixY = new Float32Array(width + 1);
  const prefixCb = new Float32Array(width + 1);
  const prefixCr = new Float32Array(width + 1);
  const prefixCount = new Uint16Array(width + 1);
  const radius = config.detectionRadius;
  const inner = config.detectionInner;
  const minPixels = Math.max(2, Math.round((radius - inner + 1) * 0.25));

  for (let y = 0; y < height; y += config.detectionStride) {
    prefixY[0] = 0;
    prefixCb[0] = 0;
    prefixCr[0] = 0;
    prefixCount[0] = 0;
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      let lum = 0;
      let cb = 0;
      let cr = 0;
      let count = 0;
      if (data[idx + 3] > ALPHA_THRESHOLD) {
        lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
        cb = computeCb(lum, data[idx + 2]);
        cr = computeCr(lum, data[idx]);
        count = 1;
      }
      yRow[x] = lum;
      cbRow[x] = cb;
      crRow[x] = cr;
      prefixY[x + 1] = prefixY[x] + lum;
      prefixCb[x + 1] = prefixCb[x] + cb;
      prefixCr[x + 1] = prefixCr[x] + cr;
      prefixCount[x + 1] = prefixCount[x] + count;
    }

    for (let x = radius + inner; x < width - radius - inner; x++) {
      const leftStart = Math.max(0, x - radius);
      const leftEnd = Math.max(leftStart + 1, x - inner + 1);
      const rightStart = Math.min(width - 1, x + inner);
      const rightEnd = Math.min(width, x + radius + 1);
      const leftCount = prefixCount[leftEnd] - prefixCount[leftStart];
      const rightCount = prefixCount[rightEnd] - prefixCount[rightStart];
      if (leftCount < minPixels || rightCount < minPixels) continue;
      const leftY = (prefixY[leftEnd] - prefixY[leftStart]) / leftCount;
      const rightY = (prefixY[rightEnd] - prefixY[rightStart]) / rightCount;
      const leftCb = (prefixCb[leftEnd] - prefixCb[leftStart]) / leftCount;
      const rightCb = (prefixCb[rightEnd] - prefixCb[rightStart]) / rightCount;
      const leftCr = (prefixCr[leftEnd] - prefixCr[leftStart]) / leftCount;
      const rightCr = (prefixCr[rightEnd] - prefixCr[rightStart]) / rightCount;
      profile[x] += Math.abs(rightY - leftY) + chromaWeight * (Math.abs(rightCb - leftCb) + Math.abs(rightCr - leftCr));
      counts[x]++;
    }
  }

  for (let x = 0; x < width; x++) {
    if (counts[x] > 0) profile[x] /= counts[x];
  }
  return smoothProfile(profile, clampNumber(Math.round(config.detectionRadius * 0.35), 1, 18));
}

function buildHorizontalProfile(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  config: ProfileConfig,
  chromaWeight: number,
): Float32Array {
  const profile = new Float32Array(height);
  const counts = new Uint16Array(height);
  const yCol = new Float32Array(height);
  const cbCol = new Float32Array(height);
  const crCol = new Float32Array(height);
  const prefixY = new Float32Array(height + 1);
  const prefixCb = new Float32Array(height + 1);
  const prefixCr = new Float32Array(height + 1);
  const prefixCount = new Uint16Array(height + 1);
  const radius = config.detectionRadius;
  const inner = config.detectionInner;
  const minPixels = Math.max(2, Math.round((radius - inner + 1) * 0.25));

  for (let x = 0; x < width; x += config.detectionStride) {
    prefixY[0] = 0;
    prefixCb[0] = 0;
    prefixCr[0] = 0;
    prefixCount[0] = 0;
    for (let y = 0; y < height; y++) {
      const idx = (y * width + x) * 4;
      let lum = 0;
      let cb = 0;
      let cr = 0;
      let count = 0;
      if (data[idx + 3] > ALPHA_THRESHOLD) {
        lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
        cb = computeCb(lum, data[idx + 2]);
        cr = computeCr(lum, data[idx]);
        count = 1;
      }
      yCol[y] = lum;
      cbCol[y] = cb;
      crCol[y] = cr;
      prefixY[y + 1] = prefixY[y] + lum;
      prefixCb[y + 1] = prefixCb[y] + cb;
      prefixCr[y + 1] = prefixCr[y] + cr;
      prefixCount[y + 1] = prefixCount[y] + count;
    }

    for (let y = radius + inner; y < height - radius - inner; y++) {
      const topStart = Math.max(0, y - radius);
      const topEnd = Math.max(topStart + 1, y - inner + 1);
      const bottomStart = Math.min(height - 1, y + inner);
      const bottomEnd = Math.min(height, y + radius + 1);
      const topCount = prefixCount[topEnd] - prefixCount[topStart];
      const bottomCount = prefixCount[bottomEnd] - prefixCount[bottomStart];
      if (topCount < minPixels || bottomCount < minPixels) continue;
      const topY = (prefixY[topEnd] - prefixY[topStart]) / topCount;
      const bottomY = (prefixY[bottomEnd] - prefixY[bottomStart]) / bottomCount;
      const topCb = (prefixCb[topEnd] - prefixCb[topStart]) / topCount;
      const bottomCb = (prefixCb[bottomEnd] - prefixCb[bottomStart]) / bottomCount;
      const topCr = (prefixCr[topEnd] - prefixCr[topStart]) / topCount;
      const bottomCr = (prefixCr[bottomEnd] - prefixCr[bottomStart]) / bottomCount;
      profile[y] += Math.abs(bottomY - topY) + chromaWeight * (Math.abs(bottomCb - topCb) + Math.abs(bottomCr - topCr));
      counts[y]++;
    }
  }

  for (let y = 0; y < height; y++) {
    if (counts[y] > 0) profile[y] /= counts[y];
  }
  return smoothProfile(profile, clampNumber(Math.round(config.detectionRadius * 0.35), 1, 18));
}

function detectPeaks(
  profile: Float32Array,
  orientation: SeamOrientation,
  nominalWidth: number,
  config: ProfileConfig,
): InternalCandidate[] {
  const stats = computeProfileStats(profile);
  if (stats.max <= 1e-3) return [];
  const threshold = Math.max(stats.mean + stats.std * 0.85, stats.max * 0.28);
  const peaks: InternalCandidate[] = [];
  for (let i = 1; i < profile.length - 1; i++) {
    const value = profile[i];
    if (value < threshold) continue;
    if (value < profile[i - 1] || value < profile[i + 1]) continue;
    const confidence = clampNumber((value - threshold) / Math.max(1e-3, stats.max - threshold), 0.15, 1);
    peaks.push({
      orientation,
      position: i,
      nominalWidth,
      confidence,
      source: 'detected',
      score: value,
    });
  }
  peaks.sort((a, b) => b.score - a.score);
  const accepted: InternalCandidate[] = [];
  for (const peak of peaks) {
    if (accepted.some((candidate) => Math.abs(candidate.position - peak.position) < config.peakSpacing)) continue;
    accepted.push(peak);
    if (accepted.length >= config.maxSeamsPerOrientation) break;
  }
  return accepted.sort((a, b) => a.position - b.position);
}

function refineMetadataCandidates(
  profile: Float32Array,
  candidates: SeamCandidate[],
  config: ProfileConfig,
): InternalCandidate[] {
  if (candidates.length === 0) return [];
  const stats = computeProfileStats(profile);
  const maxScore = Math.max(1e-3, stats.max);
  const refined: InternalCandidate[] = [];
  for (const candidate of candidates) {
    const searchRadius = Math.max(4, Math.round(candidate.nominalWidth * 2));
    const center = clampNumber(Math.round(candidate.position), 0, profile.length - 1);
    let bestPos = center;
    let bestScore = profile[center];
    for (let pos = Math.max(0, center - searchRadius); pos <= Math.min(profile.length - 1, center + searchRadius); pos++) {
      if (profile[pos] > bestScore) {
        bestScore = profile[pos];
        bestPos = pos;
      }
    }
    const profileConfidence = clampNumber(bestScore / maxScore, 0, 1);
    refined.push({
      ...candidate,
      position: bestPos,
      confidence: clampNumber(Math.max(candidate.confidence, 0.35 + profileConfidence * 0.65), 0.15, 1),
      source: bestScore > stats.mean ? 'hybrid' : 'metadata',
      score: Math.max(0.1, bestScore, candidate.confidence),
    });
  }
  return clusterCandidates(refined, Math.max(4, config.peakSpacing * 0.35));
}

function resolveCandidates(
  orientation: SeamOrientation,
  profile: Float32Array,
  nominalWidth: number,
  metadataSeams: SeamCandidate[],
  config: ProfileConfig,
  autoDetect: boolean,
): InternalCandidate[] {
  const refinedMetadata = refineMetadataCandidates(
    profile,
    metadataSeams.filter((candidate) => candidate.orientation === orientation),
    config,
  );
  const detected = autoDetect ? detectPeaks(profile, orientation, nominalWidth, config) : [];
  const combined = [...refinedMetadata];

  for (const detectedCandidate of detected) {
    const conflict = combined.find(
      (candidate) => Math.abs(candidate.position - detectedCandidate.position) < Math.max(8, config.peakSpacing * 0.5),
    );
    if (conflict) {
      if (detectedCandidate.score > conflict.score) {
        conflict.position = detectedCandidate.position;
        conflict.score = detectedCandidate.score;
      }
      conflict.confidence = Math.max(conflict.confidence, detectedCandidate.confidence);
      conflict.source = conflict.source === 'metadata' ? 'hybrid' : conflict.source;
      continue;
    }
    combined.push(detectedCandidate);
  }

  return clusterCandidates(combined, Math.max(6, config.peakSpacing * 0.3))
    .sort((a, b) => b.confidence - a.confidence)
    .slice(0, config.maxSeamsPerOrientation)
    .sort((a, b) => a.position - b.position);
}

function measureVerticalSampleBias(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  seamPos: number,
  radius: number,
  innerMargin: number,
  sampleY: number,
  patchHalfSpan: number,
): { deltaY: number; deltaCb: number; deltaCr: number; weight: number } | null {
  const leftStart = Math.max(0, Math.floor(seamPos - radius));
  const leftEnd = Math.max(leftStart + 1, Math.floor(seamPos - innerMargin));
  const rightStart = Math.min(width - 1, Math.ceil(seamPos + innerMargin));
  const rightEnd = Math.min(width, Math.ceil(seamPos + radius));
  if (leftEnd - leftStart < 3 || rightEnd - rightStart < 3) return null;
  const rowStart = Math.max(0, sampleY - patchHalfSpan);
  const rowEnd = Math.min(height - 1, sampleY + patchHalfSpan);
  const leftSpan = leftEnd - leftStart;
  const rightSpan = rightEnd - rightStart;
  const leftYSum = [0, 0, 0];
  const leftCbSum = [0, 0, 0];
  const leftCrSum = [0, 0, 0];
  const leftCount = [0, 0, 0];
  const rightYSum = [0, 0, 0];
  const rightCbSum = [0, 0, 0];
  const rightCrSum = [0, 0, 0];
  const rightCount = [0, 0, 0];

  for (let y = rowStart; y <= rowEnd; y++) {
    for (let x = leftStart; x < leftEnd; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const bin = Math.min(2, Math.floor(((x - leftStart) * 3) / Math.max(1, leftSpan)));
      const lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
      leftYSum[bin] += lum;
      leftCbSum[bin] += computeCb(lum, data[idx + 2]);
      leftCrSum[bin] += computeCr(lum, data[idx]);
      leftCount[bin]++;
    }
    for (let x = rightStart; x < rightEnd; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const bin = Math.min(2, Math.floor(((x - rightStart) * 3) / Math.max(1, rightSpan)));
      const lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
      rightYSum[bin] += lum;
      rightCbSum[bin] += computeCb(lum, data[idx + 2]);
      rightCrSum[bin] += computeCr(lum, data[idx]);
      rightCount[bin]++;
    }
  }

  const diffY: number[] = [];
  const diffCb: number[] = [];
  const diffCr: number[] = [];
  let validBins = 0;
  for (let bin = 0; bin < 3; bin++) {
    if (leftCount[bin] < 2 || rightCount[bin] < 2) continue;
    diffY.push(rightYSum[bin] / rightCount[bin] - leftYSum[bin] / leftCount[bin]);
    diffCb.push(rightCbSum[bin] / rightCount[bin] - leftCbSum[bin] / leftCount[bin]);
    diffCr.push(rightCrSum[bin] / rightCount[bin] - leftCrSum[bin] / leftCount[bin]);
    validBins++;
  }
  if (validBins === 0) return null;
  return {
    deltaY: medianOf(diffY),
    deltaCb: medianOf(diffCb),
    deltaCr: medianOf(diffCr),
    weight: validBins / 3,
  };
}

function measureHorizontalSampleBias(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  seamPos: number,
  radius: number,
  innerMargin: number,
  sampleX: number,
  patchHalfSpan: number,
): { deltaY: number; deltaCb: number; deltaCr: number; weight: number } | null {
  const topStart = Math.max(0, Math.floor(seamPos - radius));
  const topEnd = Math.max(topStart + 1, Math.floor(seamPos - innerMargin));
  const bottomStart = Math.min(height - 1, Math.ceil(seamPos + innerMargin));
  const bottomEnd = Math.min(height, Math.ceil(seamPos + radius));
  if (topEnd - topStart < 3 || bottomEnd - bottomStart < 3) return null;
  const colStart = Math.max(0, sampleX - patchHalfSpan);
  const colEnd = Math.min(width - 1, sampleX + patchHalfSpan);
  const topSpan = topEnd - topStart;
  const bottomSpan = bottomEnd - bottomStart;
  const topYSum = [0, 0, 0];
  const topCbSum = [0, 0, 0];
  const topCrSum = [0, 0, 0];
  const topCount = [0, 0, 0];
  const bottomYSum = [0, 0, 0];
  const bottomCbSum = [0, 0, 0];
  const bottomCrSum = [0, 0, 0];
  const bottomCount = [0, 0, 0];

  for (let x = colStart; x <= colEnd; x++) {
    for (let y = topStart; y < topEnd; y++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const bin = Math.min(2, Math.floor(((y - topStart) * 3) / Math.max(1, topSpan)));
      const lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
      topYSum[bin] += lum;
      topCbSum[bin] += computeCb(lum, data[idx + 2]);
      topCrSum[bin] += computeCr(lum, data[idx]);
      topCount[bin]++;
    }
    for (let y = bottomStart; y < bottomEnd; y++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const bin = Math.min(2, Math.floor(((y - bottomStart) * 3) / Math.max(1, bottomSpan)));
      const lum = computeY(data[idx], data[idx + 1], data[idx + 2]);
      bottomYSum[bin] += lum;
      bottomCbSum[bin] += computeCb(lum, data[idx + 2]);
      bottomCrSum[bin] += computeCr(lum, data[idx]);
      bottomCount[bin]++;
    }
  }

  const diffY: number[] = [];
  const diffCb: number[] = [];
  const diffCr: number[] = [];
  let validBins = 0;
  for (let bin = 0; bin < 3; bin++) {
    if (topCount[bin] < 2 || bottomCount[bin] < 2) continue;
    diffY.push(bottomYSum[bin] / bottomCount[bin] - topYSum[bin] / topCount[bin]);
    diffCb.push(bottomCbSum[bin] / bottomCount[bin] - topCbSum[bin] / topCount[bin]);
    diffCr.push(bottomCrSum[bin] / bottomCount[bin] - topCrSum[bin] / topCount[bin]);
    validBins++;
  }
  if (validBins === 0) return null;
  return {
    deltaY: medianOf(diffY),
    deltaCb: medianOf(diffCb),
    deltaCr: medianOf(diffCr),
    weight: validBins / 3,
  };
}

function summariseConsistency(values: Float32Array, weights: Float32Array): { meanAbs: number; std: number } {
  let weightedAbs = 0;
  let totalWeight = 0;
  let weightedMean = 0;
  for (let i = 0; i < values.length; i++) {
    const weight = weights[i];
    if (weight <= 0) continue;
    weightedAbs += Math.abs(values[i]) * weight;
    weightedMean += values[i] * weight;
    totalWeight += weight;
  }
  if (totalWeight <= 0) return { meanAbs: 0, std: 0 };
  const mean = weightedMean / totalWeight;
  let varSum = 0;
  for (let i = 0; i < values.length; i++) {
    const weight = weights[i];
    if (weight <= 0) continue;
    const d = values[i] - mean;
    varSum += d * d * weight;
  }
  return {
    meanAbs: weightedAbs / totalWeight,
    std: Math.sqrt(varSum / totalWeight),
  };
}

function buildVerticalBiasField(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  candidate: InternalCandidate,
  radius: number,
  config: ProfileConfig,
): BiasFieldResult | null {
  const sampleCount = Math.ceil(height / config.sampleStride);
  const rawY = new Float32Array(sampleCount);
  const rawCb = new Float32Array(sampleCount);
  const rawCr = new Float32Array(sampleCount);
  const weights = new Float32Array(sampleCount);
  const innerMargin = clampNumber(Math.round(radius * 0.18), 1, Math.max(2, radius - 2));

  for (let i = 0; i < sampleCount; i++) {
    const y = Math.min(height - 1, i * config.sampleStride);
    const sample = measureVerticalSampleBias(
      data,
      width,
      height,
      candidate.position,
      radius,
      innerMargin,
      y,
      config.patchHalfSpan,
    );
    if (!sample) continue;
    rawY[i] = sample.deltaY;
    rawCb[i] = sample.deltaCb;
    rawCr[i] = sample.deltaCr;
    weights[i] = sample.weight;
  }

  const smoothY = smoothWeighted(rawY, weights, config.smoothingRadius);
  const smoothCb = smoothWeighted(rawCb, weights, config.smoothingRadius);
  const smoothCr = smoothWeighted(rawCr, weights, config.smoothingRadius);
  const stats = summariseConsistency(smoothY, weights);
  if (stats.meanAbs < getCandidateBiasThreshold(candidate)) return null;
  const consistency = clampNumber(1 - stats.std / Math.max(2, stats.meanAbs * 2), 0.25, 1);
  const denseY = resampleToDense(height, config.sampleStride, smoothY);
  const denseCb = resampleToDense(height, config.sampleStride, smoothCb);
  const denseCr = resampleToDense(height, config.sampleStride, smoothCr);
  return {
    denseY,
    denseCb,
    denseCr,
    meanDeltaY: stats.meanAbs,
    meanDeltaCb: summariseConsistency(smoothCb, weights).meanAbs,
    meanDeltaCr: summariseConsistency(smoothCr, weights).meanAbs,
    consistency,
  };
}

function buildHorizontalBiasField(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  candidate: InternalCandidate,
  radius: number,
  config: ProfileConfig,
): BiasFieldResult | null {
  const sampleCount = Math.ceil(width / config.sampleStride);
  const rawY = new Float32Array(sampleCount);
  const rawCb = new Float32Array(sampleCount);
  const rawCr = new Float32Array(sampleCount);
  const weights = new Float32Array(sampleCount);
  const innerMargin = clampNumber(Math.round(radius * 0.18), 1, Math.max(2, radius - 2));

  for (let i = 0; i < sampleCount; i++) {
    const x = Math.min(width - 1, i * config.sampleStride);
    const sample = measureHorizontalSampleBias(
      data,
      width,
      height,
      candidate.position,
      radius,
      innerMargin,
      x,
      config.patchHalfSpan,
    );
    if (!sample) continue;
    rawY[i] = sample.deltaY;
    rawCb[i] = sample.deltaCb;
    rawCr[i] = sample.deltaCr;
    weights[i] = sample.weight;
  }

  const smoothY = smoothWeighted(rawY, weights, config.smoothingRadius);
  const smoothCb = smoothWeighted(rawCb, weights, config.smoothingRadius);
  const smoothCr = smoothWeighted(rawCr, weights, config.smoothingRadius);
  const stats = summariseConsistency(smoothY, weights);
  if (stats.meanAbs < getCandidateBiasThreshold(candidate)) return null;
  const consistency = clampNumber(1 - stats.std / Math.max(2, stats.meanAbs * 2), 0.25, 1);
  const denseY = resampleToDense(width, config.sampleStride, smoothY);
  const denseCb = resampleToDense(width, config.sampleStride, smoothCb);
  const denseCr = resampleToDense(width, config.sampleStride, smoothCr);
  return {
    denseY,
    denseCb,
    denseCr,
    meanDeltaY: stats.meanAbs,
    meanDeltaCb: summariseConsistency(smoothCb, weights).meanAbs,
    meanDeltaCr: summariseConsistency(smoothCr, weights).meanAbs,
    consistency,
  };
}

function applyVerticalCorrection(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  candidate: InternalCandidate,
  options: SeamPostprocessOptions,
  config: ProfileConfig,
): AppliedSeamCorrection | null {
  const coreRadius = computeCoreRadius(candidate, options, width, height);
  const bandRadius = computeBandRadius(coreRadius, candidate, options, width, height);
  const bias = buildVerticalBiasField(data, width, height, candidate, coreRadius, config);
  if (!bias) return null;
  const strength = computeCorrectionStrength(candidate, bias.consistency);
  const maxClamp = clampNumber(options.maxCorrectionClamp, 1, 48);
  const edgeTau = computeEdgeGateTau(candidate, options);
  const chromaWeight = clampNumber(options.chromaCorrectionWeight, 0, 1);
  const startX = Math.max(0, Math.floor(candidate.position - bandRadius));
  const endX = Math.min(width - 1, Math.ceil(candidate.position + bandRadius));

  for (let y = 0; y < height; y++) {
    const rowDeltaY = clampNumber(bias.denseY[y], -maxClamp * 2, maxClamp * 2);
    const rowDeltaCb = clampNumber(bias.denseCb[y] * chromaWeight, -maxClamp * 2, maxClamp * 2);
    const rowDeltaCr = clampNumber(bias.denseCr[y] * chromaWeight, -maxClamp * 2, maxClamp * 2);
    if (Math.abs(rowDeltaY) < 0.25 && Math.abs(rowDeltaCb) < 0.25 && Math.abs(rowDeltaCr) < 0.25) continue;
    for (let x = startX; x <= endX; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const dist = Math.abs((x + 0.5) - candidate.position);
      if (dist > bandRadius) continue;
      const proximity = computeSeamBlend(dist, coreRadius, bandRadius, candidate);
      const side = (x + 0.5) < candidate.position ? 1 : -1;
      const edge = estimateEdgeMagnitude(data, width, height, x, y);
      const gate = 1 / (1 + Math.pow(edge / edgeTau, 4));
      const blend = strength * proximity * gate;
      if (blend <= 0.02) continue;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = computeY(r, g, b);
      const cb = computeCb(lum, b);
      const cr = computeCr(lum, r);
      const nextY = lum + clampNumber(side * 0.56 * rowDeltaY * blend, -maxClamp, maxClamp);
      const nextCb = cb + clampNumber(side * 0.56 * rowDeltaCb * blend, -maxClamp, maxClamp);
      const nextCr = cr + clampNumber(side * 0.56 * rowDeltaCr * blend, -maxClamp, maxClamp);
      writeWorkingColor(data, idx, nextY, nextCb, nextCr);
    }
  }

  return {
    orientation: 'vertical',
    position: candidate.position,
    nominalWidth: candidate.nominalWidth,
    radius: bandRadius,
    confidence: candidate.confidence,
    source: candidate.source,
    meanDeltaY: bias.meanDeltaY,
    meanDeltaCb: bias.meanDeltaCb,
    meanDeltaCr: bias.meanDeltaCr,
    consistency: bias.consistency,
  };
}

function applyHorizontalCorrection(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  candidate: InternalCandidate,
  options: SeamPostprocessOptions,
  config: ProfileConfig,
): AppliedSeamCorrection | null {
  const coreRadius = computeCoreRadius(candidate, options, width, height);
  const bandRadius = computeBandRadius(coreRadius, candidate, options, width, height);
  const bias = buildHorizontalBiasField(data, width, height, candidate, coreRadius, config);
  if (!bias) return null;
  const strength = computeCorrectionStrength(candidate, bias.consistency);
  const maxClamp = clampNumber(options.maxCorrectionClamp, 1, 48);
  const edgeTau = computeEdgeGateTau(candidate, options);
  const chromaWeight = clampNumber(options.chromaCorrectionWeight, 0, 1);
  const startY = Math.max(0, Math.floor(candidate.position - bandRadius));
  const endY = Math.min(height - 1, Math.ceil(candidate.position + bandRadius));

  for (let x = 0; x < width; x++) {
    const colDeltaY = clampNumber(bias.denseY[x], -maxClamp * 2, maxClamp * 2);
    const colDeltaCb = clampNumber(bias.denseCb[x] * chromaWeight, -maxClamp * 2, maxClamp * 2);
    const colDeltaCr = clampNumber(bias.denseCr[x] * chromaWeight, -maxClamp * 2, maxClamp * 2);
    if (Math.abs(colDeltaY) < 0.25 && Math.abs(colDeltaCb) < 0.25 && Math.abs(colDeltaCr) < 0.25) continue;
    for (let y = startY; y <= endY; y++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      const dist = Math.abs((y + 0.5) - candidate.position);
      if (dist > bandRadius) continue;
      const proximity = computeSeamBlend(dist, coreRadius, bandRadius, candidate);
      const side = (y + 0.5) < candidate.position ? 1 : -1;
      const edge = estimateEdgeMagnitude(data, width, height, x, y);
      const gate = 1 / (1 + Math.pow(edge / edgeTau, 4));
      const blend = strength * proximity * gate;
      if (blend <= 0.02) continue;
      const r = data[idx];
      const g = data[idx + 1];
      const b = data[idx + 2];
      const lum = computeY(r, g, b);
      const cb = computeCb(lum, b);
      const cr = computeCr(lum, r);
      const nextY = lum + clampNumber(side * 0.56 * colDeltaY * blend, -maxClamp, maxClamp);
      const nextCb = cb + clampNumber(side * 0.56 * colDeltaCb * blend, -maxClamp, maxClamp);
      const nextCr = cr + clampNumber(side * 0.56 * colDeltaCr * blend, -maxClamp, maxClamp);
      writeWorkingColor(data, idx, nextY, nextCb, nextCr);
    }
  }

  return {
    orientation: 'horizontal',
    position: candidate.position,
    nominalWidth: candidate.nominalWidth,
    radius: bandRadius,
    confidence: candidate.confidence,
    source: candidate.source,
    meanDeltaY: bias.meanDeltaY,
    meanDeltaCb: bias.meanDeltaCb,
    meanDeltaCr: bias.meanDeltaCr,
    consistency: bias.consistency,
  };
}

export function applySeamPostprocess(
  image: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  options: SeamPostprocessOptions,
): SeamPostprocessResult {
  const out = makeCopy(image);
  if (!options.enabled || width < 8 || height < 8) {
    return { image: out, seams: [] };
  }

  const baseWidth = clampNumber(options.bandBaseWidth, 4, 64);
  const metadata = options.metadata ?? null;
  const metadataCandidates = metadata?.seams ?? [];
  const derivedMetadata = metadata?.bounds?.length
    ? buildSeamMetadataFromBounds(metadata.bounds, width, height, baseWidth).seams ?? []
    : [];
  const allMetadata = [...metadataCandidates, ...derivedMetadata];
  const config = resolveProfileConfig(options.mode, width, height, baseWidth);
  const autoDetect = options.autoDetect !== false;
  const verticalProfile = buildVerticalProfile(out, width, height, config, clampNumber(options.chromaCorrectionWeight, 0, 1) * 0.75);
  const horizontalProfile = buildHorizontalProfile(out, width, height, config, clampNumber(options.chromaCorrectionWeight, 0, 1) * 0.75);
  const verticalCandidates = resolveCandidates('vertical', verticalProfile, baseWidth, allMetadata, config, autoDetect);
  const horizontalCandidates = resolveCandidates('horizontal', horizontalProfile, baseWidth, allMetadata, config, autoDetect);
  const ordered = [...verticalCandidates, ...horizontalCandidates].sort((a, b) => b.confidence - a.confidence);
  const seams: AppliedSeamCorrection[] = [];

  for (const candidate of ordered) {
    const applied = candidate.orientation === 'vertical'
      ? applyVerticalCorrection(out, width, height, candidate, options, config)
      : applyHorizontalCorrection(out, width, height, candidate, options, config);
    if (applied) seams.push(applied);
  }

  return options.debug
    ? {
      image: out,
      seams,
      debug: {
        verticalProfile: Array.from(verticalProfile),
        horizontalProfile: Array.from(horizontalProfile),
        seams,
      },
    }
    : { image: out, seams };
}

function measureSeamContrastWindow(
  data: Uint8ClampedArray,
  width: number,
  height: number,
  seamX: number,
  innerOffset: number,
  outerOffset: number,
  excludeYMin = -1,
  excludeYMax = -1,
): number {
  const leftStart = Math.max(0, seamX - outerOffset);
  const leftEnd = Math.max(leftStart + 1, seamX - innerOffset);
  const rightStart = Math.min(width - 1, seamX + innerOffset);
  const rightEnd = Math.min(width, seamX + outerOffset + 1);
  let leftSum = 0;
  let leftCount = 0;
  let rightSum = 0;
  let rightCount = 0;

  for (let y = 0; y < height; y++) {
    if (excludeYMin >= 0 && excludeYMax >= excludeYMin && y >= excludeYMin && y <= excludeYMax) continue;
    for (let x = leftStart; x < leftEnd; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      leftSum += computeY(data[idx], data[idx + 1], data[idx + 2]);
      leftCount++;
    }
    for (let x = rightStart; x < rightEnd; x++) {
      const idx = (y * width + x) * 4;
      if (data[idx + 3] <= ALPHA_THRESHOLD) continue;
      rightSum += computeY(data[idx], data[idx + 1], data[idx + 2]);
      rightCount++;
    }
  }

  if (leftCount === 0 || rightCount === 0) return 0;
  return Math.abs(rightSum / rightCount - leftSum / leftCount);
}

function measureSeamContrast(data: Uint8ClampedArray, width: number, height: number, seamX: number): number {
  return measureSeamContrastWindow(data, width, height, seamX, 2, 8);
}

function measureCrossingEdge(data: Uint8ClampedArray, width: number, height: number, sampleX: number, edgeY: number): number {
  const x0 = clampNumber(sampleX - 3, 0, width - 1);
  const x1 = clampNumber(sampleX + 3, 0, width - 1);
  let totalPeakGradient = 0;
  let samples = 0;

  for (let x = x0; x <= x1; x++) {
    let peakGradient = 0;
    for (let y = Math.max(1, edgeY - 5); y <= Math.min(height - 2, edgeY + 5); y++) {
      const upIdx = ((y - 1) * width + x) * 4;
      const downIdx = ((y + 1) * width + x) * 4;
      const upY = computeY(data[upIdx], data[upIdx + 1], data[upIdx + 2]);
      const downY = computeY(data[downIdx], data[downIdx + 1], data[downIdx + 2]);
      peakGradient = Math.max(peakGradient, Math.abs(downY - upY));
    }
    totalPeakGradient += peakGradient;
    samples++;
  }

  return samples > 0 ? totalPeakGradient / samples : 0;
}

export function runSeamPostprocessSyntheticSelfTest(): {
  beforeStep: number;
  afterStep: number;
  wideBefore: number;
  wideAfter: number;
  edgeBefore: number;
  edgeAfter: number;
  edgeRetention: number;
  seamCount: number;
} {
  const width = 160;
  const height = 96;
  const seamX = 80;
  const edgeY = 48;
  const image = new Uint8ClampedArray(width * height * 4);

  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const idx = (y * width + x) * 4;
      let base = 88 + x * 0.45 + y * 0.15 + ((x + y) % 7) * 1.1;
      if (Math.abs(y - edgeY) <= 2) base += 55;
      if (x >= seamX) base += 22;
      const r = clampNumber(base + (x >= seamX ? 5 : 0), 0, 255);
      const g = clampNumber(base, 0, 255);
      const b = clampNumber(base + (x >= seamX ? 10 : 0), 0, 255);
      image[idx] = Math.round(r);
      image[idx + 1] = Math.round(g);
      image[idx + 2] = Math.round(b);
      image[idx + 3] = 255;
    }
  }

  const beforeStep = measureSeamContrast(image, width, height, seamX);
  const wideBefore = measureSeamContrastWindow(image, width, height, seamX, 24, 56, edgeY - 8, edgeY + 8);
  const edgeBefore = measureCrossingEdge(image, width, height, seamX, edgeY);
  const result = applySeamPostprocess(image, width, height, {
    enabled: true,
    mode: 'standard',
    bandBaseWidth: 18,
    bandScale: 1.45,
    chromaCorrectionWeight: 0.55,
    edgeGateStrength: 22,
    maxCorrectionClamp: 24,
    autoDetect: true,
    metadata: {
      seams: [{
        orientation: 'vertical',
        position: seamX,
        nominalWidth: 24,
        confidence: 1,
        source: 'metadata',
      }],
    },
  });
  const afterStep = measureSeamContrast(result.image, width, height, seamX);
  const wideAfter = measureSeamContrastWindow(result.image, width, height, seamX, 24, 56, edgeY - 8, edgeY + 8);
  const edgeAfter = measureCrossingEdge(result.image, width, height, seamX, edgeY);

  return {
    beforeStep,
    afterStep,
    wideBefore,
    wideAfter,
    edgeBefore,
    edgeAfter,
    edgeRetention: edgeBefore > 0 ? edgeAfter / edgeBefore : 1,
    seamCount: result.seams.length,
  };
}
