/**
 * Composition utilities — FBO-based incremental composition with seam masking.
 * Provides:
 *  - Fullscreen blend pass (composite + new image + mask → output)
 *  - Separable Gaussian blur for feathering seam masks
 *  - Block-grid cost computation from pixel readback
 */

import { createProgram } from './programs';
import { createEmptyTexture, type ManagedTexture } from './textures';
import { createFBO, type ManagedFBO } from './framebuffers';

const CONTENT_ALPHA_THRESHOLD = 10;
const MASK_HARD_LOW = 8;
const MASK_HARD_HIGH = 247;

// ── Blend shader: mix(composite, new, mask) ──────────────

const BLEND_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const BLEND_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_composite;
uniform sampler2D u_newImage;
uniform sampler2D u_mask;      // R channel: 0 = keep composite, 1 = take new
out vec4 fragColor;
void main() {
  vec4 comp = texture(u_composite, v_uv);
  vec4 newI = texture(u_newImage, v_uv);
  float m = texture(u_mask, v_uv).r;
  // The mask already encodes boundary information:
  //  - Graph-cut path: hard constraints force m=0 outside new image
  //  - Feather-only path: mask is derived from newImage alpha
  // We do NOT multiply m by newI.a here — that double-applies the
  // boundary falloff, producing visible seam bands.
  fragColor = mix(comp, newI, m);
  fragColor.a = max(comp.a, newI.a);
}
`;

// ── Separable Gaussian blur ──────────────────────────────

const BLUR_VERT = `#version 300 es
precision highp float;
layout(location = 0) in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const BLUR_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec2 u_direction;  // (1/w, 0) or (0, 1/h)
uniform float u_radius;     // blur radius in texels
out vec4 fragColor;
void main() {
  vec4 sum = vec4(0.0);
  float totalW = 0.0;
  float sigma = u_radius * 0.5;
  float invSigma2 = 1.0 / (2.0 * sigma * sigma);
  int iRadius = int(u_radius);
  for (int i = -iRadius; i <= iRadius; i++) {
    float fi = float(i);
    float w = exp(-fi * fi * invSigma2);
    vec2 offset = u_direction * fi;
    sum += texture(u_texture, v_uv + offset) * w;
    totalW += w;
  }
  fragColor = sum / totalW;
}
`;

// ── Types ────────────────────────────────────────────────

export interface Compositor {
  /** 
   * Blend newImage into composite using seam mask.
   * Renders to outputFBO (or default FB if null).
   */
  blendWithMask(
    compositeTex: WebGLTexture,
    newImageTex: WebGLTexture,
    maskTex: WebGLTexture,
    outputFBO: WebGLFramebuffer | null,
    width: number,
    height: number,
  ): void;

  /**
   * Apply separable Gaussian blur to a texture.
   * Requires a temp texture+FBO for the two-pass blur.
   * Modifies the source in-place (pingpong between source FBO and temp FBO).
   */
  blur(
    sourceTex: WebGLTexture,
    sourceFBO: WebGLFramebuffer,
    tempTex: WebGLTexture,
    tempFBO: WebGLFramebuffer,
    width: number,
    height: number,
    radius: number,
  ): void;

  dispose(): void;
}

/**
 * Create block-grid seam costs from two RGBA pixel buffers.
 * Uses distance-from-boundary data costs so the seam goes through the
 * centre of the overlap zone rather than hugging image edges.
 * Returns { dataCosts, edgeWeights, hardConstraints } for seam-worker.
 */
export interface FaceRectComposite {
  x: number;
  y: number;
  width: number;
  height: number;
  imageLabel: 0 | 1;  // 0 = belongs to composite, 1 = belongs to new image
}

export interface AdaptiveBlendMaskOptions {
  ghostBlockSize?: number;
}

export interface AdaptiveBlendMaskResult {
  mask: Uint8Array;
  featherRadius: number;
  ghostPixels: number;
  ghostThreshold: number;
  ghostMedianDiff: number;
}

export function computeBlockCosts(
  compositePixels: Uint8Array,   // RGBA at composite resolution
  newImagePixels: Uint8Array,    // RGBA at composite resolution
  compW: number,
  compH: number,
  blockSize: number,
  _depthBias: number = 0, // reserved for future depth-aware seam placement
  faceRects: FaceRectComposite[] = [],
  /** Optional saliency map (Float32 0–1) at composite resolution. */
  saliencyMap?: Float32Array | null,
): {
  gridW: number;
  gridH: number;
  dataCosts: Float32Array;       // 2 * gridW * gridH
  edgeWeights: Float32Array;     // (gridW-1)*gridH + gridW*(gridH-1)
  hardConstraints: Uint8Array;   // gridW * gridH
} {
  // Guard against zero-size composites
  if (compW <= 0 || compH <= 0) {
    return { gridW: 1, gridH: 1, dataCosts: new Float32Array(2), edgeWeights: new Float32Array(0), hardConstraints: new Uint8Array(1) };
  }
  const gridW = Math.max(1, Math.ceil(compW / blockSize));
  const gridH = Math.max(1, Math.ceil(compH / blockSize));
  const nNodes = gridW * gridH;

  const dataCosts = new Float32Array(nNodes * 2);
  const hardConstraints = new Uint8Array(nNodes);

  // ── Per-block alpha coverage ────────────────────────────
  // Instead of single center-pixel alpha, sample multiple pixels per block
  // to robustly determine which images have data.
  const compHasBlock = new Uint8Array(nNodes);  // 1 = composite has data
  const newHasBlock = new Uint8Array(nNodes);   // 1 = new image has data

  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const nodeIdx = gy * gridW + gx;
      const blockLeft = gx * blockSize;
      const blockTop = gy * blockSize;
      const blockW = Math.max(1, Math.min(blockSize, compW - blockLeft));
      const blockH = Math.max(1, Math.min(blockSize, compH - blockTop));
      // Sample a 3×3 grid within the block for robust alpha detection
      let compCount = 0, newCount = 0;
      const samples = 3;
      for (let sy = 0; sy < samples; sy++) {
        for (let sx = 0; sx < samples; sx++) {
          const px = blockLeft + Math.min(blockW - 1, Math.floor(((sx + 0.5) * blockW) / samples));
          const py = blockTop + Math.min(blockH - 1, Math.floor(((sy + 0.5) * blockH) / samples));
          const idx = (py * compW + px) * 4 + 3;
          if (compositePixels[idx] > CONTENT_ALPHA_THRESHOLD) compCount++;
          if (newImagePixels[idx] > CONTENT_ALPHA_THRESHOLD) newCount++;
        }
      }
      // Require majority (≥50%) pixel coverage to count as "has data".
      // This avoids counting blocks with only a sliver of valid pixels.
      const thresh = Math.ceil(samples * samples * 0.5);
      compHasBlock[nodeIdx] = compCount >= thresh ? 1 : 0;
      newHasBlock[nodeIdx] = newCount >= thresh ? 1 : 0;
    }
  }

  // ── Distance-from-boundary via Manhattan (L1) raster scan ───────────
  // For each image, compute distance from the nearest block WITHOUT data.
  // Blocks far from their boundary → strong preference to keep that image.
  const compDist = computeBlockDistanceField(compHasBlock, gridW, gridH);
  const newDist = computeBlockDistanceField(newHasBlock, gridW, gridH);

  // Normalise distances for data cost weighting
  let maxCompDist = 1, maxNewDist = 1;
  for (let i = 0; i < nNodes; i++) {
    if (compDist[i] > maxCompDist) maxCompDist = compDist[i];
    if (newDist[i] > maxNewDist) maxNewDist = newDist[i];
  }

  // ── Set hard constraints and data costs ─────────────────
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const nodeIdx = gy * gridW + gx;
      const cHas = compHasBlock[nodeIdx];
      const nHas = newHasBlock[nodeIdx];

      if (!cHas && !nHas) {
        dataCosts[nodeIdx * 2] = 0;
        dataCosts[nodeIdx * 2 + 1] = 0;
      } else if (cHas && !nHas) {
        hardConstraints[nodeIdx] = 1;
      } else if (!cHas && nHas) {
        hardConstraints[nodeIdx] = 2;
      } else {
        // ── Overlap: distance-based preference ───────
        // Prefer the image whose boundary is FURTHER away (more interior).
        // Also add a small color-difference term so the seam follows edges.
        const cD = compDist[nodeIdx] / maxCompDist;   // 0..1 (0 = near edge, 1 = deep interior)
        const nD = newDist[nodeIdx] / maxNewDist;

        const colorDiff = sampleBlockColorDifference(
          compositePixels,
          newImagePixels,
          compW,
          compH,
          gx,
          gy,
          blockSize,
        );

        // Cost for label=composite: high when NEAR composite boundary (penalise
        // keeping composite in its thin-coverage zone near the edge).
        // Cost for label=new: high when NEAR new boundary (penalise taking new image
        // in its thin-coverage zone).
        // Result: seam goes through the zone where both distances are roughly equal —
        // the centre of the overlap.
        const distWeight = 0.8;
        const colWeight = 0.2;

        dataCosts[nodeIdx * 2]     = distWeight * (1.0 - cD) + colWeight * colorDiff;
        dataCosts[nodeIdx * 2 + 1] = distWeight * (1.0 - nD) + colWeight * colorDiff;
        // ── Saliency-aware penalty ─────────────────────────────
        // If a saliency map is provided, penalise seam cuts through highly
        // salient regions (objects, people, detailed areas). This uses
        // AI-computed saliency maps that combine gradient magnitude (Sobel),
        // colour distinctness (Achanta frequency-tuned), and focus measure
        // (Laplacian variance). The seam is pushed towards low-saliency
        // (blurred/uniform) regions — ideal for Brenizer composites.
        if (saliencyMap && saliencyMap.length >= compW * compH) {
          // Average saliency within this block
          let salSum = 0;
          let salN = 0;
          const bx0 = gx * blockSize;
          const by0 = gy * blockSize;
          const bx1 = Math.min(bx0 + blockSize, compW);
          const by1 = Math.min(by0 + blockSize, compH);
          const step = Math.max(1, Math.floor(blockSize / 4));
          for (let sy = by0; sy < by1; sy += step) {
            for (let sx = bx0; sx < bx1; sx += step) {
              salSum += saliencyMap[sy * compW + sx];
              salN++;
            }
          }
          const avgSal = salN > 0 ? salSum / salN : 0;
          // Penalty proportional to saliency: high saliency → expensive to place seam
          const SALIENCY_PENALTY = 5.0;
          const salPenalty = avgSal * SALIENCY_PENALTY;
          // Add equal penalty to both labels — this makes the graph cut
          // prefer to NOT place the seam boundary here at all.
          dataCosts[nodeIdx * 2] += salPenalty;
          dataCosts[nodeIdx * 2 + 1] += salPenalty;
        }
        // ── Face-aware penalty ────────────────────────
        // If this block overlaps a face, massively penalise the label that
        // does NOT own the face — this keeps the seam away from face regions.
        const blockLeft = gx * blockSize;
        const blockTop = gy * blockSize;
        const blockRight = blockLeft + blockSize;
        const blockBottom = blockTop + blockSize;
        for (const face of faceRects) {
          const faceRight = face.x + face.width;
          const faceBottom = face.y + face.height;
          // Expand face rect by 50% in all directions — gives a safety margin
          // so the seam doesn't graze the face boundary due to block quantisation.
          const margin = Math.max(face.width, face.height) * 0.5;
          const fLeft = face.x - margin;
          const fTop = face.y - margin;
          const fRight = faceRight + margin;
          const fBottom = faceBottom + margin;
          // Check overlap
          if (blockLeft < fRight && blockRight > fLeft &&
              blockTop < fBottom && blockBottom > fTop) {
            // This block overlaps a face — penalise the OTHER label heavily.
            // A penalty of 10 is ~10× larger than typical edge weights, making
            // it extremely unlikely the graph cut splits a face between images.
            const FACE_PENALTY = 10.0;
            if (face.imageLabel === 0) {
              // Face belongs to composite: penalise label=new
              dataCosts[nodeIdx * 2 + 1] += FACE_PENALTY;
            } else {
              // Face belongs to new image: penalise label=composite
              dataCosts[nodeIdx * 2] += FACE_PENALTY;
            }
          }
        }
      }
    }
  }

  // ── Edge weights (smoothness term) ──────────────────────
  // Gradient-domain seam energy (inspired by Poisson image editing,
  // Pérez et al., SIGGRAPH 2003): instead of only penalising edges in the
  // *image* domain, we measure how well each image's gradients agree at
  // block boundaries.  Where both images have similar gradients, the seam
  // transition will be nearly invisible — so we *increase* the edge weight
  // there, discouraging the graph cut from placing a seam.  Where gradients
  // disagree (low agreement), the weight is lower, inviting a seam cut.
  //
  // Brenizer blur-aware enhancement: if a saliency map (which includes focus
  // measure from Laplacian variance) is provided, edges in blurred regions
  // get REDUCED weights — actively encouraging the seam to pass through
  // the bokeh zone. This is the key innovation for Brenizer composites.
  //
  // Concretely, each boundary edge weight is:
  //   w = (0.4×edgeStrength + 0.6×gradientAgreement) × blurDiscount
  // where blurDiscount = 1.0 in focused areas, down to 0.2 in blurred areas.
  const nHEdges = (gridW - 1) * gridH;
  const nVEdges = gridW * (gridH - 1);
  const edgeWeights = new Float32Array(nHEdges + nVEdges);

  const edgeSamples = Math.max(2, Math.min(blockSize, 8));

  /** Sample average saliency at the boundary between two adjacent blocks. */
  function sampleBoundarySaliency(bx: number, by: number, isHorizontal: boolean): number {
    if (!saliencyMap || saliencyMap.length < compW * compH) return 1.0;
    let sum = 0, count = 0;
    const samples = 3;
    for (let s = 0; s < samples; s++) {
      let px: number, py: number;
      if (isHorizontal) {
        px = bx;
        py = Math.min(by + Math.round((s + 0.5) * blockSize / samples), compH - 1);
      } else {
        px = Math.min(bx + Math.round((s + 0.5) * blockSize / samples), compW - 1);
        py = by;
      }
      if (px >= 0 && px < compW && py >= 0 && py < compH) {
        sum += saliencyMap[py * compW + px];
        count++;
      }
    }
    return count > 0 ? sum / count : 1.0;
  }

  // Horizontal edges
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW - 1; gx++) {
      const eIdx = gy * (gridW - 1) + gx;
      const bx = Math.min((gx + 1) * blockSize, compW - 1);
      let maxGrad = 0;
      let gradConsistencySum = 0;
      let gradSamples = 0;
      for (let s = 0; s < edgeSamples; s++) {
        const by = Math.min(gy * blockSize + Math.round((s + 0.5) * blockSize / edgeSamples), compH - 1);
        const pixIdx = (by * compW + bx) * 4;
        const prevIdx = Math.max(0, pixIdx - 4);
        // Compute per-image horizontal gradients and cross-image gradient difference
        let grad = 0;
        let gradDiffSum = 0;
        for (let ch = 0; ch < 3; ch++) {
          const compGrad = compositePixels[pixIdx + ch] - compositePixels[prevIdx + ch];
          const newGrad = newImagePixels[pixIdx + ch] - newImagePixels[prevIdx + ch];
          const cg = Math.abs(compGrad);
          const ng = Math.abs(newGrad);
          const xg = Math.abs(compositePixels[pixIdx + ch] - newImagePixels[pixIdx + ch]);
          grad += Math.max(cg, ng, xg);
          // Gradient consistency: how similar are the gradients?
          // Low value = gradients agree = good place for a seam
          gradDiffSum += Math.abs(compGrad - newGrad);
        }
        grad /= (255 * 3);
        if (grad > maxGrad) maxGrad = grad;
        // Normalize gradient consistency to [0, 1]
        gradConsistencySum += gradDiffSum / (255 * 3);
        gradSamples++;
      }
      // Combine: edge weight = blend of edge-avoidance and gradient consistency
      // High weight = strong penalty for cutting here = seam avoids this edge
      // We WANT the seam to cut where gradients agree (low gradConsistency)
      const avgGradConsistency = gradSamples > 0 ? gradConsistencySum / gradSamples : 0;
      const edgeStrength = Math.max(0.01, 1.0 - maxGrad);
      // Invert gradient consistency: high agreement → high penalty (don't cut here)
      // High disagreement → low penalty (okay to cut here... but actually no,
      // we want to cut where gradients AGREE, so low disagreement → good seam)
      const gradientAgreement = Math.max(0.01, 1.0 - avgGradConsistency);
      // Weighted combination: 60% gradient-domain, 40% edge avoidance
      let w = 0.4 * edgeStrength + 0.6 * gradientAgreement;
      // ── Brenizer blur discount ──────────────────────────
      // In blurred (bokeh) regions, reduce edge weight to encourage the seam
      // to pass through. Low saliency ≈ blurred ≈ good seam location.
      const sal = sampleBoundarySaliency(bx, gy * blockSize, true);
      const blurDiscount = 0.2 + 0.8 * sal; // 0.2 in pure blur → 1.0 in sharp
      w *= blurDiscount;
      edgeWeights[eIdx] = w;
    }
  }

  // Vertical edges
  for (let gy = 0; gy < gridH - 1; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const eIdx = nHEdges + gy * gridW + gx;
      const by = Math.min((gy + 1) * blockSize, compH - 1);
      const stride = compW * 4;
      let maxGrad = 0;
      let gradConsistencySum = 0;
      let gradSamples = 0;
      for (let s = 0; s < edgeSamples; s++) {
        const bx = Math.min(gx * blockSize + Math.round((s + 0.5) * blockSize / edgeSamples), compW - 1);
        const pixIdx = (by * compW + bx) * 4;
        const prevIdx = Math.max(0, pixIdx - stride);
        let grad = 0;
        let gradDiffSum = 0;
        for (let ch = 0; ch < 3; ch++) {
          const compGrad = compositePixels[pixIdx + ch] - compositePixels[prevIdx + ch];
          const newGrad = newImagePixels[pixIdx + ch] - newImagePixels[prevIdx + ch];
          const cg = Math.abs(compGrad);
          const ng = Math.abs(newGrad);
          const xg = Math.abs(compositePixels[pixIdx + ch] - newImagePixels[pixIdx + ch]);
          grad += Math.max(cg, ng, xg);
          gradDiffSum += Math.abs(compGrad - newGrad);
        }
        grad /= (255 * 3);
        if (grad > maxGrad) maxGrad = grad;
        gradConsistencySum += gradDiffSum / (255 * 3);
        gradSamples++;
      }
      const avgGradConsistency = gradSamples > 0 ? gradConsistencySum / gradSamples : 0;
      const edgeStrength = Math.max(0.01, 1.0 - maxGrad);
      const gradientAgreement = Math.max(0.01, 1.0 - avgGradConsistency);
      let vw = 0.4 * edgeStrength + 0.6 * gradientAgreement;
      // Brenizer blur discount for vertical edges
      const vSal = sampleBoundarySaliency(gx * blockSize, by, false);
      const vBlurDiscount = 0.2 + 0.8 * vSal;
      vw *= vBlurDiscount;
      edgeWeights[eIdx] = vw;
    }
  }

  return { gridW, gridH, dataCosts, edgeWeights, hardConstraints };
}

/**
 * Compute Manhattan (L1) distance transform on a binary block grid.
 * Distance = 0 at boundary (adjacent to a 0-block or grid edge), increases inward.
 * Uses two-pass raster scan (O(n) approximation of the exact distance transform).
 */
function computeBlockDistanceField(
  hasData: Uint8Array,
  gridW: number,
  gridH: number,
): Float32Array {
  const dist = new Float32Array(gridW * gridH);
  const INF = gridW + gridH;

  // Initialise: seed distance zero at the true image boundary, not just
  // at empty blocks. This keeps edge-touching image corners from being
  // misclassified as deep interior by the graph-cut data term.
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
      dist[idx] = (touchesGridEdge || touchesVoid) ? 0 : INF;
    }
  }

  // Forward pass (top-left → bottom-right)
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const idx = gy * gridW + gx;
      if (gx > 0) dist[idx] = Math.min(dist[idx], dist[idx - 1] + 1);
      if (gy > 0) dist[idx] = Math.min(dist[idx], dist[(gy - 1) * gridW + gx] + 1);
    }
  }

  // Backward pass (bottom-right → top-left)
  for (let gy = gridH - 1; gy >= 0; gy--) {
    for (let gx = gridW - 1; gx >= 0; gx--) {
      const idx = gy * gridW + gx;
      if (gx < gridW - 1) dist[idx] = Math.min(dist[idx], dist[idx + 1] + 1);
      if (gy < gridH - 1) dist[idx] = Math.min(dist[idx], dist[(gy + 1) * gridW + gx] + 1);
    }
  }

  return dist;
}

/**
 * Estimate the overlap width between two image regions for adaptive feathering.
 * PTGui-style: the feather width should match the overlap zone width so the
 * blend transition spans exactly the overlapping region.
 *
 * Measures overlap in both horizontal and vertical directions and returns
 * the minimum (to match the narrowest dimension of the overlap zone).
 * Returns the estimated overlap width in pixels, or the fallback if no overlap detected.
 */
export function estimateOverlapWidth(
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  width: number,
  height: number,
  fallback: number,
): number {
  const rowSpans = collectOverlapSpans(compositePixels, newImagePixels, width, height, true);
  const colSpans = collectOverlapSpans(compositePixels, newImagePixels, width, height, false);

  const spanCandidates: number[] = [];
  if (rowSpans.length > 0) spanCandidates.push(quantile(rowSpans, 0.35));
  if (colSpans.length > 0) spanCandidates.push(quantile(colSpans, 0.35));

  if (spanCandidates.length > 0) {
    // Bias toward the narrower stable overlap dimension. This avoids
    // over-feathering when images only touch in a corner or tapered wedge.
    const narrowOverlap = Math.min(...spanCandidates);
    if (Number.isFinite(narrowOverlap) && narrowOverlap > 0) {
      const estimated = Math.max(4, Math.round(narrowOverlap * 0.5));
      // Keep the adaptive width from collapsing too far below the configured
      // feather. Coverage clamping already protects corner slivers, so a
      // modest floor here is safer than opening hairline alpha gaps.
      const minAdaptive = Math.max(4, Math.round(fallback * 0.5));
      const maxAdaptive = Math.max(minAdaptive, Math.round(fallback * 2));
      return Math.max(minAdaptive, Math.min(maxAdaptive, estimated));
    }
  }
  return fallback;
}

/**
 * Build a feathered seam mask and post-process it against real coverage.
 * This is shared by preview + export so both paths clamp corners, harden
 * obvious ghosts, and smooth residual exposure mismatch the same way.
 */
export function buildAdaptiveBlendMask(
  baseMask: Uint8Array,
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  width: number,
  height: number,
  fallbackRadius: number,
  options: AdaptiveBlendMaskOptions = {},
): AdaptiveBlendMaskResult {
  const featherRadius = estimateOverlapWidth(compositePixels, newImagePixels, width, height, fallbackRadius);
  const mask = featherMask(baseMask, width, height, featherRadius);
  clampBlendMaskToCoverage(mask, compositePixels, newImagePixels);
  const ghostStats = hardenGhostRegions(
    mask,
    compositePixels,
    newImagePixels,
    width,
    height,
    options.ghostBlockSize,
  );
  refineSeamMaskForLighting(mask, compositePixels, newImagePixels, width, height, featherRadius);
  return {
    mask,
    featherRadius,
    ghostPixels: ghostStats.ghostPixels,
    ghostThreshold: ghostStats.ghostThreshold,
    ghostMedianDiff: ghostStats.ghostMedianDiff,
  };
}

/**
 * Convert block-grid labels to a full-resolution alpha mask.
 * label 0 = composite region (mask=0), label 1 = new image region (mask=255).
 */
export function labelsToMask(
  labels: Uint8Array,
  gridW: number,
  gridH: number,
  blockSize: number,
  outW: number,
  outH: number,
): Uint8Array {
  const mask = new Uint8Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    const gy = Math.min(Math.floor(y / blockSize), gridH - 1);
    for (let x = 0; x < outW; x++) {
      const gx = Math.min(Math.floor(x / blockSize), gridW - 1);
      mask[y * outW + x] = labels[gy * gridW + gx] ? 255 : 0;
    }
  }
  return mask;
}

function sampleBlockColorDifference(
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  compW: number,
  compH: number,
  gx: number,
  gy: number,
  blockSize: number,
): number {
  const blockLeft = gx * blockSize;
  const blockTop = gy * blockSize;
  const blockW = Math.max(1, Math.min(blockSize, compW - blockLeft));
  const blockH = Math.max(1, Math.min(blockSize, compH - blockTop));
  const samples = 3;
  let sum = 0;
  let count = 0;
  let fallbackOff = (Math.min(blockTop + (blockH >> 1), compH - 1) * compW + Math.min(blockLeft + (blockW >> 1), compW - 1)) * 4;

  for (let sy = 0; sy < samples; sy++) {
    for (let sx = 0; sx < samples; sx++) {
      const px = blockLeft + Math.min(blockW - 1, Math.floor(((sx + 0.5) * blockW) / samples));
      const py = blockTop + Math.min(blockH - 1, Math.floor(((sy + 0.5) * blockH) / samples));
      const off = (py * compW + px) * 4;
      fallbackOff = off;
      if (compositePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD || newImagePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD) {
        continue;
      }
      sum += pixelColorMismatch(compositePixels, newImagePixels, off) / 255;
      count++;
    }
  }

  if (count > 0) return sum / count;
  return pixelColorMismatch(compositePixels, newImagePixels, fallbackOff) / 255;
}

function createSamplePositions(size: number, maxSamples: number): number[] {
  const limit = Math.max(1, Math.min(size, maxSamples));
  const positions: number[] = [];
  let prev = -1;
  for (let i = 0; i < limit; i++) {
    const pos = Math.min(size - 1, Math.max(0, Math.round(((i + 0.5) * size) / limit - 0.5)));
    if (pos !== prev) {
      positions.push(pos);
      prev = pos;
    }
  }
  return positions;
}

function collectOverlapSpans(
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  width: number,
  height: number,
  horizontal: boolean,
): number[] {
  const spans: number[] = [];
  const positions = createSamplePositions(horizontal ? height : width, 24);

  for (const pos of positions) {
    let runStart = -1;
    let bestRun = 0;
    const length = horizontal ? width : height;
    for (let i = 0; i < length; i++) {
      const x = horizontal ? i : pos;
      const y = horizontal ? pos : i;
      const alphaIdx = (y * width + x) * 4 + 3;
      const overlaps =
        compositePixels[alphaIdx] > CONTENT_ALPHA_THRESHOLD
        && newImagePixels[alphaIdx] > CONTENT_ALPHA_THRESHOLD;
      if (overlaps) {
        if (runStart < 0) runStart = i;
      } else if (runStart >= 0) {
        bestRun = Math.max(bestRun, i - runStart);
        runStart = -1;
      }
    }
    if (runStart >= 0) bestRun = Math.max(bestRun, length - runStart);
    if (bestRun > 0) spans.push(bestRun);
  }

  return spans;
}

function quantile(values: number[], q: number): number {
  if (values.length === 0) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const t = Math.max(0, Math.min(1, q));
  const idx = (sorted.length - 1) * t;
  const lo = Math.floor(idx);
  const hi = Math.ceil(idx);
  if (lo === hi) return sorted[lo];
  const frac = idx - lo;
  return sorted[lo] * (1 - frac) + sorted[hi] * frac;
}

function clampUnit(v: number): number {
  return Math.max(0, Math.min(1, v));
}

function pixelColorMismatch(
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  off: number,
): number {
  const compR = compositePixels[off];
  const compG = compositePixels[off + 1];
  const compB = compositePixels[off + 2];
  const newR = newImagePixels[off];
  const newG = newImagePixels[off + 1];
  const newB = newImagePixels[off + 2];
  const compLum = 0.2126 * compR + 0.7152 * compG + 0.0722 * compB;
  const newLum = 0.2126 * newR + 0.7152 * newG + 0.0722 * newB;
  const lumDiff = Math.abs(compLum - newLum);
  const chromaDiff = (Math.abs(compR - newR) + Math.abs(compG - newG) + Math.abs(compB - newB)) / 3;
  return lumDiff * 0.6 + chromaDiff * 0.4;
}

function clampBlendMaskToCoverage(
  mask: Uint8Array,
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
): void {
  for (let px = 0; px < mask.length; px++) {
    const off = px * 4;
    const compAlpha = compositePixels[off + 3];
    const newAlpha = newImagePixels[off + 3];
    if (newAlpha <= CONTENT_ALPHA_THRESHOLD) {
      mask[px] = 0;
    } else if (compAlpha <= CONTENT_ALPHA_THRESHOLD) {
      mask[px] = 255;
    }
  }
}

function hardenGhostRegions(
  mask: Uint8Array,
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  width: number,
  height: number,
  ghostBlockSize?: number,
): {
  ghostPixels: number;
  ghostThreshold: number;
  ghostMedianDiff: number;
} {
  const blockSize = Math.max(8, Math.round(ghostBlockSize ?? Math.max(16, Math.min(128, width > 0 ? Math.sqrt(width * height) * 0.025 : 16))));
  const gridW = Math.ceil(width / blockSize);
  const gridH = Math.ceil(height / blockSize);
  const blockDiffs = new Float32Array(gridW * gridH);
  const blockCounts = new Uint32Array(gridW * gridH);

  for (let y = 0; y < height; y++) {
    const gy = Math.min(Math.floor(y / blockSize), gridH - 1);
    for (let x = 0; x < width; x++) {
      const px = y * width + x;
      const off = px * 4;
      if (compositePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD || newImagePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD) {
        continue;
      }
      const gx = Math.min(Math.floor(x / blockSize), gridW - 1);
      const gi = gy * gridW + gx;
      blockDiffs[gi] += pixelColorMismatch(compositePixels, newImagePixels, off);
      blockCounts[gi]++;
    }
  }

  const means: number[] = [];
  for (let i = 0; i < blockDiffs.length; i++) {
    if (blockCounts[i] === 0) continue;
    blockDiffs[i] /= blockCounts[i];
    means.push(blockDiffs[i]);
  }
  if (means.length < 4) {
    return { ghostPixels: 0, ghostThreshold: 0, ghostMedianDiff: 0 };
  }

  const ghostMedianDiff = quantile(means, 0.5);
  const ghostThreshold = Math.max(ghostMedianDiff * 3, 30);
  let ghostPixels = 0;

  for (let y = 0; y < height; y++) {
    const gy = Math.min(Math.floor(y / blockSize), gridH - 1);
    for (let x = 0; x < width; x++) {
      const px = y * width + x;
      const off = px * 4;
      if (compositePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD || newImagePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD) {
        continue;
      }
      const gx = Math.min(Math.floor(x / blockSize), gridW - 1);
      const gi = gy * gridW + gx;
      if (blockDiffs[gi] <= ghostThreshold) continue;
      const m = mask[px];
      if (m <= MASK_HARD_LOW || m >= MASK_HARD_HIGH) continue;
      mask[px] = m > 127 ? 255 : 0;
      ghostPixels++;
    }
  }

  return { ghostPixels, ghostThreshold, ghostMedianDiff };
}

function refineSeamMaskForLighting(
  mask: Uint8Array,
  compositePixels: Uint8Array,
  newImagePixels: Uint8Array,
  width: number,
  height: number,
  baseFeatherRadius: number,
): void {
  const n = width * height;
  if (mask.length !== n) return;
  if (compositePixels.length < n * 4 || newImagePixels.length < n * 4) return;

  const blockSize = Math.max(16, Math.min(96, Math.round(Math.max(12, baseFeatherRadius * 6))));
  const gridW = Math.ceil(width / blockSize);
  const gridH = Math.ceil(height / blockSize);
  const blockDiff = new Float32Array(gridW * gridH);
  const blockCount = new Uint32Array(gridW * gridH);
  const overlapMask = new Uint8Array(n);

  for (let y = 0; y < height; y++) {
    const gy = Math.min(Math.floor(y / blockSize), gridH - 1);
    for (let x = 0; x < width; x++) {
      const px = y * width + x;
      const off = px * 4;
      if (compositePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD || newImagePixels[off + 3] <= CONTENT_ALPHA_THRESHOLD) {
        continue;
      }
      overlapMask[px] = 1;
      const gx = Math.min(Math.floor(x / blockSize), gridW - 1);
      const gi = gy * gridW + gx;
      blockDiff[gi] += pixelColorMismatch(compositePixels, newImagePixels, off);
      blockCount[gi]++;
    }
  }

  const means: number[] = [];
  for (let gi = 0; gi < blockDiff.length; gi++) {
    if (blockCount[gi] === 0) continue;
    blockDiff[gi] /= blockCount[gi];
    if (blockCount[gi] >= 8) means.push(blockDiff[gi]);
  }
  if (means.length < 4) return;

  const median = quantile(means, 0.5);
  const softStart = Math.max(6, median * 1.15);
  const softEnd = Math.max(softStart + 12, median * 3.2);
  const denom = Math.max(1, softEnd - softStart);

  let softened = 0;
  for (let y = 0; y < height; y++) {
    const gy = Math.min(Math.floor(y / blockSize), gridH - 1);
    for (let x = 0; x < width; x++) {
      const px = y * width + x;
      if (!overlapMask[px]) continue;
      const m = mask[px];
      if (m <= MASK_HARD_LOW || m >= MASK_HARD_HIGH) continue;
      const gx = Math.min(Math.floor(x / blockSize), gridW - 1);
      const d = blockDiff[gy * gridW + gx];
      if (d <= softStart) continue;
      const t = clampUnit((d - softStart) / denom);
      const blend = 0.08 + 0.26 * t;
      mask[px] = Math.round(m * (1 - blend) + 128 * blend);
      softened++;
    }
  }
  if (softened < 32) return;

  const microRadius = Math.max(1, Math.round(baseFeatherRadius * 0.22));
  const smoothed = featherMask(mask, width, height, microRadius, 2);
  for (let px = 0; px < n; px++) {
    if (!overlapMask[px]) continue;
    const m = mask[px];
    if (m <= MASK_HARD_LOW || m >= MASK_HARD_HIGH) continue;
    mask[px] = Math.round(m * 0.7 + smoothed[px] * 0.3);
  }
}

/**
 * Feather (blur) a single-channel Uint8 mask on CPU using separable box blur.
 * Multiple passes approximate Gaussian. Returns new mask.
 */
export function featherMask(
  mask: Uint8Array,
  w: number,
  h: number,
  radius: number,
  passes: number = 3,
): Uint8Array {
  let src = new Float32Array(mask.length);
  let dst = new Float32Array(mask.length);
  for (let i = 0; i < mask.length; i++) src[i] = mask[i];

  const r = Math.max(1, Math.round(radius));

  for (let p = 0; p < passes; p++) {
    // Horizontal pass
    for (let y = 0; y < h; y++) {
      let sum = 0;
      const row = y * w;
      // Initialize window
      for (let x = 0; x <= r && x < w; x++) sum += src[row + x];
      let count = Math.min(r + 1, w);

      for (let x = 0; x < w; x++) {
        dst[row + x] = sum / count;
        // Expand/shrink window
        const addX = x + r + 1;
        const remX = x - r;
        if (addX < w) { sum += src[row + addX]; count++; }
        if (remX >= 0) { sum -= src[row + remX]; count--; }
      }
    }

    // Vertical pass
    [src, dst] = [dst, src];
    for (let x = 0; x < w; x++) {
      let sum = 0;
      for (let y = 0; y <= r && y < h; y++) sum += src[y * w + x];
      let count = Math.min(r + 1, h);

      for (let y = 0; y < h; y++) {
        dst[y * w + x] = sum / count;
        const addY = y + r + 1;
        const remY = y - r;
        if (addY < h) { sum += src[addY * w + x]; count++; }
        if (remY >= 0) { sum -= src[remY * w + x]; count--; }
      }
    }

    [src, dst] = [dst, src];
  }

  const out = new Uint8Array(mask.length);
  for (let i = 0; i < mask.length; i++) out[i] = Math.round(Math.max(0, Math.min(255, src[i])));
  return out;
}

/** Create a single-channel (luminance) texture from a Uint8 mask. */
export function createMaskTexture(
  gl: WebGL2RenderingContext,
  mask: Uint8Array,
  w: number,
  h: number,
): ManagedTexture {
  const tex = gl.createTexture()!;
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.R8, w, h, 0, gl.RED, gl.UNSIGNED_BYTE, mask);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 4);
  gl.bindTexture(gl.TEXTURE_2D, null);
  return { texture: tex, width: w, height: h, dispose: () => gl.deleteTexture(tex) };
}

/** Create compositor for performing seam-masked composition. */
export function createCompositor(gl: WebGL2RenderingContext): Compositor {
  const blendProg = createProgram(gl, BLEND_VERT, BLEND_FRAG);
  const blendLoc = {
    aPos: gl.getAttribLocation(blendProg, 'a_position'),
    uComp: gl.getUniformLocation(blendProg, 'u_composite'),
    uNew: gl.getUniformLocation(blendProg, 'u_newImage'),
    uMask: gl.getUniformLocation(blendProg, 'u_mask'),
  };

  const blurProg = createProgram(gl, BLUR_VERT, BLUR_FRAG);
  const blurLoc = {
    aPos: gl.getAttribLocation(blurProg, 'a_position'),
    uTex: gl.getUniformLocation(blurProg, 'u_texture'),
    uDir: gl.getUniformLocation(blurProg, 'u_direction'),
    uRad: gl.getUniformLocation(blurProg, 'u_radius'),
  };

  // Fullscreen quad VAO
  const quadVAO = gl.createVertexArray()!;
  const quadVBO = gl.createBuffer()!;
  gl.bindVertexArray(quadVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(blendLoc.aPos);
  gl.vertexAttribPointer(blendLoc.aPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  return {
    blendWithMask(compositeTex, newImageTex, maskTex, outputFBO, width, height) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, outputFBO);
      gl.viewport(0, 0, width, height);
      gl.useProgram(blendProg);

      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, compositeTex);
      gl.uniform1i(blendLoc.uComp, 0);

      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, newImageTex);
      gl.uniform1i(blendLoc.uNew, 1);

      gl.activeTexture(gl.TEXTURE2);
      gl.bindTexture(gl.TEXTURE_2D, maskTex);
      gl.uniform1i(blendLoc.uMask, 2);

      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    },

    blur(sourceTex, sourceFBO, tempTex, tempFBO, width, height, radius) {
      if (radius < 1) return;

      gl.useProgram(blurProg);
      gl.bindVertexArray(quadVAO);

      // Horizontal pass: source → temp
      gl.bindFramebuffer(gl.FRAMEBUFFER, tempFBO);
      gl.viewport(0, 0, width, height);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, sourceTex);
      gl.uniform1i(blurLoc.uTex, 0);
      gl.uniform2f(blurLoc.uDir, 1.0 / width, 0);
      gl.uniform1f(blurLoc.uRad, radius);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      // Vertical pass: temp → source
      gl.bindFramebuffer(gl.FRAMEBUFFER, sourceFBO);
      gl.viewport(0, 0, width, height);
      gl.bindTexture(gl.TEXTURE_2D, tempTex);
      gl.uniform2f(blurLoc.uDir, 0, 1.0 / height);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);

      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    },

    dispose() {
      gl.deleteProgram(blendProg);
      gl.deleteProgram(blurProg);
      gl.deleteVertexArray(quadVAO);
      gl.deleteBuffer(quadVBO);
    },
  };
}
