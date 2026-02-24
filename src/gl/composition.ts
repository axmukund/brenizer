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

// ── Blend shader: mix(composite, new, mask) ──────────────

const BLEND_VERT = `#version 300 es
precision highp float;
in vec2 a_position;
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
  // Where new image has zero alpha (outside warp), keep composite
  m *= newI.a;
  fragColor = mix(comp, newI, m);
  fragColor.a = max(comp.a, newI.a * m);
}
`;

// ── Separable Gaussian blur ──────────────────────────────

const BLUR_VERT = `#version 300 es
precision highp float;
in vec2 a_position;
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
 * Returns { dataCosts, edgeWeights, hardConstraints } for seam-worker.
 */
export function computeBlockCosts(
  compositePixels: Uint8Array,   // RGBA at composite resolution
  newImagePixels: Uint8Array,    // RGBA at composite resolution
  compW: number,
  compH: number,
  blockSize: number,
  depthBias: number = 0,
): {
  gridW: number;
  gridH: number;
  dataCosts: Float32Array;       // 2 * gridW * gridH
  edgeWeights: Float32Array;     // (gridW-1)*gridH + gridW*(gridH-1)
  hardConstraints: Uint8Array;   // gridW * gridH
} {
  const gridW = Math.max(1, Math.floor(compW / blockSize));
  const gridH = Math.max(1, Math.floor(compH / blockSize));
  const nNodes = gridW * gridH;

  const dataCosts = new Float32Array(nNodes * 2);
  const hardConstraints = new Uint8Array(nNodes);

  // Compute per-block data terms
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const nodeIdx = gy * gridW + gx;

      // Sample center pixel of block
      const cx = Math.min(gx * blockSize + Math.floor(blockSize / 2), compW - 1);
      const cy = Math.min(gy * blockSize + Math.floor(blockSize / 2), compH - 1);
      const pixIdx = (cy * compW + cx) * 4;

      const compR = compositePixels[pixIdx];
      const compG = compositePixels[pixIdx + 1];
      const compB = compositePixels[pixIdx + 2];
      const compA = compositePixels[pixIdx + 3];

      const newR = newImagePixels[pixIdx];
      const newG = newImagePixels[pixIdx + 1];
      const newB = newImagePixels[pixIdx + 2];
      const newA = newImagePixels[pixIdx + 3];

      const compHasData = compA > 10;
      const newHasData = newA > 10;

      if (!compHasData && !newHasData) {
        // Neither has data — free
        dataCosts[nodeIdx * 2] = 0;
        dataCosts[nodeIdx * 2 + 1] = 0;
      } else if (compHasData && !newHasData) {
        // Only composite — force keep composite
        hardConstraints[nodeIdx] = 1;
        dataCosts[nodeIdx * 2] = 0;
        dataCosts[nodeIdx * 2 + 1] = 0;
      } else if (!compHasData && newHasData) {
        // Only new — force take new
        hardConstraints[nodeIdx] = 2;
        dataCosts[nodeIdx * 2] = 0;
        dataCosts[nodeIdx * 2 + 1] = 0;
      } else {
        // Both have data — overlap region
        // Color difference as data cost (slightly prefer composite to avoid unnecessary cuts)
        const diffR = Math.abs(compR - newR) / 255;
        const diffG = Math.abs(compG - newG) / 255;
        const diffB = Math.abs(compB - newB) / 255;
        const diff = (diffR + diffG + diffB) / 3;

        // Cost for keeping composite (slightly lower = prefer composite where similar)
        dataCosts[nodeIdx * 2] = diff * 0.5;
        // Cost for taking new (slightly higher = prefer composite where similar)
        dataCosts[nodeIdx * 2 + 1] = diff * 0.5;
      }
    }
  }

  // Compute edge weights (smoothness term between adjacent blocks)
  const nHEdges = (gridW - 1) * gridH;
  const nVEdges = gridW * (gridH - 1);
  const edgeWeights = new Float32Array(nHEdges + nVEdges);

  // Horizontal edges
  for (let gy = 0; gy < gridH; gy++) {
    for (let gx = 0; gx < gridW - 1; gx++) {
      const eIdx = gy * (gridW - 1) + gx;
      // Sample boundary between blocks gx and gx+1
      const bx = Math.min((gx + 1) * blockSize, compW - 1);
      const by = Math.min(gy * blockSize + Math.floor(blockSize / 2), compH - 1);
      const pixIdx = (by * compW + bx) * 4;

      // Weight inversely proportional to color gradient in composite+new
      const compGrad = Math.abs(
        compositePixels[pixIdx] - compositePixels[Math.max(0, pixIdx - 4)]
      ) / 255;
      const newGrad = Math.abs(
        newImagePixels[pixIdx] - newImagePixels[Math.max(0, pixIdx - 4)]
      ) / 255;
      const grad = Math.max(compGrad, newGrad);
      // High weight where image is smooth (seam should go through edges/gradients)
      edgeWeights[eIdx] = Math.max(0.01, 1.0 - grad);
    }
  }

  // Vertical edges
  for (let gy = 0; gy < gridH - 1; gy++) {
    for (let gx = 0; gx < gridW; gx++) {
      const eIdx = nHEdges + gy * gridW + gx;
      const bx = Math.min(gx * blockSize + Math.floor(blockSize / 2), compW - 1);
      const by = Math.min((gy + 1) * blockSize, compH - 1);
      const pixIdx = (by * compW + bx) * 4;

      const stride = compW * 4;
      const compGrad = Math.abs(
        compositePixels[pixIdx] - compositePixels[Math.max(0, pixIdx - stride)]
      ) / 255;
      const newGrad = Math.abs(
        newImagePixels[pixIdx] - newImagePixels[Math.max(0, pixIdx - stride)]
      ) / 255;
      const grad = Math.max(compGrad, newGrad);
      edgeWeights[eIdx] = Math.max(0.01, 1.0 - grad);
    }
  }

  return { gridW, gridH, dataCosts, edgeWeights, hardConstraints };
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
