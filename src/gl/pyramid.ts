/**
 * Laplacian pyramid blending for multi-band compositing.
 * Uses GPU shaders for downsample, upsample, Laplacian computation and blending.
 * 
 * Pipeline per new image:
 *  1. Build Gaussian pyramid Gc for composite, Gn for new image, Gm for mask
 *  2. Build Laplacian pyramid Lc from Gc, Ln from Gn
 *  3. Blend Laplacian levels: Lb[l] = (1-Gm[l]) * Lc[l] + Gm[l] * Ln[l]
 *  4. Reconstruct from blended Laplacian pyramid
 */

import { createProgram } from './programs';
import { createEmptyTexture, type ManagedTexture } from './textures';
import { createFBO, type ManagedFBO } from './framebuffers';

// ── Shared fullscreen vert shader ────────────────────────

const FS_VERT = `#version 300 es
precision highp float;
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

// ── Downsample: simple 2x2 box filter ────────────────────

const DOWNSAMPLE_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec2 u_texelSize;  // 1/srcWidth, 1/srcHeight
out vec4 fragColor;
void main() {
  // Sample 4 texels for 2x2 downsample
  vec2 hs = u_texelSize * 0.5;
  vec4 a = texture(u_texture, v_uv + vec2(-hs.x, -hs.y));
  vec4 b = texture(u_texture, v_uv + vec2( hs.x, -hs.y));
  vec4 c = texture(u_texture, v_uv + vec2(-hs.x,  hs.y));
  vec4 d = texture(u_texture, v_uv + vec2( hs.x,  hs.y));
  fragColor = (a + b + c + d) * 0.25;
}
`;

// ── Upsample: bilinear 2x ────────────────────────────────

const UPSAMPLE_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
  fragColor = texture(u_texture, v_uv);
}
`;

// ── Laplacian: L = current - upsample(downsampled) ──────

const LAPLACIAN_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_current;
uniform sampler2D u_upsampled;
out vec4 fragColor;
void main() {
  vec4 cur = texture(u_current, v_uv);
  vec4 up = texture(u_upsampled, v_uv);
  // Laplacian = detail that was lost in downsampling
  // Store as offset from 0.5 to keep it in [0,1] range for RGBA8
  fragColor = vec4((cur.rgb - up.rgb) * 0.5 + 0.5, cur.a);
}
`;

// ── Laplacian float: no bias needed for RGBA16F ──────

const LAPLACIAN_FLOAT_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_current;
uniform sampler2D u_upsampled;
out vec4 fragColor;
void main() {
  vec4 cur = texture(u_current, v_uv);
  vec4 up = texture(u_upsampled, v_uv);
  fragColor = vec4(cur.rgb - up.rgb, cur.a);
}
`;

// ── Blend Laplacian levels ───────────────────────────────

const BLEND_LAP_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_lapComp;   // Laplacian of composite
uniform sampler2D u_lapNew;    // Laplacian of new image
uniform sampler2D u_mask;      // Gaussian blurred mask at this level
out vec4 fragColor;
void main() {
  vec4 lc = texture(u_lapComp, v_uv);
  vec4 ln = texture(u_lapNew, v_uv);
  float m = texture(u_mask, v_uv).r;
  // Handle alpha: where new image has no data, keep composite
  float effectiveM = m * ln.a;
  fragColor = vec4(mix(lc.rgb, ln.rgb, effectiveM), max(lc.a, ln.a));
}
`;

// ── Reconstruct: add Laplacian + upsampled lower level ──

const RECONSTRUCT_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_laplacian;
uniform sampler2D u_upsampled;
out vec4 fragColor;
void main() {
  vec4 lap = texture(u_laplacian, v_uv);
  vec4 up = texture(u_upsampled, v_uv);
  // Undo the 0.5 offset from Laplacian storage
  vec3 detail = (lap.rgb - 0.5) * 2.0;
  fragColor = vec4(up.rgb + detail, lap.a);
}
`;

// ── Reconstruct float: no bias undo for RGBA16F ─────

const RECONSTRUCT_FLOAT_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_laplacian;
uniform sampler2D u_upsampled;
out vec4 fragColor;
void main() {
  vec4 lap = texture(u_laplacian, v_uv);
  vec4 up = texture(u_upsampled, v_uv);
  fragColor = vec4(up.rgb + lap.rgb, lap.a);
}
`;

// ── Types ────────────────────────────────────────────────

interface PyramidLevel {
  tex: ManagedTexture;
  fbo: ManagedFBO;
  w: number;
  h: number;
}

export interface PyramidBlender {
  /**
   * Perform multi-band blend of composite + new image using seam mask.
   * Writes result to outputFBO (or default FB if null).
   * @param levels Number of pyramid levels (3–7)
   */
  blend(
    compositeTex: WebGLTexture,
    newImageTex: WebGLTexture,
    maskTex: WebGLTexture,
    outputFBO: WebGLFramebuffer | null,
    width: number,
    height: number,
    levels: number,
  ): void;

  dispose(): void;
}

/**
 * Create a GPU-accelerated Laplacian pyramid blender.
 * @param useFloat When true, intermediate Laplacian levels use RGBA16F textures.
 *                 This avoids 8-bit clamping artefacts in the detail bands,
 *                 producing smoother transitions at the cost of more VRAM.
 *                 Falls back to RGBA8 if the extension is unavailable.
 */
export function createPyramidBlender(gl: WebGL2RenderingContext, useFloat: boolean = false): PyramidBlender {
  // Compile all shaders — pick float variants when available
  const downsampleProg = createProgram(gl, FS_VERT, DOWNSAMPLE_FRAG);
  const upsampleProg = createProgram(gl, FS_VERT, UPSAMPLE_FRAG);
  const laplacianProg = createProgram(gl, FS_VERT, useFloat ? LAPLACIAN_FLOAT_FRAG : LAPLACIAN_FRAG);
  const blendLapProg = createProgram(gl, FS_VERT, BLEND_LAP_FRAG);
  const reconstructProg = createProgram(gl, FS_VERT, useFloat ? RECONSTRUCT_FLOAT_FRAG : RECONSTRUCT_FRAG);

  // Uniform locations
  const dsLoc = {
    uTex: gl.getUniformLocation(downsampleProg, 'u_texture'),
    uTexelSize: gl.getUniformLocation(downsampleProg, 'u_texelSize'),
  };
  const usLoc = {
    uTex: gl.getUniformLocation(upsampleProg, 'u_texture'),
  };
  const lapLoc = {
    uCurrent: gl.getUniformLocation(laplacianProg, 'u_current'),
    uUpsampled: gl.getUniformLocation(laplacianProg, 'u_upsampled'),
  };
  const blLoc = {
    uLapComp: gl.getUniformLocation(blendLapProg, 'u_lapComp'),
    uLapNew: gl.getUniformLocation(blendLapProg, 'u_lapNew'),
    uMask: gl.getUniformLocation(blendLapProg, 'u_mask'),
  };
  const recLoc = {
    uLap: gl.getUniformLocation(reconstructProg, 'u_laplacian'),
    uUp: gl.getUniformLocation(reconstructProg, 'u_upsampled'),
  };

  // Fullscreen quad VAO
  const quadVAO = gl.createVertexArray()!;
  const quadVBO = gl.createBuffer()!;
  gl.bindVertexArray(quadVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const posLoc = gl.getAttribLocation(downsampleProg, 'a_position');
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  function createLevel(w: number, h: number): PyramidLevel {
    const tex = createEmptyTexture(gl, w, h);
    const fbo = createFBO(gl, tex.texture);
    return { tex, fbo, w, h };
  }

  function createLapLevel(w: number, h: number): PyramidLevel {
    if (useFloat) {
      const tex = createEmptyTexture(gl, w, h, gl.RGBA16F, gl.RGBA, gl.HALF_FLOAT);
      const fbo = createFBO(gl, tex.texture);
      return { tex, fbo, w, h };
    }
    return createLevel(w, h);
  }

  function freeLevel(level: PyramidLevel) {
    level.fbo.dispose();
    level.tex.dispose();
  }

  function freeLevels(levels: PyramidLevel[]) {
    levels.forEach(freeLevel);
  }

  /** Render source texture downsampled 2x into dst FBO */
  function downsample(srcTex: WebGLTexture, srcW: number, srcH: number, dstFBO: WebGLFramebuffer, dstW: number, dstH: number) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
    gl.viewport(0, 0, dstW, dstH);
    gl.useProgram(downsampleProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(dsLoc.uTex, 0);
    gl.uniform2f(dsLoc.uTexelSize, 1 / srcW, 1 / srcH);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Render source texture upsampled 2x into dst FBO */
  function upsample(srcTex: WebGLTexture, dstFBO: WebGLFramebuffer, dstW: number, dstH: number) {
    gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
    gl.viewport(0, 0, dstW, dstH);
    gl.useProgram(upsampleProg);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(usLoc.uTex, 0);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  /** Build Gaussian pyramid from source texture. Returns array of levels [0=original, 1=half, ...]. */
  function buildGaussianPyramid(srcTex: WebGLTexture, w: number, h: number, levels: number): PyramidLevel[] {
    const pyramid: PyramidLevel[] = [];
    // Level 0: copy of source
    const l0 = createLevel(w, h);
    gl.bindFramebuffer(gl.FRAMEBUFFER, l0.fbo.fbo);
    gl.viewport(0, 0, w, h);
    gl.useProgram(upsampleProg); // Just copies
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, srcTex);
    gl.uniform1i(usLoc.uTex, 0);
    gl.bindVertexArray(quadVAO);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    gl.bindVertexArray(null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    pyramid.push(l0);

    for (let i = 1; i < levels; i++) {
      const prev = pyramid[i - 1];
      const nw = Math.max(1, Math.floor(prev.w / 2));
      const nh = Math.max(1, Math.floor(prev.h / 2));
      const level = createLevel(nw, nh);
      downsample(prev.tex.texture, prev.w, prev.h, level.fbo.fbo, nw, nh);
      pyramid.push(level);
    }

    return pyramid;
  }

  /** Build Laplacian pyramid from Gaussian pyramid. Returns Laplacian levels [0..L-2] + residual [L-1]. */
  function buildLaplacianPyramid(gaussian: PyramidLevel[]): PyramidLevel[] {
    const lap: PyramidLevel[] = [];

    for (let i = 0; i < gaussian.length - 1; i++) {
      const cur = gaussian[i];
      const next = gaussian[i + 1];

      // Upsample next level to current size
      const upsampled = createLevel(cur.w, cur.h);
      upsample(next.tex.texture, upsampled.fbo.fbo, cur.w, cur.h);

      // Compute Laplacian: current - upsampled
      const lapLevel = createLapLevel(cur.w, cur.h);
      gl.bindFramebuffer(gl.FRAMEBUFFER, lapLevel.fbo.fbo);
      gl.viewport(0, 0, cur.w, cur.h);
      gl.useProgram(laplacianProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, cur.tex.texture);
      gl.uniform1i(lapLoc.uCurrent, 0);
      gl.activeTexture(gl.TEXTURE1);
      gl.bindTexture(gl.TEXTURE_2D, upsampled.tex.texture);
      gl.uniform1i(lapLoc.uUpsampled, 1);
      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);

      freeLevel(upsampled);
      lap.push(lapLevel);
    }

    // Residual = lowest Gaussian level (copy)
    const lowest = gaussian[gaussian.length - 1];
    const residual = createLevel(lowest.w, lowest.h);
    upsample(lowest.tex.texture, residual.fbo.fbo, lowest.w, lowest.h); // copy
    lap.push(residual);

    return lap;
  }

  return {
    blend(compositeTex, newImageTex, maskTex, outputFBO, width, height, numLevels) {
      numLevels = Math.max(2, Math.min(numLevels, 7));
      gl.disable(gl.BLEND);

      // ── Phase 1: Build Gaussian pyramids for both images and mask ──
      const gaussComp = buildGaussianPyramid(compositeTex, width, height, numLevels);
      const gaussNew = buildGaussianPyramid(newImageTex, width, height, numLevels);
      const gaussMask = buildGaussianPyramid(maskTex, width, height, numLevels);

      // ── Phase 2: Build Laplacian pyramids (band-pass detail) ──
      const lapComp = buildLaplacianPyramid(gaussComp);
      const lapNew = buildLaplacianPyramid(gaussNew);

      // ── Phase 3: Blend Laplacian levels using downsampled mask ──
      const lapBlended: PyramidLevel[] = [];
      for (let i = 0; i < numLevels; i++) {
        const lc = lapComp[i];
        const ln = lapNew[i];
        const m = gaussMask[i];
        const blended = createLapLevel(lc.w, lc.h);

        gl.bindFramebuffer(gl.FRAMEBUFFER, blended.fbo.fbo);
        gl.viewport(0, 0, lc.w, lc.h);
        gl.useProgram(blendLapProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, lc.tex.texture);
        gl.uniform1i(blLoc.uLapComp, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, ln.tex.texture);
        gl.uniform1i(blLoc.uLapNew, 1);
        gl.activeTexture(gl.TEXTURE2);
        gl.bindTexture(gl.TEXTURE_2D, m.tex.texture);
        gl.uniform1i(blLoc.uMask, 2);
        gl.bindVertexArray(quadVAO);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        lapBlended.push(blended);
      }

      // ── Phase 4: Reconstruct from blended Laplacian pyramid ──
      // Start from lowest level (residual) and work up
      let current = lapBlended[numLevels - 1]; // residual is already a full image
      for (let i = numLevels - 2; i >= 0; i--) {
        const lapLevel = lapBlended[i];
        const targetW = lapLevel.w;
        const targetH = lapLevel.h;

        // Upsample current to lapLevel size
        const upsampled = createLapLevel(targetW, targetH);
        upsample(current.tex.texture, upsampled.fbo.fbo, targetW, targetH);

        // Reconstruct: upsampled + laplacian_detail
        // Use float for intermediate levels; final level (i==0) writes to RGBA8 output
        const result = (i > 0) ? createLapLevel(targetW, targetH) : createLevel(targetW, targetH);
        gl.bindFramebuffer(gl.FRAMEBUFFER, result.fbo.fbo);
        gl.viewport(0, 0, targetW, targetH);
        gl.useProgram(reconstructProg);
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_2D, lapLevel.tex.texture);
        gl.uniform1i(recLoc.uLap, 0);
        gl.activeTexture(gl.TEXTURE1);
        gl.bindTexture(gl.TEXTURE_2D, upsampled.tex.texture);
        gl.uniform1i(recLoc.uUp, 1);
        gl.bindVertexArray(quadVAO);
        gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
        gl.bindVertexArray(null);
        gl.bindFramebuffer(gl.FRAMEBUFFER, null);

        freeLevel(upsampled);
        if (current !== lapBlended[numLevels - 1]) freeLevel(current);
        current = result;
      }

      // Copy result to output
      if (outputFBO === null) {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, current.fbo.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
        gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
      } else {
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, current.fbo.fbo);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, outputFBO);
        gl.blitFramebuffer(0, 0, width, height, 0, 0, width, height, gl.COLOR_BUFFER_BIT, gl.LINEAR);
        gl.bindFramebuffer(gl.READ_FRAMEBUFFER, null);
        gl.bindFramebuffer(gl.DRAW_FRAMEBUFFER, null);
      }

      // Cleanup
      if (current !== lapBlended[numLevels - 1]) freeLevel(current);
      freeLevels(gaussComp);
      freeLevels(gaussNew);
      freeLevels(gaussMask);
      freeLevels(lapComp);
      freeLevels(lapNew);
      freeLevels(lapBlended);
    },

    dispose() {
      gl.deleteProgram(downsampleProg);
      gl.deleteProgram(upsampleProg);
      gl.deleteProgram(laplacianProg);
      gl.deleteProgram(blendLapProg);
      gl.deleteProgram(reconstructProg);
      gl.deleteVertexArray(quadVAO);
      gl.deleteBuffer(quadVBO);
    },
  };
}
