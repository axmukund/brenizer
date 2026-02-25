/**
 * Mesh generation and warp-mesh rendering.
 * Supports both identity (pass-through) mesh and warped meshes
 * for APAP-style local projective correction.
 */

import { createProgram } from './programs';

// ── inline shaders (simple textured mesh) ──────────────

const WARP_VERT = `#version 300 es
precision highp float;
in vec2 a_position;   // warped vertex position in output coords
in vec2 a_texcoord;   // UV in source image [0,1]
uniform mat3 u_viewMatrix; // maps output coords to clip space
out vec2 v_uv;
void main() {
  vec3 p = u_viewMatrix * vec3(a_position, 1.0);
  gl_Position = vec4(p.xy, 0.0, 1.0);
  v_uv = a_texcoord;
}
`;

const WARP_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
uniform vec3 u_gainRGB;  // per-channel RGB exposure gain
uniform float u_alpha;
// Vignetting correction: V(r) = 1 + a*r² + b*r⁴ (PTGui polynomial model)
uniform float u_vigA;     // vignette coefficient a
uniform float u_vigB;     // vignette coefficient b
// HDR tone mapping for extreme exposure handling
uniform float u_toneMap;  // 0 = off, 1 = Reinhard tone mapping
out vec4 fragColor;
void main() {
  vec4 c = texture(u_texture, v_uv);

  // ── Vignetting correction ──────────────────────
  // Undo radial darkening: divide by V(r) where r = distance from center.
  // V(r) = 1 + a*r² + b*r⁴ estimated per-image (PTGui polynomial model).
  vec2 centered = v_uv - 0.5;
  float r2 = dot(centered, centered) * 4.0; // normalised r² ∈ [0, ~1]
  float r4 = r2 * r2;
  float vignette = 1.0 + u_vigA * r2 + u_vigB * r4;
  // Prevent division by near-zero
  vignette = max(vignette, 0.1);
  c.rgb /= vignette;

  // ── Per-channel exposure gain ──────────────────
  c.rgb *= u_gainRGB;

  // ── HDR tone mapping (extended Reinhard) ───────
  // Handles extreme exposure by compressing highlights while preserving
  // shadow detail. Uses Reinhard global operator: L_d = L(1 + L/L²_white) / (1 + L)
  // where L_white is the maximum displayable luminance.
  if (u_toneMap > 0.5) {
    float Lwhite2 = 4.0; // white point squared
    c.rgb = c.rgb * (1.0 + c.rgb / Lwhite2) / (1.0 + c.rgb);
  }

  c.a *= u_alpha;
  fragColor = c;
}
`;

// ── fullscreen quad shaders ─────────────────────────────

const FULLSCREEN_VERT = `#version 300 es
precision highp float;
in vec2 a_position;
out vec2 v_uv;
void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FULLSCREEN_FRAG = `#version 300 es
precision highp float;
in vec2 v_uv;
uniform sampler2D u_texture;
out vec4 fragColor;
void main() {
  fragColor = texture(u_texture, v_uv);
}
`;

// ── keypoint overlay shaders ────────────────────────────

const POINT_VERT = `#version 300 es
precision highp float;
in vec2 a_position;
uniform mat3 u_viewMatrix;
uniform float u_pointSize;
out vec2 v_pos;
void main() {
  vec3 p = u_viewMatrix * vec3(a_position, 1.0);
  gl_Position = vec4(p.xy, 0.0, 1.0);
  gl_PointSize = u_pointSize;
  v_pos = a_position;
}
`;

const POINT_FRAG = `#version 300 es
precision highp float;
uniform vec4 u_color;
out vec4 fragColor;
void main() {
  // Draw circle with smooth edge
  vec2 pc = gl_PointCoord - 0.5;
  float dist = length(pc);
  if (dist > 0.5) discard;
  float alpha = smoothstep(0.5, 0.35, dist);
  fragColor = vec4(u_color.rgb, u_color.a * alpha);
}
`;

// ── types ────────────────────────────────────────────────

export interface MeshData {
  positions: Float32Array;  // [x0,y0, x1,y1, ...] in output-space coords
  uvs: Float32Array;        // [u0,v0, u1,v1, ...]
  indices: Uint32Array;     // triangle indices
}

export interface WarpRenderer {
  drawTexture(texture: WebGLTexture, width: number, height: number): void;
  /**
   * Draw a warped mesh with per-channel exposure compensation.
   * @param gain Exposure gain — scalar (applied uniformly to RGB) or
   *             [R, G, B] tuple for per-channel correction. Defaults to 1.0.
   * @param alpha Opacity [0-1] for blending. Defaults to 1.0.
   * @param vigA Vignetting coefficient a (polynomial radial model). Default 0.
   * @param vigB Vignetting coefficient b (polynomial radial model). Default 0.
   * @param toneMap Enable Reinhard HDR tone mapping for extreme exposure. Default false.
   */
  drawMesh(
    texture: WebGLTexture,
    mesh: MeshData,
    viewMatrix: Float32Array, // 3x3 col-major
    gain?: number | [number, number, number],  // scalar or per-channel [R,G,B]
    alpha?: number,
    vigA?: number,
    vigB?: number,
    toneMap?: boolean,
  ): void;
  dispose(): void;
}

export interface KeypointRenderer {
  /**
   * Draw keypoints as coloured dots over the current framebuffer.
   * @param keypoints  Float32Array of [x0,y0,x1,y1,...] in image (pixel) coords
   * @param viewMatrix 3x3 col-major mapping image coords → clip space
   * @param color      RGBA colour [0-1]
   * @param pointSize  Size in pixels
   */
  drawKeypoints(
    keypoints: Float32Array,
    viewMatrix: Float32Array,
    color?: [number, number, number, number],
    pointSize?: number,
  ): void;
  dispose(): void;
}

/**
 * Generate an identity (no-warp) mesh for an image.
 * Grid of gridW × gridH quads covering [0, imgW] × [0, imgH].
 */
export function createIdentityMesh(
  imgW: number,
  imgH: number,
  gridW: number = 1,
  gridH: number = 1,
): MeshData {
  const cols = gridW + 1;
  const rows = gridH + 1;
  const positions = new Float32Array(cols * rows * 2);
  const uvs = new Float32Array(cols * rows * 2);
  let vi = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const u = c / gridW;
      const v = r / gridH;
      positions[vi] = u * imgW;
      positions[vi + 1] = v * imgH;
      uvs[vi] = u;
      uvs[vi + 1] = v;
      vi += 2;
    }
  }
  // Triangulate
  const indices: number[] = [];
  for (let r = 0; r < gridH; r++) {
    for (let c = 0; c < gridW; c++) {
      const tl = r * cols + c;
      const tr = tl + 1;
      const bl = tl + cols;
      const br = bl + 1;
      indices.push(tl, bl, tr, tr, bl, br);
    }
  }
  return {
    positions,
    uvs,
    indices: new Uint32Array(indices),
  };
}

/**
 * Build a view matrix that maps [0, w] × [0, h] to clip space [-1, 1]
 * with optional pan/zoom.
 */
export function makeViewMatrix(
  canvasW: number,
  canvasH: number,
  panX: number = 0,
  panY: number = 0,
  zoom: number = 1,
  contentW?: number,
  contentH?: number,
): Float32Array {
  const cw = contentW ?? canvasW;
  const ch = contentH ?? canvasH;
  // Fit content into canvas maintaining aspect ratio
  const scale = Math.min(canvasW / cw, canvasH / ch) * zoom;
  const sx = (2 * scale) / canvasW;
  const sy = -(2 * scale) / canvasH; // flip y: canvas y-down → GL clip-space y-up
  const tx = -cw * scale / canvasW + panX * 2 / canvasW;
  const ty = ch * scale / canvasH + panY * 2 / canvasH;
  // Column-major 3×3
  return new Float32Array([
    sx, 0, 0,
    0, sy, 0,
    tx, ty, 1,
  ]);
}

/** Create a WarpRenderer using the provided GL context. */
export function createWarpRenderer(gl: WebGL2RenderingContext): WarpRenderer {
  // Compile programs
  const warpProg = createProgram(gl, WARP_VERT, WARP_FRAG);
  const fullProg = createProgram(gl, FULLSCREEN_VERT, FULLSCREEN_FRAG);

  // Warp program locations
  const wLoc = {
    aPos: gl.getAttribLocation(warpProg, 'a_position'),
    aTex: gl.getAttribLocation(warpProg, 'a_texcoord'),
    uView: gl.getUniformLocation(warpProg, 'u_viewMatrix'),
    uTex: gl.getUniformLocation(warpProg, 'u_texture'),
    uGainRGB: gl.getUniformLocation(warpProg, 'u_gainRGB'),
    uAlpha: gl.getUniformLocation(warpProg, 'u_alpha'),
    uVigA: gl.getUniformLocation(warpProg, 'u_vigA'),
    uVigB: gl.getUniformLocation(warpProg, 'u_vigB'),
    uToneMap: gl.getUniformLocation(warpProg, 'u_toneMap'),
  };

  // Fullscreen program locations
  const fLoc = {
    aPos: gl.getAttribLocation(fullProg, 'a_position'),
    uTex: gl.getUniformLocation(fullProg, 'u_texture'),
  };

  // Fullscreen quad VAO
  const quadVAO = gl.createVertexArray()!;
  const quadVBO = gl.createBuffer()!;
  gl.bindVertexArray(quadVAO);
  gl.bindBuffer(gl.ARRAY_BUFFER, quadVBO);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1, 1,1]), gl.STATIC_DRAW);
  gl.enableVertexAttribArray(fLoc.aPos);
  gl.vertexAttribPointer(fLoc.aPos, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);

  // Mesh VAO (will be updated per draw)
  const meshVAO = gl.createVertexArray()!;
  const posBuf = gl.createBuffer()!;
  const uvBuf = gl.createBuffer()!;
  const idxBuf = gl.createBuffer()!;

  return {
    drawTexture(texture, width, height) {
      const cw = gl.canvas.width;
      const ch = gl.canvas.height;
      gl.viewport(0, 0, cw, ch);
      gl.clearColor(0, 0, 0, 1);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.useProgram(fullProg);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(fLoc.uTex, 0);
      gl.bindVertexArray(quadVAO);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      gl.bindVertexArray(null);
    },

    drawMesh(texture, mesh, viewMatrix, gain: number | [number, number, number] = 1.0, alpha = 1.0, vigA = 0, vigB = 0, toneMap = false) {
      gl.useProgram(warpProg);

      // Upload mesh data
      gl.bindVertexArray(meshVAO);

      gl.bindBuffer(gl.ARRAY_BUFFER, posBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.positions, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(wLoc.aPos);
      gl.vertexAttribPointer(wLoc.aPos, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, uvBuf);
      gl.bufferData(gl.ARRAY_BUFFER, mesh.uvs, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(wLoc.aTex);
      gl.vertexAttribPointer(wLoc.aTex, 2, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, idxBuf);
      gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, mesh.indices, gl.DYNAMIC_DRAW);

      // Uniforms
      gl.uniformMatrix3fv(wLoc.uView, false, viewMatrix);
      gl.activeTexture(gl.TEXTURE0);
      gl.bindTexture(gl.TEXTURE_2D, texture);
      gl.uniform1i(wLoc.uTex, 0);
      // Per-channel RGB gain: accept scalar or [R,G,B] tuple
      if (typeof gain === 'number') {
        gl.uniform3f(wLoc.uGainRGB, gain, gain, gain);
      } else {
        gl.uniform3f(wLoc.uGainRGB, gain[0], gain[1], gain[2]);
      }

      gl.uniform1f(wLoc.uAlpha, alpha);
      // Vignetting correction coefficients (PTGui polynomial radial model)
      gl.uniform1f(wLoc.uVigA, vigA);
      gl.uniform1f(wLoc.uVigB, vigB);
      // HDR tone mapping (Reinhard) for extreme exposure handling
      gl.uniform1f(wLoc.uToneMap, toneMap ? 1.0 : 0.0);

      gl.drawElements(gl.TRIANGLES, mesh.indices.length, gl.UNSIGNED_INT, 0);
      gl.bindVertexArray(null);
    },

    dispose() {
      gl.deleteProgram(warpProg);
      gl.deleteProgram(fullProg);
      gl.deleteVertexArray(quadVAO);
      gl.deleteBuffer(quadVBO);
      gl.deleteVertexArray(meshVAO);
      gl.deleteBuffer(posBuf);
      gl.deleteBuffer(uvBuf);
      gl.deleteBuffer(idxBuf);
    },
  };
}

/** Create a renderer for drawing keypoint overlays as coloured dots. */
export function createKeypointRenderer(gl: WebGL2RenderingContext): KeypointRenderer {
  const prog = createProgram(gl, POINT_VERT, POINT_FRAG);
  const aPos = gl.getAttribLocation(prog, 'a_position');
  const uView = gl.getUniformLocation(prog, 'u_viewMatrix');
  const uColor = gl.getUniformLocation(prog, 'u_color');
  const uSize = gl.getUniformLocation(prog, 'u_pointSize');

  const vao = gl.createVertexArray()!;
  const vbo = gl.createBuffer()!;

  return {
    drawKeypoints(
      keypoints: Float32Array,
      viewMatrix: Float32Array,
      color: [number, number, number, number] = [0, 1, 0, 0.85],
      pointSize = 6,
    ) {
      if (keypoints.length < 2) return;

      gl.enable(gl.BLEND);
      gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

      gl.useProgram(prog);

      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
      gl.bufferData(gl.ARRAY_BUFFER, keypoints, gl.DYNAMIC_DRAW);
      gl.enableVertexAttribArray(aPos);
      gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

      gl.uniformMatrix3fv(uView, false, viewMatrix);
      gl.uniform4fv(uColor, color);
      gl.uniform1f(uSize, pointSize);

      gl.drawArrays(gl.POINTS, 0, keypoints.length / 2);
      gl.bindVertexArray(null);
      gl.disable(gl.BLEND);
    },

    dispose() {
      gl.deleteProgram(prog);
      gl.deleteVertexArray(vao);
      gl.deleteBuffer(vbo);
    },
  };
}
