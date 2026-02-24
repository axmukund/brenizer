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
uniform float u_gain;  // exposure gain
uniform float u_alpha;
out vec4 fragColor;
void main() {
  vec4 c = texture(u_texture, v_uv);
  c.rgb *= u_gain;
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

// ── types ────────────────────────────────────────────────

export interface MeshData {
  positions: Float32Array;  // [x0,y0, x1,y1, ...] in output-space coords
  uvs: Float32Array;        // [u0,v0, u1,v1, ...]
  indices: Uint32Array;     // triangle indices
}

export interface WarpRenderer {
  drawTexture(texture: WebGLTexture, width: number, height: number): void;
  drawMesh(
    texture: WebGLTexture,
    mesh: MeshData,
    viewMatrix: Float32Array, // 3x3 col-major
    gain?: number,
    alpha?: number,
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
  const sy = -(2 * scale) / canvasH; // flip y
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
    uGain: gl.getUniformLocation(warpProg, 'u_gain'),
    uAlpha: gl.getUniformLocation(warpProg, 'u_alpha'),
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

    drawMesh(texture, mesh, viewMatrix, gain = 1.0, alpha = 1.0) {
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
      gl.uniform1f(wLoc.uGain, gain);
      gl.uniform1f(wLoc.uAlpha, alpha);

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
