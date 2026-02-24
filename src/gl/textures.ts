/** Texture creation and management helpers for WebGL2. */

export interface ManagedTexture {
  texture: WebGLTexture;
  width: number;
  height: number;
  dispose(): void;
}

/**
 * Create a RGBA8 texture from an ImageBitmap, HTMLImageElement, or canvas.
 */
export function createTextureFromImage(
  gl: WebGL2RenderingContext,
  source: TexImageSource,
  width: number,
  height: number,
): ManagedTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error('Failed to create texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, width, height, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, gl.RGBA, gl.UNSIGNED_BYTE, source);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return {
    texture: tex,
    width,
    height,
    dispose() { gl.deleteTexture(tex); },
  };
}

/** Create an empty RGBA8 texture of given dimensions. */
export function createEmptyTexture(
  gl: WebGL2RenderingContext,
  width: number,
  height: number,
  internalFormat: number = gl.RGBA8,
  format: number = gl.RGBA,
  type: number = gl.UNSIGNED_BYTE,
): ManagedTexture {
  const tex = gl.createTexture();
  if (!tex) throw new Error('Failed to create texture');
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texImage2D(gl.TEXTURE_2D, 0, internalFormat, width, height, 0, format, type, null);
  gl.bindTexture(gl.TEXTURE_2D, null);

  return {
    texture: tex,
    width,
    height,
    dispose() { gl.deleteTexture(tex); },
  };
}
