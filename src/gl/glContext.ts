/** WebGL2 context wrapper with capability checks and resource management. */

export interface GLContext {
  gl: WebGL2RenderingContext;
  canvas: HTMLCanvasElement;
  maxTextureSize: number;
  floatFBO: boolean;
  dispose(): void;
}

/**
 * Initialize a WebGL2 context on the given canvas.
 * Throws if WebGL2 is unavailable.
 */
export function createGLContext(canvas: HTMLCanvasElement): GLContext {
  const gl = canvas.getContext('webgl2', {
    alpha: false,
    antialias: false,
    premultipliedAlpha: false,
    preserveDrawingBuffer: true,
  });
  if (!gl) throw new Error('WebGL2 not available');

  const floatExt = gl.getExtension('EXT_color_buffer_float');
  const maxTextureSize = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;

  // Enable blending by default
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

  return {
    gl,
    canvas,
    maxTextureSize,
    floatFBO: !!floatExt,
    dispose() {
      const ext = gl.getExtension('WEBGL_lose_context');
      if (ext) ext.loseContext();
    },
  };
}
