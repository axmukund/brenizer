/** Framebuffer Object helpers. */

export interface ManagedFBO {
  fbo: WebGLFramebuffer;
  dispose(): void;
}

/** Create an FBO and attach a texture as COLOR_ATTACHMENT0. */
export function createFBO(
  gl: WebGL2RenderingContext,
  colorTexture: WebGLTexture
): ManagedFBO {
  const fbo = gl.createFramebuffer();
  if (!fbo) throw new Error('Failed to create FBO');
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, colorTexture, 0);
  const status = gl.checkFramebufferStatus(gl.FRAMEBUFFER);
  if (status !== gl.FRAMEBUFFER_COMPLETE) {
    gl.deleteFramebuffer(fbo);
    throw new Error(`Framebuffer incomplete: 0x${status.toString(16)}`);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  return {
    fbo,
    dispose() { gl.deleteFramebuffer(fbo); },
  };
}
