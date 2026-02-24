/** Capabilities detection module.
 *  Probes the browser/device for GPU, memory, concurrency, etc.
 *  Results are used by the preset/mode selector. */

export interface Capabilities {
  isMobile: boolean;
  deviceMemory: number | null;      // GB, null if API unavailable
  hardwareConcurrency: number;
  webgl2: boolean;
  floatFBO: boolean;
  maxTextureSize: number;
  webgpuAvailable: boolean;
  crossOriginIsolated: boolean;
  glRenderer: string;
  glVendor: string;
}

/** Quick mobile heuristic using UA + screen size. */
function detectMobile(): boolean {
  const ua = navigator.userAgent || '';
  if (/Mobi|Android|iPhone|iPad|iPod/i.test(ua)) return true;
  if (typeof screen !== 'undefined' && screen.width < 768) return true;
  // iPadOS fakes desktop UA but has touch
  if (navigator.maxTouchPoints > 1 && /Mac/.test(ua)) return true;
  return false;
}

/** Probe WebGL2 capabilities. Returns partial caps. */
function probeGL(): Pick<Capabilities, 'webgl2' | 'floatFBO' | 'maxTextureSize' | 'glRenderer' | 'glVendor'> {
  const canvas = document.createElement('canvas');
  const gl = canvas.getContext('webgl2') as WebGL2RenderingContext | null;
  if (!gl) {
    return { webgl2: false, floatFBO: false, maxTextureSize: 2048, glRenderer: '', glVendor: '' };
  }
  const floatExt = gl.getExtension('EXT_color_buffer_float');
  const maxTex = gl.getParameter(gl.MAX_TEXTURE_SIZE) as number;

  let renderer = '';
  let vendor = '';
  const dbg = gl.getExtension('WEBGL_debug_renderer_info');
  if (dbg) {
    renderer = gl.getParameter(dbg.UNMASKED_RENDERER_WEBGL) || '';
    vendor = gl.getParameter(dbg.UNMASKED_VENDOR_WEBGL) || '';
  }
  // Clean up
  const loseCtx = gl.getExtension('WEBGL_lose_context');
  if (loseCtx) loseCtx.loseContext();

  return {
    webgl2: true,
    floatFBO: !!floatExt,
    maxTextureSize: maxTex,
    glRenderer: renderer,
    glVendor: vendor,
  };
}

/** Detect all capabilities. Call once after DOM is ready. */
export async function detectCapabilities(): Promise<Capabilities> {
  const glCaps = probeGL();

  const caps: Capabilities = {
    isMobile: detectMobile(),
    deviceMemory: (navigator as any).deviceMemory ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency || 2,
    webgpuAvailable: 'gpu' in navigator,
    crossOriginIsolated: self.crossOriginIsolated ?? false,
    ...glCaps,
  };

  return caps;
}

/** Human-readable summary for UI display. */
export function capsSummary(c: Capabilities): { label: string; status: 'ok' | 'warn' | 'no' }[] {
  const items: { label: string; status: 'ok' | 'warn' | 'no' }[] = [];

  items.push({ label: c.isMobile ? 'Mobile' : 'Desktop', status: c.isMobile ? 'warn' : 'ok' });
  items.push({ label: `WebGL2`, status: c.webgl2 ? 'ok' : 'no' });
  items.push({ label: `Float FBO`, status: c.floatFBO ? 'ok' : 'warn' });
  items.push({
    label: `Max Tex ${c.maxTextureSize}`,
    status: c.maxTextureSize >= 8192 ? 'ok' : 'warn',
  });
  items.push({ label: `WebGPU`, status: c.webgpuAvailable ? 'ok' : 'warn' });
  items.push({
    label: `Cores ${c.hardwareConcurrency}`,
    status: c.hardwareConcurrency >= 4 ? 'ok' : 'warn',
  });
  if (c.deviceMemory !== null) {
    items.push({
      label: `${c.deviceMemory} GB RAM`,
      status: c.deviceMemory >= 4 ? 'ok' : 'warn',
    });
  }
  items.push({ label: `COOP/COEP`, status: c.crossOriginIsolated ? 'ok' : 'warn' });

  if (c.glRenderer) {
    items.push({ label: c.glRenderer.slice(0, 30), status: 'ok' });
  }
  return items;
}
