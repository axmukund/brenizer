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
  wasmSimd: boolean;
  wasmThreads: boolean;
  browserFamily: 'chromium' | 'firefox' | 'safari' | 'other';
  seamAccelerationTier: 'desktopTurbo' | 'webgpu' | 'webglGrid' | 'legacyCpu';
}

const VALID_SEAM_TIERS = new Set(['desktopTurbo', 'webgpu', 'webglGrid', 'legacyCpu']);

function detectBrowserFamily(): Capabilities['browserFamily'] {
  const ua = navigator.userAgent || '';
  if (/Chrome|Chromium|Edg\//i.test(ua) && !/OPR|Opera/i.test(ua)) return 'chromium';
  if (/Firefox\//i.test(ua)) return 'firefox';
  if (/Safari\//i.test(ua) && !/Chrome|Chromium|Edg\//i.test(ua)) return 'safari';
  return 'other';
}

function detectWasmSimd(): boolean {
  try {
    return WebAssembly.validate(new Uint8Array([
      0x00, 0x61, 0x73, 0x6d,
      0x01, 0x00, 0x00, 0x00,
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
      0x03, 0x02, 0x01, 0x00,
      0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0x0b,
    ]));
  } catch {
    return false;
  }
}

function detectWasmThreads(crossOriginIsolated: boolean): boolean {
  return crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined';
}

function detectSeamTierOverride(): Capabilities['seamAccelerationTier'] | null {
  try {
    const qs = new URLSearchParams(window.location.search);
    const queryOverride = qs.get('seamTier');
    const storedOverride = window.localStorage.getItem('brenizer.seamTier');
    const raw = queryOverride || storedOverride;
    if (raw && VALID_SEAM_TIERS.has(raw)) {
      return raw as Capabilities['seamAccelerationTier'];
    }
  } catch {
    // Ignore storage/query failures and use auto detection.
  }
  return null;
}

function resolveSeamAccelerationTier(caps: Omit<Capabilities, 'seamAccelerationTier'>): Capabilities['seamAccelerationTier'] {
  const override = detectSeamTierOverride();
  if (override) return override;
  if (!caps.webgl2) return 'legacyCpu';
  if (
    caps.browserFamily === 'chromium'
    && caps.webgpuAvailable
    && caps.crossOriginIsolated
    && caps.hardwareConcurrency >= 8
    && (caps.deviceMemory ?? 0) >= 8
  ) {
    return 'desktopTurbo';
  }
  if (caps.webgpuAvailable) return 'webgpu';
  return 'webglGrid';
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
  const crossOriginIsolated = self.crossOriginIsolated ?? false;

  const partialCaps: Omit<Capabilities, 'seamAccelerationTier'> = {
    isMobile: detectMobile(),
    deviceMemory: (navigator as any).deviceMemory ?? null,
    hardwareConcurrency: navigator.hardwareConcurrency || 2,
    webgpuAvailable: 'gpu' in navigator,
    crossOriginIsolated,
    ...glCaps,
    wasmSimd: detectWasmSimd(),
    wasmThreads: detectWasmThreads(crossOriginIsolated),
    browserFamily: detectBrowserFamily(),
  };

  const caps: Capabilities = {
    ...partialCaps,
    seamAccelerationTier: resolveSeamAccelerationTier(partialCaps),
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
    label: `Seam ${c.seamAccelerationTier}`,
    status: c.seamAccelerationTier === 'legacyCpu' ? 'warn' : 'ok',
  });
  items.push({
    label: `Cores ${c.hardwareConcurrency}`,
    status: c.hardwareConcurrency >= 4 ? 'ok' : 'warn',
  });
  items.push({ label: `WASM SIMD`, status: c.wasmSimd ? 'ok' : 'warn' });
  items.push({ label: `WASM Threads`, status: c.wasmThreads ? 'ok' : 'warn' });
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
