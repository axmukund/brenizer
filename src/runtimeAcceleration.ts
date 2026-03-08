export type CrossOriginIsolationMode = 'none' | 'coi-serviceworker' | 'headers';

const TURBO_MODE_QUERY_KEY = 'turboMode';
const TURBO_MODE_STORAGE_KEY = 'brenizer.turboMode';
const COI_ATTEMPT_KEY = 'brenizer.coi.attempted';
const COI_FAILED_KEY = 'brenizer.coi.failed';
const COI_SOURCE_KEY = 'brenizer.coi.source';
const COI_RELOAD_COUNT_KEY = 'brenizer.coi.reloadCount';
const COI_SW_FILENAME = 'coi-serviceworker.js';
const COI_MAX_AUTO_RELOADS = 2;
const COI_READY_WAIT_MS = 4000;

function parseBooleanish(raw: string | null): boolean | null {
  if (raw === null) return null;
  if (/^(1|true|on|yes)$/i.test(raw)) return true;
  if (/^(0|false|off|no)$/i.test(raw)) return false;
  return null;
}

function detectMobileLike(): boolean {
  const ua = navigator.userAgent || '';
  if (/Mobi|Android|iPhone|iPad|iPod/i.test(ua)) return true;
  if (typeof screen !== 'undefined' && screen.width < 768) return true;
  if (navigator.maxTouchPoints > 1 && /Mac/.test(ua)) return true;
  return false;
}

function isLocalDevelopmentHost(): boolean {
  return /^(localhost|127\.0\.0\.1|\[::1\])$/i.test(window.location.hostname);
}

function getTurboModeQueryOverride(): boolean | null {
  try {
    const qs = new URLSearchParams(window.location.search);
    return parseBooleanish(qs.get(TURBO_MODE_QUERY_KEY));
  } catch {
    return null;
  }
}

function getTurboModeStoredOverride(): boolean | null {
  try {
    return parseBooleanish(window.localStorage.getItem(TURBO_MODE_STORAGE_KEY));
  } catch {
    return null;
  }
}

function getCoiServiceWorkerUrl(): URL {
  return new URL(`./${COI_SW_FILENAME}`, window.location.href);
}

function getCoiServiceWorkerScope(): string {
  return new URL('./', window.location.href).pathname;
}

function clearCoiMarkers(): void {
  try {
    window.sessionStorage.removeItem(COI_ATTEMPT_KEY);
    window.sessionStorage.removeItem(COI_FAILED_KEY);
    window.sessionStorage.removeItem(COI_SOURCE_KEY);
    window.sessionStorage.removeItem(COI_RELOAD_COUNT_KEY);
  } catch {
    // Ignore storage failures; runtime detection will still work.
  }
}

function markCoiAttempted(): void {
  try {
    window.sessionStorage.setItem(COI_ATTEMPT_KEY, '1');
    window.sessionStorage.setItem(COI_SOURCE_KEY, 'coi-serviceworker');
    window.sessionStorage.removeItem(COI_FAILED_KEY);
  } catch {
    // Ignore storage failures; service worker registration can still proceed.
  }
}

function markCoiFailed(): void {
  try {
    window.sessionStorage.setItem(COI_FAILED_KEY, '1');
  } catch {
    // Ignore storage failures; fallback path remains usable.
  }
}

function getCoiReloadCount(): number {
  try {
    const raw = Number(window.sessionStorage.getItem(COI_RELOAD_COUNT_KEY) || '0');
    return Number.isFinite(raw) && raw >= 0 ? Math.floor(raw) : 0;
  } catch {
    return 0;
  }
}

function setCoiReloadCount(value: number): void {
  try {
    window.sessionStorage.setItem(COI_RELOAD_COUNT_KEY, String(Math.max(0, Math.floor(value))));
  } catch {
    // Ignore storage failures and continue with best-effort behavior.
  }
}

function getActiveCoiControllerScriptUrl(): string {
  if (!('serviceWorker' in navigator)) return '';
  return navigator.serviceWorker.controller?.scriptURL || '';
}

function hasMatchingCoiController(): boolean {
  const controllerScript = getActiveCoiControllerScriptUrl();
  return controllerScript.endsWith(`/${COI_SW_FILENAME}`) || controllerScript.endsWith(COI_SW_FILENAME);
}

async function waitForCoiServiceWorkerReady(timeoutMs = COI_READY_WAIT_MS): Promise<boolean> {
  if (!('serviceWorker' in navigator)) return false;
  if (hasMatchingCoiController()) return true;

  return await new Promise<boolean>((resolve) => {
    let settled = false;
    let timeoutId = 0;

    const finish = (value: boolean): void => {
      if (settled) return;
      settled = true;
      navigator.serviceWorker.removeEventListener('controllerchange', onControllerChange);
      if (timeoutId !== 0) window.clearTimeout(timeoutId);
      resolve(value);
    };

    const onControllerChange = (): void => {
      finish(hasMatchingCoiController());
    };

    navigator.serviceWorker.addEventListener('controllerchange', onControllerChange);
    navigator.serviceWorker.ready
      .then(() => finish(true))
      .catch(() => undefined);
    timeoutId = window.setTimeout(() => finish(false), timeoutMs);
  });
}

function requestCoiReload(): { reloading: true; active: false } {
  setCoiReloadCount(getCoiReloadCount() + 1);
  window.location.reload();
  return { reloading: true, active: false };
}

async function findCoiServiceWorkerRegistration(): Promise<ServiceWorkerRegistration | null> {
  if (!('serviceWorker' in navigator)) return null;
  const swUrl = getCoiServiceWorkerUrl().toString();
  const regs = await navigator.serviceWorker.getRegistrations();
  return regs.find((reg) => {
    const scriptUrl = reg.active?.scriptURL || reg.waiting?.scriptURL || reg.installing?.scriptURL || '';
    return scriptUrl === swUrl;
  }) ?? null;
}

function shouldEnableTurboByDefault(): boolean {
  if (import.meta.env.DEV) return false;
  if (isLocalDevelopmentHost()) return false;
  if (detectMobileLike()) return false;
  return true;
}

export function getTurboModePreference(): boolean {
  const queryOverride = getTurboModeQueryOverride();
  if (queryOverride !== null) return queryOverride;
  const storedOverride = getTurboModeStoredOverride();
  if (storedOverride !== null) return storedOverride;
  return shouldEnableTurboByDefault();
}

export function persistTurboModePreference(enabled: boolean): void {
  try {
    window.localStorage.setItem(TURBO_MODE_STORAGE_KEY, enabled ? '1' : '0');
  } catch {
    // Ignore storage failures and keep runtime preference in-memory only.
  }
}

export function detectCrossOriginIsolationMode(crossOriginIsolated: boolean): CrossOriginIsolationMode {
  if (!crossOriginIsolated) return 'none';
  try {
    if (window.sessionStorage.getItem(COI_SOURCE_KEY) === 'coi-serviceworker') {
      return 'coi-serviceworker';
    }
  } catch {
    // Ignore storage failures and fall back to controller detection.
  }
  if ('serviceWorker' in navigator) {
    const controllerScript = navigator.serviceWorker.controller?.scriptURL || '';
    if (controllerScript.endsWith(`/${COI_SW_FILENAME}`) || controllerScript.endsWith(COI_SW_FILENAME)) {
      return 'coi-serviceworker';
    }
  }
  return 'headers';
}

export async function prepareTurboModeRuntime(enabled: boolean): Promise<{ reloading: boolean; active: boolean; reason?: string }> {
  if (!window.isSecureContext || !('serviceWorker' in navigator)) {
    if (!enabled) clearCoiMarkers();
    return { reloading: false, active: false, reason: 'service-worker-unavailable' };
  }

  if (!enabled) {
    clearCoiMarkers();
    const reg = await findCoiServiceWorkerRegistration();
    if (reg) {
      await reg.unregister();
      window.location.reload();
      return { reloading: true, active: false };
    }
    return { reloading: false, active: false };
  }

  if (self.crossOriginIsolated) {
    try {
      window.sessionStorage.removeItem(COI_FAILED_KEY);
      window.sessionStorage.removeItem(COI_RELOAD_COUNT_KEY);
      window.sessionStorage.setItem(COI_SOURCE_KEY, detectCrossOriginIsolationMode(true));
    } catch {
      // Ignore storage failures.
    }
    return { reloading: false, active: true };
  }

  const reloadCount = getCoiReloadCount();
  if (reloadCount >= COI_MAX_AUTO_RELOADS) {
    markCoiFailed();
    return { reloading: false, active: false, reason: 'coi-retry-limit-exceeded' };
  }

  try {
    markCoiAttempted();
    const registration = await findCoiServiceWorkerRegistration()
      ?? await navigator.serviceWorker.register(getCoiServiceWorkerUrl(), {
        scope: getCoiServiceWorkerScope(),
        updateViaCache: 'none',
      });
    await registration.update().catch(() => undefined);
    await waitForCoiServiceWorkerReady();
    return requestCoiReload();
  } catch (err) {
    markCoiFailed();
    return {
      reloading: false,
      active: false,
      reason: err instanceof Error ? err.message : String(err),
    };
  }
}

export async function applyTurboModePreference(enabled: boolean): Promise<{ reloading: boolean; active: boolean; reason?: string }> {
  persistTurboModePreference(enabled);
  return prepareTurboModeRuntime(enabled);
}
