/**
 * Worker manager — init, messaging, and lifecycle for all three workers.
 * Workers are lazily created and initialized on first use (when user clicks Stitch).
 */

import type {
  CVInMsg, CVOutMsg,
  DepthInMsg, DepthOutMsg,
  SeamInMsg, SeamOutMsg,
} from './workerTypes';

export interface WorkerManager {
  initAll(opts?: { enableDepth?: boolean; enableSeam?: boolean }): Promise<{ cv: boolean; depth: boolean; seam: boolean }>;
  sendCV(msg: CVInMsg, transfer?: Transferable[]): void;
  sendDepth(msg: DepthInMsg, transfer?: Transferable[]): void;
  sendSeam(msg: SeamInMsg, transfer?: Transferable[]): void;
  /** Register handler; returns an unsubscribe function. */
  onCV(handler: (msg: CVOutMsg) => void): () => void;
  onDepth(handler: (msg: DepthOutMsg) => void): () => void;
  onSeam(handler: (msg: SeamOutMsg) => void): () => void;
  /** Wait for next message of a given type from a worker. */
  waitCV<T extends CVOutMsg['type']>(type: T, timeoutMs?: number): Promise<Extract<CVOutMsg, { type: T }>>;
  waitDepth<T extends DepthOutMsg['type']>(type: T, timeoutMs?: number): Promise<Extract<DepthOutMsg, { type: T }>>;
  waitSeam<T extends SeamOutMsg['type']>(type: T, timeoutMs?: number): Promise<Extract<SeamOutMsg, { type: T }>>;
  dispose(): void;
}

type MsgHandler<T> = (msg: T) => void;

export function createWorkerManager(): WorkerManager {
  let cvWorker: Worker | null = null;
  let depthWorker: Worker | null = null;
  let seamWorker: Worker | null = null;

  const cvHandlers: MsgHandler<CVOutMsg>[] = [];
  const depthHandlers: MsgHandler<DepthOutMsg>[] = [];
  const seamHandlers: MsgHandler<SeamOutMsg>[] = [];

  function getBaseUrl(): string {
    // Use Vite's compile-time base path so GH Pages subpath deploys work
    // even when the page is opened without a trailing slash.
    return new URL(import.meta.env.BASE_URL, window.location.origin).toString();
  }

  function createCV(): Worker {
    if (cvWorker) return cvWorker;
    cvWorker = new Worker(new URL('workers/cv-worker.js', getBaseUrl()).toString());
    cvWorker.onmessage = (e) => {
      const msg = e.data as CVOutMsg;
      cvHandlers.forEach(h => h(msg));
    };
    cvWorker.onerror = (e) => {
      console.error('cv-worker error:', e);
      cvHandlers.forEach(h => h({ type: 'error', message: e.message || 'cv-worker crashed' }));
    };
    return cvWorker;
  }

  function createDepth(): Worker {
    if (depthWorker) return depthWorker;
    // depth.worker.ts is a module worker, bundled by Vite
    depthWorker = new Worker(
      new URL('./depth.worker.ts', import.meta.url),
      { type: 'module' }
    );
    depthWorker.onmessage = (e) => {
      const msg = e.data as DepthOutMsg;
      depthHandlers.forEach(h => h(msg));
    };
    depthWorker.onerror = (e) => {
      console.error('depth-worker error:', e);
      depthHandlers.forEach(h => h({ type: 'error', message: e.message || 'depth-worker crashed' }));
    };
    return depthWorker;
  }

  function createSeam(): Worker {
    if (seamWorker) return seamWorker;
    seamWorker = new Worker(new URL('workers/seam-worker.js', getBaseUrl()).toString());
    seamWorker.onmessage = (e) => {
      const msg = e.data as SeamOutMsg;
      seamHandlers.forEach(h => h(msg));
    };
    seamWorker.onerror = (e) => {
      console.error('seam-worker error:', e);
      seamHandlers.forEach(h => h({ type: 'error', message: e.message || 'seam-worker crashed' }));
    };
    return seamWorker;
  }

  function waitForMsg<T extends { type: string }>(
    handlers: MsgHandler<T>[],
    type: string,
    timeoutMs = 30000,
  ): Promise<any> {
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        cleanup();
        reject(new Error(`Timeout waiting for ${type}`));
      }, timeoutMs);

      function handler(msg: T) {
        if (msg.type === type) {
          cleanup();
          resolve(msg);
        } else if (msg.type === 'error') {
          cleanup();
          reject(new Error((msg as any).message || 'Worker error'));
        }
      }
      function cleanup() {
        clearTimeout(timer);
        const idx = handlers.indexOf(handler);
        if (idx >= 0) handlers.splice(idx, 1);
      }
      handlers.push(handler);
    });
  }

  return {
    async initAll(opts = {}) {
      const enableDepth = opts.enableDepth !== false;
      const enableSeam = opts.enableSeam !== false;
      const baseUrl = getBaseUrl();
      const results = { cv: false, depth: false, seam: false };

      // Init CV worker — generous timeout because opencv.js is ~11 MB WASM
      try {
        const w = createCV();
        w.postMessage({ type: 'init', baseUrl, opencvPath: 'opencv/opencv.js' });
        await waitForMsg(cvHandlers, 'progress', 60000);
        results.cv = true;
        console.log('cv-worker ready');
      } catch (e) {
        console.warn('cv-worker init failed:', e);
      }

      // Init Depth worker (best-effort; may fail if no model)
      if (enableDepth) {
        try {
          const w = createDepth();
          w.postMessage({
            type: 'init',
            baseUrl,
            modelPath: 'models/depth_256.onnx',
            preferWebGPU: true,
            targetSize: 256,
          });
          await waitForMsg(depthHandlers, 'progress', 60000);
          results.depth = true;
          console.log('depth-worker ready');
        } catch (e) {
          console.warn('depth-worker init failed (depth disabled):', e);
        }
      }

      // Init Seam worker (simple JS — should init quickly)
      if (enableSeam) {
        try {
          const w = createSeam();
          w.postMessage({ type: 'init', baseUrl, maxflowPath: 'wasm/maxflow/maxflow.js' });
          await waitForMsg(seamHandlers, 'progress', 60000);
          results.seam = true;
          console.log('seam-worker ready');
        } catch (e) {
          console.warn('seam-worker init failed (seam disabled):', e);
        }
      }

      return results;
    },

    sendCV(msg, transfer = []) {
      createCV().postMessage(msg, transfer);
    },
    sendDepth(msg, transfer = []) {
      createDepth().postMessage(msg, transfer);
    },
    sendSeam(msg, transfer = []) {
      createSeam().postMessage(msg, transfer);
    },

    onCV(handler) {
      cvHandlers.push(handler);
      return () => { const i = cvHandlers.indexOf(handler); if (i >= 0) cvHandlers.splice(i, 1); };
    },
    onDepth(handler) {
      depthHandlers.push(handler);
      return () => { const i = depthHandlers.indexOf(handler); if (i >= 0) depthHandlers.splice(i, 1); };
    },
    onSeam(handler) {
      seamHandlers.push(handler);
      return () => { const i = seamHandlers.indexOf(handler); if (i >= 0) seamHandlers.splice(i, 1); };
    },

    waitCV(type, timeoutMs) { return waitForMsg(cvHandlers, type, timeoutMs); },
    waitDepth(type, timeoutMs) { return waitForMsg(depthHandlers, type, timeoutMs); },
    waitSeam(type, timeoutMs) { return waitForMsg(seamHandlers, type, timeoutMs); },

    dispose() {
      cvWorker?.terminate();
      depthWorker?.terminate();
      seamWorker?.terminate();
      cvWorker = depthWorker = seamWorker = null;
    },
  };
}
