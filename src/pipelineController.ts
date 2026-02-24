/**
 * Pipeline controller — orchestrates the image stitching pipeline.
 * Manages worker lifecycle, runs pipeline stages in order.
 */

import { createWorkerManager, type WorkerManager } from './workers/workerManager';
import { getState, setState } from './appState';
import { setStatus } from './ui';

let workerManager: WorkerManager | null = null;

export function getWorkerManager(): WorkerManager | null {
  return workerManager;
}

/** Initialize all workers. Returns readiness status. */
export async function initWorkers(): Promise<{ cv: boolean; depth: boolean; seam: boolean }> {
  if (workerManager) {
    workerManager.dispose();
  }
  workerManager = createWorkerManager();
  setStatus('Initializing workers…');

  const result = await workerManager.initAll();

  const parts: string[] = [];
  if (result.cv) parts.push('CV ✓');
  else parts.push('CV ✗');
  if (result.depth) parts.push('Depth ✓');
  else parts.push('Depth ✗');
  if (result.seam) parts.push('Seam ✓');
  else parts.push('Seam ✗');

  setStatus(`Workers: ${parts.join(' | ')}`);
  return result;
}

/** Run the full stitch preview pipeline. */
export async function runStitchPreview(): Promise<void> {
  const { images, settings } = getState();
  const active = images.filter(i => !i.excluded);

  if (active.length < 2) {
    setStatus('Need at least 2 images to stitch.');
    return;
  }

  if (!settings) {
    setStatus('Settings not loaded.');
    return;
  }

  setState({ pipelineStatus: 'running' });
  setStatus('Starting pipeline…');

  // Step 1: Init workers
  const ready = await initWorkers();
  if (!ready.cv) {
    setStatus('CV worker failed to initialize. Cannot stitch.');
    setState({ pipelineStatus: 'error' });
    return;
  }

  setStatus('Workers ready. Pipeline placeholder complete.');
  setState({ pipelineStatus: 'idle' });
}
