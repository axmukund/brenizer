import { setState } from './appState';
import { getTurboModePreference, prepareTurboModeRuntime } from './runtimeAcceleration';

async function start(): Promise<void> {
  const turboModeEnabled = getTurboModePreference();
  setState({ turboModeEnabled });

  const preflight = await prepareTurboModeRuntime(turboModeEnabled);
  if (preflight.reloading) return;
  if (preflight.reason) {
    console.warn('Turbo mode preflight fallback:', preflight.reason);
  }

  const { boot } = await import('./main');
  await boot();
}

start().catch((err) => {
  console.error('Bootstrap error:', err);
  const message = err instanceof Error ? err.message : String(err);
  const bar = document.getElementById('status-bar');
  const msgEl = bar?.querySelector('.status-msg');
  if (msgEl) msgEl.textContent = `Error: ${message}`;
  else if (bar) bar.textContent = `Error: ${message}`;
});
