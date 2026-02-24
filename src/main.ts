import { detectCapabilities } from './capabilities';
import { resolveMode, getPreset } from './presets';
import { setState, getState, subscribe } from './appState';
import { initUI, renderCapabilities, setStatus } from './ui';

async function boot(): Promise<void> {
  // Init UI first so elements are wired
  initUI();

  // Detect capabilities
  setStatus('Detecting capabilities…');
  const caps = await detectCapabilities();
  setState({ capabilities: caps });
  renderCapabilities(caps);

  // Resolve mode and apply preset
  const { userMode, mobileSafeFlag } = getState();
  const resolved = resolveMode(userMode, mobileSafeFlag, caps);
  const settings = getPreset(resolved);
  setState({ resolvedMode: resolved, settings });
  setStatus(`Ready — mode: ${resolved}`);

  // Re-resolve mode on setting changes
  subscribe(() => {
    const s = getState();
    if (s.capabilities) {
      const newMode = resolveMode(s.userMode, s.mobileSafeFlag, s.capabilities);
      if (newMode !== s.resolvedMode) {
        const newSettings = getPreset(newMode);
        // Use direct assignment to avoid infinite loop
        setState({ resolvedMode: newMode, settings: newSettings });
        setStatus(`Mode changed to ${newMode}`);
      }
    }
  });
}

boot().catch(err => {
  console.error('Boot failed:', err);
  document.getElementById('status-bar')!.textContent = `Error: ${err.message}`;
});
