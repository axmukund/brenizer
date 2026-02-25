import type { Capabilities } from './capabilities';
import type { PipelineSettings, ModeName } from './presets';

/** Represents a single loaded image. */
export interface ImageEntry {
  id: string;
  file: File;
  name: string;
  width: number;
  height: number;
  thumbUrl: string;
  excluded: boolean;
}

export interface AppState {
  images: ImageEntry[];
  capabilities: Capabilities | null;
  userMode: ModeName;
  mobileSafeFlag: boolean;
  resolvedMode: string;
  settings: PipelineSettings | null;
  pipelineStatus: 'idle' | 'running' | 'error';
  degradations: string[];
}

let _state: AppState = {
  images: [],
  capabilities: null,
  userMode: 'auto',
  mobileSafeFlag: false,
  resolvedMode: 'auto',
  settings: null,
  pipelineStatus: 'idle',
  degradations: [],
};

type Listener = () => void;
const _listeners: Listener[] = [];

export function getState(): AppState {
  return _state;
}

export function setState(partial: Partial<AppState>): void {
  _state = { ..._state, ...partial };
  _listeners.forEach(fn => fn());
}

export function subscribe(fn: Listener): () => void {
  _listeners.push(fn);
  return () => {
    const idx = _listeners.indexOf(fn);
    if (idx >= 0) _listeners.splice(idx, 1);
  };
}
