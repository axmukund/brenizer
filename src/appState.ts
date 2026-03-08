import type { ExifMetadata } from './exif';
import type { Capabilities } from './capabilities';
import type { PipelineSettings, ModeName } from './presets';

/** Represents a single loaded image. */
export type ImageExifMeta = ExifMetadata;

export interface ImageEntry {
  id: string;
  file: File;
  name: string;
  width: number;
  height: number;
  thumbUrl: string;
  excluded: boolean;
  exif?: ImageExifMeta;
}

export interface AppState {
  images: ImageEntry[];
  keyImageId: string | null;
  capabilities: Capabilities | null;
  turboModeEnabled: boolean;
  userMode: ModeName;
  mobileSafeFlag: boolean;
  resolvedMode: string;
  settings: PipelineSettings | null;
  pipelineStatus: 'idle' | 'running' | 'error';
  workflowAlignmentMode: 'alignmentOnly' | 'alignAndAdjust' | null;
  workflowAlignmentChoiceMade: boolean;
  workflowSameCameraChoiceMade: boolean;
  workflowOptimized: boolean;
  workflowPreviewReady: boolean;
  degradations: string[];
}

let _state: AppState = {
  images: [],
  keyImageId: null,
  capabilities: null,
  turboModeEnabled: false,
  userMode: 'auto',
  mobileSafeFlag: false,
  resolvedMode: 'auto',
  settings: null,
  pipelineStatus: 'idle',
  workflowAlignmentMode: null,
  workflowAlignmentChoiceMade: false,
  workflowSameCameraChoiceMade: false,
  workflowOptimized: false,
  workflowPreviewReady: false,
  degradations: [],
};

type Listener = () => void;
const _listeners: Listener[] = [];

/** Return the current global application state (immutable snapshot). */
export function getState(): AppState {
  return _state;
}

/** Merge `partial` into the current state and notify all subscribers. */
export function setState(partial: Partial<AppState>): void {
  _state = { ..._state, ...partial };
  _listeners.forEach(fn => fn());
}

/** Subscribe to state changes. Returns an unsubscribe function. */
export function subscribe(fn: Listener): () => void {
  _listeners.push(fn);
  return () => {
    const idx = _listeners.indexOf(fn);
    if (idx >= 0) _listeners.splice(idx, 1);
  };
}
