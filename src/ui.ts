import { getState, setState, subscribe, type ImageEntry } from './appState';
import { capsSummary, type Capabilities } from './capabilities';
import { getPreset } from './presets';
import heic2any from 'heic2any';
import UTIF from 'utif';

// Lazy import to avoid circular dep — set by main.ts
let _renderImagePreview: ((entry: ImageEntry) => Promise<void>) | null = null;
export function setRenderImagePreview(fn: (entry: ImageEntry) => Promise<void>): void {
  _renderImagePreview = fn;
}

// ── helpers ──────────────────────────────────────────────
function $(id: string) { return document.getElementById(id)!; }

let nextId = 1;
function genId(): string { return `img-${nextId++}`; }

// ── file import ──────────────────────────────────────────
const VALID_TYPES = new Set(['image/jpeg', 'image/png', 'image/heic', 'image/heif', 'image/x-adobe-dng', 'image/dng']);

/**
 * Convert HEIC/HEIF/DNG files to JPEG.
 * Returns a Blob suitable for createImageBitmap, or null if conversion fails.
 */
async function convertToJpeg(file: File): Promise<Blob | null> {
  const type = file.type.toLowerCase();
  const name = file.name.toLowerCase();

  // HEIC/HEIF conversion using heic2any
  if (type === 'image/heic' || type === 'image/heif' || name.endsWith('.heic') || name.endsWith('.heif')) {
    try {
      const result = await heic2any({ blob: file, toType: 'image/jpeg', quality: 0.95 });
      return Array.isArray(result) ? result[0] : result;
    } catch (err) {
      console.error('HEIC conversion failed:', err);
      return null;
    }
  }

  // DNG conversion: extract embedded JPEG preview
  if (type === 'image/x-adobe-dng' || type === 'image/dng' || name.endsWith('.dng')) {
    try {
      const buffer = await file.arrayBuffer();
      const ifds = UTIF.decode(buffer);
      
      // Find the largest JPEG preview (usually IFD with compression 7)
      let bestIfd = null;
      let bestSize = 0;
      
      for (const ifd of ifds) {
        // Compression 7 = JPEG
        const compression = ifd.t259;
        if (compression && Array.isArray(compression) && compression[0] === 7) {
          const size = (ifd.width || 0) * (ifd.height || 0);
          if (size > bestSize) {
            bestSize = size;
            bestIfd = ifd;
          }
        }
      }

      if (bestIfd) {
        // Extract JPEG data from the IFD
        const offsetTag = bestIfd.t513;  // StripOffsets or JPEGInterchangeFormat
        const lengthTag = bestIfd.t514;  // StripByteCounts or JPEGInterchangeFormatLength
        const offset = (offsetTag && Array.isArray(offsetTag)) ? Number(offsetTag[0]) : 0;
        const length = (lengthTag && Array.isArray(lengthTag)) ? Number(lengthTag[0]) : 0;
        
        if (offset && length) {
          const jpegData = buffer.slice(offset, offset + length);
          return new Blob([jpegData], { type: 'image/jpeg' });
        }
      }
      
      console.error('No JPEG preview found in DNG file');
      return null;
    } catch (err) {
      console.error('DNG conversion failed:', err);
      return null;
    }
  }

  return null;
}

async function importFiles(files: FileList | File[]): Promise<void> {
  const st = getState();
  const maxImages = st.settings?.maxImages ?? 25;
  
  // Accept files if they have valid MIME type OR valid extension
  const arr = Array.from(files).filter(f => {
    const hasValidType = VALID_TYPES.has(f.type);
    const name = f.name.toLowerCase();
    const hasValidExt = name.endsWith('.jpg') || name.endsWith('.jpeg') || name.endsWith('.png') || 
                        name.endsWith('.heic') || name.endsWith('.heif') || name.endsWith('.dng');
    return hasValidType || hasValidExt;
  });

  if (arr.length === 0) {
    setStatus('No valid image files selected (JPG/PNG/HEIC/DNG).');
    return;
  }

  const total = st.images.length + arr.length;
  if (total > maxImages) {
    setStatus(`Image cap is ${maxImages}. ${total - maxImages} file(s) skipped.`);
  }

  const toAdd = arr.slice(0, maxImages - st.images.length);
  const newEntries: ImageEntry[] = [];
  let convertedCount = 0;

  for (const file of toAdd) {
    try {
      let sourceBlob: Blob | File = file;
      
      // Try to convert HEIC/DNG files to JPEG
      const converted = await convertToJpeg(file);
      if (converted) {
        sourceBlob = converted;
        convertedCount++;
      }
      
      const bmp = await createImageBitmap(sourceBlob);
      const thumbUrl = makeThumb(bmp, 96);
      
      // Store original file for reference, but use converted blob if available
      const storedFile = converted ? new File([converted], file.name.replace(/\.(heic|heif|dng)$/i, '.jpg'), { type: 'image/jpeg' }) : file;
      
      newEntries.push({
        id: genId(),
        file: storedFile,
        name: file.name,
        width: bmp.width,
        height: bmp.height,
        thumbUrl,
        excluded: false,
      });
      bmp.close();
    } catch (err) {
      console.warn('Failed to decode', file.name, err);
    }
  }

  setState({ images: [...st.images, ...newEntries] });
  const msg = convertedCount > 0 
    ? `${newEntries.length} image(s) added (${convertedCount} converted from HEIC/DNG).`
    : `${newEntries.length} image(s) added.`;
  setStatus(msg);
}

function makeThumb(bmp: ImageBitmap, maxDim: number): string {
  const scale = Math.min(maxDim / bmp.width, maxDim / bmp.height, 1);
  const w = Math.round(bmp.width * scale);
  const h = Math.round(bmp.height * scale);
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.drawImage(bmp, 0, 0, w, h);
  return c.toDataURL('image/jpeg', 0.6);
}

// ── image list rendering ─────────────────────────────────
let dragSrcIdx: number | null = null;

function renderImageList(): void {
  const list = $('image-list');
  const { images } = getState();

  // Clear children except empty-state
  list.innerHTML = '';
  if (images.length === 0) {
    const el = document.createElement('div');
    el.className = 'empty-state';
    el.id = 'empty-state';
    el.innerHTML = 'Drop or upload images to begin<br/><span style="font-size:11px; color:var(--text-dim);">JPG, PNG, HEIC, DNG</span>';
    list.appendChild(el);
    return;
  }

  $('image-count').textContent = `(${images.length})`;

  images.forEach((img, idx) => {
    const item = document.createElement('div');
    item.className = 'img-item' + (img.excluded ? ' excluded' : '');
    item.draggable = true;
    item.dataset.idx = String(idx);

    // Drag handlers
    item.addEventListener('dragstart', (e) => {
      dragSrcIdx = idx;
      item.classList.add('dragging');
      e.dataTransfer!.effectAllowed = 'move';
    });
    item.addEventListener('dragend', () => {
      item.classList.remove('dragging');
      dragSrcIdx = null;
      document.querySelectorAll('.drag-over').forEach(el => el.classList.remove('drag-over'));
    });
    item.addEventListener('dragover', (e) => {
      e.preventDefault();
      e.dataTransfer!.dropEffect = 'move';
      item.classList.add('drag-over');
    });
    item.addEventListener('dragleave', () => item.classList.remove('drag-over'));
    item.addEventListener('drop', (e) => {
      e.preventDefault();
      item.classList.remove('drag-over');
      if (dragSrcIdx !== null && dragSrcIdx !== idx) {
        reorder(dragSrcIdx, idx);
      }
    });

    const thumb = document.createElement('img');
    thumb.src = img.thumbUrl;
    thumb.alt = img.name;
    thumb.style.cursor = 'pointer';
    thumb.addEventListener('click', (e) => {
      e.stopPropagation();
      if (_renderImagePreview) _renderImagePreview(img);
    });

    const info = document.createElement('div');
    info.className = 'img-info';
    const nameEl = document.createElement('div');
    nameEl.className = 'name';
    nameEl.textContent = img.name;
    const dimsEl = document.createElement('div');
    dimsEl.className = 'dims';
    dimsEl.textContent = `${img.width} × ${img.height}`;
    info.appendChild(nameEl);
    info.appendChild(dimsEl);

    const excBtn = document.createElement('button');
    excBtn.className = 'exclude-btn';
    excBtn.textContent = img.excluded ? 'Include' : 'Exclude';
    excBtn.onclick = () => toggleExclude(idx);

    const rmBtn = document.createElement('button');
    rmBtn.className = 'remove-btn';
    rmBtn.textContent = '×';
    rmBtn.title = 'Remove';
    rmBtn.onclick = () => removeImage(idx);

    item.appendChild(thumb);
    item.appendChild(info);
    item.appendChild(excBtn);
    item.appendChild(rmBtn);
    list.appendChild(item);
  });
}

function reorder(from: number, to: number): void {
  const images = [...getState().images];
  const [moved] = images.splice(from, 1);
  images.splice(to, 0, moved);
  setState({ images });
}

function toggleExclude(idx: number): void {
  const images = [...getState().images];
  images[idx] = { ...images[idx], excluded: !images[idx].excluded };
  setState({ images });
}

function removeImage(idx: number): void {
  const images = [...getState().images];
  images.splice(idx, 1);
  setState({ images });
}

// ── capabilities UI ──────────────────────────────────────
function renderCapabilities(caps: Capabilities): void {
  const bar = $('capabilities-bar');
  bar.innerHTML = '';
  const items = capsSummary(caps);
  items.forEach(({ label, status }) => {
    const el = document.createElement('span');
    el.className = 'cap-item';
    const dot = document.createElement('span');
    dot.className = 'cap-dot cap-' + status;
    el.appendChild(dot);
    el.appendChild(document.createTextNode(' ' + label));
    bar.appendChild(el);
  });
}

// ── status & progress ────────────────────────────────────
let _pipelineStart = 0;
let _stageWeights: Record<string, number> = {};
let _stageOrder: string[] = [];
let _stageProgress: Record<string, number> = {};
let _currentStage = '';

/**
 * Begin tracking pipeline progress. Call at pipeline start.
 * Stages: array of { name, weight } where weight is relative time estimate.
 */
function startProgress(stages: { name: string; weight: number }[]): void {
  _pipelineStart = performance.now();
  _stageWeights = {};
  _stageOrder = [];
  _stageProgress = {};
  _currentStage = '';
  for (const s of stages) {
    _stageWeights[s.name] = s.weight;
    _stageOrder.push(s.name);
    _stageProgress[s.name] = 0;
  }
  const bar = $('status-bar');
  bar.classList.add('has-progress');
  (bar.querySelector('.progress-fill') as HTMLElement).style.width = '0%';
  (bar.querySelector('.status-eta') as HTMLElement).textContent = '';
}

/** End progress tracking. */
function endProgress(): void {
  _pipelineStart = 0;
  _currentStage = '';
  const bar = $('status-bar');
  bar.classList.remove('has-progress');
  (bar.querySelector('.progress-fill') as HTMLElement).style.width = '0%';
  (bar.querySelector('.status-eta') as HTMLElement).textContent = '';
}

/** Update progress within a stage (0-1). */
function updateProgress(stage: string, fraction: number): void {
  _stageProgress[stage] = Math.max(_stageProgress[stage] ?? 0, Math.min(1, fraction));
  _currentStage = stage;

  // Compute overall weighted progress
  let totalWeight = 0;
  let completedWeight = 0;
  for (const name of _stageOrder) {
    const w = _stageWeights[name] ?? 0;
    totalWeight += w;
    completedWeight += w * (_stageProgress[name] ?? 0);
  }
  const overall = totalWeight > 0 ? completedWeight / totalWeight : 0;

  // Update progress bar
  const fill = $('status-bar').querySelector('.progress-fill') as HTMLElement;
  fill.style.width = `${Math.round(overall * 100)}%`;

  // Calculate ETA
  const elapsed = performance.now() - _pipelineStart;
  const etaEl = $('status-bar').querySelector('.status-eta') as HTMLElement;
  if (overall > 0.02 && elapsed > 1000) {
    const totalEstMs = elapsed / overall;
    const remainMs = totalEstMs - elapsed;
    const remainSec = Math.ceil(remainMs / 1000);
    if (remainSec > 0 && remainSec < 600) {
      if (remainSec >= 60) {
        const m = Math.floor(remainSec / 60);
        const s = remainSec % 60;
        etaEl.textContent = `~${m}m ${s}s remaining`;
      } else {
        etaEl.textContent = `~${remainSec}s remaining`;
      }
    } else {
      etaEl.textContent = '';
    }
  } else {
    etaEl.textContent = _pipelineStart ? 'Estimating…' : '';
  }
}

function setStatus(msg: string): void {
  const msgEl = $('status-bar').querySelector('.status-msg') as HTMLElement;
  if (msgEl) {
    msgEl.textContent = msg;
  } else {
    // Fallback if HTML not yet upgraded
    $('status-bar').textContent = msg;
  }
  $('status-chip').textContent = msg.slice(0, 40);
}

// ── canvas placeholder ──────────────────────────────────
function updateCanvasPlaceholder(): void {
  const { images } = getState();
  const active = images.filter(i => !i.excluded);
  const ph = $('canvas-placeholder');
  const canvas = $('preview-canvas') as HTMLCanvasElement;
  if (active.length === 0) {
    ph.style.display = '';
    canvas.style.display = 'none';
    ph.textContent = 'Upload images and click "Stitch Preview" to begin';
  } else {
    ph.textContent = `${active.length} image(s) ready — click "Stitch Preview"`;
  }
}

function updateStitchButton(): void {
  const active = getState().images.filter(i => !i.excluded);
  ($('btn-stitch') as HTMLButtonElement).disabled = active.length < 2;
}

// ── Settings panel ───────────────────────────────────────

/** Build the interactive settings panel from current PipelineSettings. */
export function buildSettingsPanel(): void {
  const panel = $('settings-panel');
  const { settings } = getState();
  if (!settings) return;

  panel.innerHTML = '<h3>Pipeline Settings</h3>';

  // Helper: add a section header
  const section = (title: string) => {
    const h = document.createElement('h4');
    h.textContent = title;
    h.style.cssText = 'font-size:12px; color:var(--accent); margin:14px 0 6px; border-bottom:1px solid var(--border); padding-bottom:3px;';
    panel.appendChild(h);
  };

  // Helper: add a numeric slider row
  const slider = (label: string, key: keyof typeof settings, min: number, max: number, step: number, suffix = '') => {
    const row = document.createElement('div');
    row.className = 'setting-row';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    lbl.style.cssText = 'flex:1; font-size:12px;';
    const valSpan = document.createElement('span');
    valSpan.style.cssText = 'font-size:11px; color:var(--text-dim); min-width:40px; text-align:right; margin-right:6px;';
    valSpan.textContent = String(settings[key]) + suffix;
    const input = document.createElement('input');
    input.type = 'range';
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(settings[key]);
    input.style.cssText = 'width:90px;';
    input.addEventListener('input', () => {
      const v = Number(input.value);
      const { settings: cur } = getState();
      if (cur) setState({ settings: { ...cur, [key]: v } });
      valSpan.textContent = (step < 1 ? v.toFixed(2) : String(v)) + suffix;
    });
    row.appendChild(lbl);
    row.appendChild(valSpan);
    row.appendChild(input);
    panel.appendChild(row);
  };

  // Helper: add a toggle row
  const toggle = (label: string, key: keyof typeof settings) => {
    const row = document.createElement('div');
    row.className = 'setting-row';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    lbl.style.cssText = 'flex:1; font-size:12px;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = !!settings[key];
    cb.style.cssText = 'width:auto;';
    cb.addEventListener('change', () => {
      const { settings: cur } = getState();
      if (cur) setState({ settings: { ...cur, [key]: cb.checked } });
    });
    row.appendChild(lbl);
    row.appendChild(cb);
    panel.appendChild(row);
  };

  // Helper: add a select row
  const select = (label: string, key: keyof typeof settings, options: { value: string; label: string }[]) => {
    const row = document.createElement('div');
    row.className = 'setting-row';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    lbl.style.cssText = 'flex:1; font-size:12px;';
    const sel = document.createElement('select');
    for (const opt of options) {
      const o = document.createElement('option');
      o.value = opt.value;
      o.textContent = opt.label;
      if (String(settings[key]) === opt.value) o.selected = true;
      sel.appendChild(o);
    }
    sel.addEventListener('change', () => {
      const { settings: cur } = getState();
      if (cur) setState({ settings: { ...cur, [key]: sel.value } });
    });
    row.appendChild(lbl);
    row.appendChild(sel);
    panel.appendChild(row);
  };

  // ── Feature Extraction ──
  section('Feature Extraction');
  slider('Max Images', 'maxImages', 4, 50, 1);
  slider('Align Scale (px)', 'alignScale', 512, 3072, 128, 'px');
  slider('ORB Features', 'orbFeatures', 500, 10000, 500);
  slider('Ratio Test', 'ratioTest', 0.5, 0.95, 0.05);
  slider('RANSAC Threshold', 'ransacThreshPx', 1, 10, 0.5, 'px');

  // ── Matching ──
  section('Matching');
  slider('Pair Window', 'pairWindowW', 2, 20, 1);
  toggle('Match All Pairs', 'matchAllPairs');
  slider('LM Iterations', 'refineIters', 0, 100, 5);
  slider('APAP Grid', 'meshGrid', 0, 24, 2);

  // ── Compositing ──
  section('Compositing');
  select('Seam Method', 'seamMethod', [
    { value: 'graphcut', label: 'Graph Cut' },
    { value: 'feather', label: 'Feather Only' },
  ]);
  slider('Block Size', 'seamBlockSize', 4, 64, 4, 'px');
  slider('Feather Width', 'featherWidth', 5, 200, 5, 'px');
  toggle('Multi-band Blend', 'multibandEnabled');
  slider('Pyramid Levels (0=auto)', 'multibandLevels', 0, 7, 1);
  toggle('Exposure Compensation', 'exposureComp');

  // ── AI / ML ──
  section('AI / ML');
  toggle('Saliency-aware Seams', 'saliencyEnabled');
  toggle('Blur-aware Stitching', 'blurAwareStitching');

  // ── Camera Model ──
  section('Camera Model');
  toggle('Same Camera Settings', 'sameCameraSettings');

  // ── Lens Corrections ──
  section('Lens Corrections');
  toggle('Vignette Correction', 'vignetteCorrection');

  // ── Export ──
  section('Export');
  slider('Export Scale', 'exportScale', 0.1, 1.0, 0.05);
  select('Format', 'exportFormat', [
    { value: 'png', label: 'PNG' },
    { value: 'jpeg', label: 'JPEG' },
  ]);
  slider('JPEG Quality', 'exportJpegQuality', 0.5, 1.0, 0.05);
  toggle('Max Resolution Export', 'maxResExport');

  // ── Depth (experimental) ──
  section('Depth (Experimental)');
  toggle('Depth Enabled', 'depthEnabled');
  slider('Depth Input Size', 'depthInputSize', 64, 512, 64, 'px');

  // Reset to defaults button
  const resetRow = document.createElement('div');
  resetRow.style.cssText = 'margin-top:16px; text-align:center;';
  const resetBtn = document.createElement('button');
  resetBtn.textContent = 'Reset to Defaults';
  resetBtn.style.cssText = 'font-size:12px; padding:6px 16px; background:var(--surface2); border:1px solid var(--border); color:var(--text); border-radius:4px; cursor:pointer;';
  resetBtn.addEventListener('click', () => {
    const { resolvedMode } = getState();
    setState({ settings: getPreset(resolvedMode || 'desktopHQ') });
    buildSettingsPanel();
  });
  resetRow.appendChild(resetBtn);
  panel.appendChild(resetRow);
}

// ── init ─────────────────────────────────────────────────
export function initUI(): void {
  // File input
  const fileInput = $('file-input') as HTMLInputElement;
  fileInput.addEventListener('change', () => {
    if (fileInput.files) importFiles(fileInput.files);
    fileInput.value = '';
  });

  // Upload button (topbar)
  $('btn-upload-topbar').addEventListener('click', () => fileInput.click());

  // Drag-and-drop on the whole page
  const body = document.body;
  body.addEventListener('dragover', (e) => { e.preventDefault(); });
  body.addEventListener('drop', (e) => {
    e.preventDefault();
    if (e.dataTransfer?.files) importFiles(e.dataTransfer.files);
  });

  // Clear all
  $('btn-clear-all').addEventListener('click', () => {
    setState({ images: [] });
    setStatus('All images cleared.');
  });

  // Mode selector
  const modeSelect = $('mode-select') as HTMLSelectElement;
  modeSelect.addEventListener('change', () => {
    setState({ userMode: modeSelect.value as any });
  });

  // Mobile safe flag
  const msfFlag = $('mobile-safe-flag') as HTMLInputElement;
  msfFlag.addEventListener('change', () => {
    setState({ mobileSafeFlag: msfFlag.checked });
  });

  // Subscribe to state changes
  subscribe(() => {
    renderImageList();
    updateCanvasPlaceholder();
    updateStitchButton();
  });

  // Initial render
  renderImageList();
  updateCanvasPlaceholder();
  updateStitchButton();
  buildSettingsPanel();
}

export { renderCapabilities, setStatus, startProgress, endProgress, updateProgress };
