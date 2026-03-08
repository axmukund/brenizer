import { getState, setState, subscribe, type ImageEntry } from './appState';
import { capsSummary, type Capabilities } from './capabilities';
import { getPreset } from './presets';
import { parseExifMetadata, normalizeImageBlobOrientation } from './exif';
import { applyTurboModePreference } from './runtimeAcceleration';
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

function isAlignmentWorkflowSelected(): boolean {
  return getState().workflowAlignmentMode !== null;
}

function setSelectPlaceholder(select: HTMLSelectElement | null, label: string): void {
  if (!select || select.options.length === 0) return;
  if (select.options[0].text !== label) {
    select.options[0].text = label;
  }
}

function syncModeControls(): void {
  const {
    userMode,
    mobileSafeFlag,
    workflowAlignmentMode,
    workflowAlignmentChoiceMade,
    settings,
    workflowSameCameraChoiceMade,
    workflowOptimized,
    workflowPreviewReady,
  } = getState();
  const modeSelect = document.getElementById('mode-select') as HTMLSelectElement | null;
  if (modeSelect && modeSelect.value !== userMode) {
    modeSelect.value = userMode;
  }

  const msfFlag = document.getElementById('mobile-safe-flag') as HTMLInputElement | null;
  if (msfFlag) {
    msfFlag.checked = mobileSafeFlag;
  }

  const alignSelect = document.getElementById('workflow-step-align') as HTMLSelectElement | null;
  if (alignSelect) {
    const selectedValue = workflowAlignmentMode ?? '';
    if (alignSelect.value !== selectedValue) alignSelect.value = selectedValue;
    alignSelect.classList.toggle('active', workflowAlignmentChoiceMade);
  }

  const cameraSelect = document.getElementById('workflow-step-camera') as HTMLSelectElement | null;
  if (cameraSelect) {
    const selectedValue = !workflowSameCameraChoiceMade
      ? ''
      : settings?.sameCameraSettings
        ? 'same'
        : 'mixed';
    if (cameraSelect.value !== selectedValue) cameraSelect.value = selectedValue;
    cameraSelect.classList.toggle('active', workflowSameCameraChoiceMade);
  }

  const optimizeBtn = document.getElementById('btn-optimize') as HTMLButtonElement | null;
  if (optimizeBtn) {
    optimizeBtn.classList.toggle('active', workflowOptimized);
  }

  const previewBtn = document.getElementById('btn-stitch') as HTMLButtonElement | null;
  if (previewBtn) {
    previewBtn.classList.toggle('active', workflowPreviewReady);
  }

  const exportSelect = document.getElementById('workflow-step-export') as HTMLSelectElement | null;
  if (exportSelect) {
    setSelectPlaceholder(exportSelect, '5. Export');
    exportSelect.value = '';
  }
}

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
  let exifCount = 0;

  for (const file of toAdd) {
    try {
      let sourceBlob: Blob | File = file;
      const exif = await parseExifMetadata(file);
      
      // Try to convert HEIC/DNG files to JPEG
      const converted = await convertToJpeg(file);
      if (converted) {
        sourceBlob = converted;
        convertedCount++;
      }

      // Normalize to upright pixel orientation using EXIF orientation.
      const normalizedBlob = await normalizeImageBlobOrientation(sourceBlob, exif.orientation);
      if ((exif.orientation ?? 1) !== 1) exifCount++;
      
      const bmp = await createImageBitmap(normalizedBlob);
      const thumbUrl = makeThumb(bmp, 96);
      
      // Store normalized file so all downstream stages operate in the same
      // upright coordinate system regardless of EXIF orientation.
      const normalizedName = file.name.replace(/\.(heic|heif|dng)$/i, '.jpg');
      const normalizedType = normalizedBlob.type || 'image/jpeg';
      const storedFile = new File([normalizedBlob], normalizedName, { type: normalizedType });
      
      newEntries.push({
        id: genId(),
        file: storedFile,
        name: file.name,
        width: bmp.width,
        height: bmp.height,
        thumbUrl,
        excluded: false,
        exif: {
          orientation: exif.orientation,
          make: exif.make,
          model: exif.model,
          focalLengthMm: exif.focalLengthMm,
          focalLength35mm: exif.focalLength35mm,
          apertureFNumber: exif.apertureFNumber,
          exposureTimeSec: exif.exposureTimeSec,
          iso: exif.iso,
          whiteBalanceMode: exif.whiteBalanceMode,
          exposureBiasEv: exif.exposureBiasEv,
          capturedAtMs: exif.capturedAtMs,
        },
      });
      bmp.close();
    } catch (err) {
      console.warn('Failed to decode', file.name, err);
    }
  }

  setState({
    images: [...st.images, ...newEntries],
    workflowOptimized: false,
    workflowPreviewReady: false,
  });
  const details: string[] = [];
  if (convertedCount > 0) details.push(`${convertedCount} converted from HEIC/DNG`);
  if (exifCount > 0) details.push(`${exifCount} EXIF-oriented`);
  const msg = details.length > 0
    ? `${newEntries.length} image(s) added (${details.join(', ')}).`
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
  const { images, keyImageId } = getState();

  // Clear children except empty-state
  list.innerHTML = '';
  if (images.length === 0) {
    window.dispatchEvent(new CustomEvent('preview-image-hover', { detail: { imageId: null } }));
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
    item.dataset.imageId = img.id;

    item.addEventListener('mouseenter', () => {
      window.dispatchEvent(new CustomEvent('preview-image-hover', { detail: { imageId: img.id } }));
    });
    item.addEventListener('mouseleave', () => {
      window.dispatchEvent(new CustomEvent('preview-image-hover', { detail: { imageId: null } }));
    });

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
    nameEl.textContent = img.id === keyImageId ? `${img.name} ★` : img.name;
    const dimsEl = document.createElement('div');
    dimsEl.className = 'dims';
    dimsEl.textContent = `${img.width} × ${img.height}`;
    info.appendChild(nameEl);
    info.appendChild(dimsEl);

    const keyBtn = document.createElement('button');
    keyBtn.className = 'exclude-btn';
    keyBtn.textContent = img.id === keyImageId ? 'Unset Key' : 'Set Key';
    keyBtn.onclick = () => setKeyImage(img.id === keyImageId ? null : img.id);

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
    item.appendChild(keyBtn);
    item.appendChild(excBtn);
    item.appendChild(rmBtn);
    list.appendChild(item);
  });
}

function setKeyImage(imageId: string | null): void {
  const st = getState();
  if (imageId === null) {
    if (st.keyImageId !== null) {
      setState({ keyImageId: null, workflowPreviewReady: false });
      setStatus('Key image cleared.');
    }
    return;
  }

  const target = st.images.find((img) => img.id === imageId);
  if (!target) return;

  const images = st.images.map((img) =>
    img.id === imageId && img.excluded
      ? { ...img, excluded: false }
      : img,
  );
  setState({ images, keyImageId: imageId, workflowPreviewReady: false });
  setStatus(`Key image set: ${target.name}`);
}

function reorder(from: number, to: number): void {
  const images = [...getState().images];
  const [moved] = images.splice(from, 1);
  images.splice(to, 0, moved);
  setState({ images, workflowOptimized: false, workflowPreviewReady: false });
}

function toggleExclude(idx: number): void {
  const st = getState();
  const images = [...st.images];
  images[idx] = { ...images[idx], excluded: !images[idx].excluded };
  let keyImageId = st.keyImageId;
  if (images[idx].excluded && images[idx].id === st.keyImageId) {
    keyImageId = null;
    setStatus('Key image cleared because it was excluded.');
  }
  setState({ images, keyImageId, workflowOptimized: false, workflowPreviewReady: false });
}

function removeImage(idx: number): void {
  const st = getState();
  const removed = st.images[idx];
  const images = [...st.images];
  images.splice(idx, 1);
  const keyImageId = removed && removed.id === st.keyImageId ? null : st.keyImageId;
  setState({ images, keyImageId, workflowOptimized: false, workflowPreviewReady: false });
  if (removed && removed.id === st.keyImageId) {
    setStatus('Key image cleared because it was removed.');
  }
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
  const { images, workflowSameCameraChoiceMade, workflowOptimized } = getState();
  const active = images.filter(i => !i.excluded);
  const ph = $('canvas-placeholder');
  const canvas = $('preview-canvas') as HTMLCanvasElement;
  if (active.length === 0) {
    ph.style.display = '';
    canvas.style.display = 'none';
    ph.textContent = 'Upload images to begin.';
  } else {
    const alignmentSelected = isAlignmentWorkflowSelected();
    if (!alignmentSelected) {
      ph.textContent = `${active.length} image(s) ready — Step 1: pick Alignment Only or Align and Adjust.`;
    } else if (!workflowSameCameraChoiceMade) {
      ph.textContent = `${active.length} image(s) ready — Step 2: pick Same Camera or Mixed Settings.`;
    } else if (!workflowOptimized) {
      ph.textContent = `${active.length} image(s) ready — Step 3: click Optimize.`;
    } else {
      ph.textContent = `${active.length} image(s) ready — Step 4: click Stitch Preview.`;
    }
  }
}

function updateActionButtons(): void {
  const st = getState();
  const activeCount = st.images.reduce((n, img) => n + (img.excluded ? 0 : 1), 0);
  const hasEnoughImages = activeCount >= 2;
  const running = st.pipelineStatus === 'running';
  const alignmentSelected = isAlignmentWorkflowSelected();
  const cameraChoiceMade = st.workflowSameCameraChoiceMade;
  const optimizationReady = hasEnoughImages && alignmentSelected && cameraChoiceMade && !running;
  const previewReady = optimizationReady && st.workflowOptimized;
  const exportReady = previewReady && st.workflowPreviewReady && !running;

  const alignSelect = document.getElementById('workflow-step-align') as HTMLSelectElement | null;
  if (alignSelect) {
    alignSelect.disabled = !hasEnoughImages || running;
    alignSelect.title = !hasEnoughImages
      ? 'Need at least 2 active images.'
      : running
        ? 'Pipeline is running.'
        : 'Step 1: choose Alignment Only or Align and Adjust.';
  }

  const modeSelect = document.getElementById('mode-select') as HTMLSelectElement | null;
  if (modeSelect) modeSelect.disabled = running;

  const cameraSelect = document.getElementById('workflow-step-camera') as HTMLSelectElement | null;
  if (cameraSelect) {
    cameraSelect.disabled = !hasEnoughImages || !alignmentSelected || running;
    cameraSelect.title = !hasEnoughImages
      ? 'Need at least 2 active images.'
      : !alignmentSelected
        ? 'Step 1: choose Alignment Only or Align and Adjust first.'
        : running
          ? 'Pipeline is running.'
          : 'Step 2: choose Same Camera or Mixed Settings.';
  }

  const optimizeBtn = document.getElementById('btn-optimize') as HTMLButtonElement | null;
  if (optimizeBtn) {
    optimizeBtn.disabled = !optimizationReady;
    optimizeBtn.title = !hasEnoughImages
      ? 'Need at least 2 active images.'
      : !alignmentSelected
        ? 'Step 1: choose Alignment Only or Align and Adjust.'
        : !cameraChoiceMade
          ? 'Step 2: choose Same Camera or Mixed Settings.'
          : running
            ? 'Pipeline is running.'
            : 'Step 3: run optimization.';
  }

  const previewBtn = document.getElementById('btn-stitch') as HTMLButtonElement | null;
  if (previewBtn) {
    previewBtn.disabled = !previewReady;
    previewBtn.title = !hasEnoughImages
      ? 'Need at least 2 active images.'
      : !alignmentSelected
        ? 'Step 1: choose Alignment Only or Align and Adjust.'
        : !cameraChoiceMade
          ? 'Step 2: choose Same Camera or Mixed Settings.'
          : !st.workflowOptimized
            ? 'Step 3: optimize settings before preview.'
            : running
              ? 'Pipeline is running.'
              : 'Step 4: stitch the preview.';
  }

  const exportSelect = document.getElementById('workflow-step-export') as HTMLSelectElement | null;
  if (exportSelect) {
    exportSelect.disabled = !exportReady;
    exportSelect.title = !hasEnoughImages
      ? 'Need at least 2 active images.'
      : !alignmentSelected
        ? 'Step 1: choose Alignment Only or Align and Adjust.'
        : !cameraChoiceMade
          ? 'Step 2: choose Same Camera or Mixed Settings.'
          : !st.workflowOptimized
            ? 'Step 3: optimize settings.'
            : !st.workflowPreviewReady
              ? 'Step 4: stitch a preview before export.'
              : running
                ? 'Pipeline is running.'
                : 'Step 5: choose an export target.';
  }
}

// ── Settings panel ───────────────────────────────────────

/** Build the interactive settings panel from current PipelineSettings. */
export function buildSettingsPanel(): void {
  const panel = $('settings-panel');
  const { settings, turboModeEnabled } = getState();
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

  const externalToggle = (label: string, checked: boolean, onChange: (checked: boolean) => Promise<void> | void) => {
    const row = document.createElement('div');
    row.className = 'setting-row';
    const lbl = document.createElement('label');
    lbl.textContent = label;
    lbl.style.cssText = 'flex:1; font-size:12px;';
    const cb = document.createElement('input');
    cb.type = 'checkbox';
    cb.checked = checked;
    cb.style.cssText = 'width:auto;';
    cb.addEventListener('change', async () => {
      cb.disabled = true;
      try {
        await onChange(cb.checked);
      } finally {
        cb.disabled = false;
      }
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
  slider('Min Inliers (0=auto)', 'minInliers', 0, 18, 1);
  slider('LM Iterations', 'refineIters', 0, 100, 5);
  slider('APAP Grid', 'meshGrid', 0, 24, 2);
  toggle('Object-aware Alignment', 'objectAwareAlignment');

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
  toggle('Final Seam Pass', 'seamFinalPassEnabled');
  select('Final Pass Mode', 'seamFinalPassMode', [
    { value: 'fast', label: 'Fast' },
    { value: 'standard', label: 'Standard' },
    { value: 'highQuality', label: 'High Quality' },
  ]);
  slider('Final Base Width', 'seamFinalPassBaseWidth', 4, 48, 1, 'px');
  slider('Final Band Scale', 'seamFinalPassScale', 0.5, 2.5, 0.1);
  slider('Final Chroma Weight', 'seamFinalPassChromaWeight', 0.0, 1.0, 0.05);
  slider('Final Edge Gate', 'seamFinalPassEdgeGate', 4, 48, 1);
  slider('Final Max Shift', 'seamFinalPassMaxCorrection', 2, 32, 1);

  // ── Runtime ──
  section('Runtime');
  externalToggle('Turbo Mode (COI SW)', turboModeEnabled, async (checked) => {
    setState({ turboModeEnabled: checked });
    setStatus(`${checked ? 'Enabling' : 'Disabling'} turbo mode…`);
    const result = await applyTurboModePreference(checked);
    if (!result.reloading) {
      if (checked && result.active) {
        setStatus('Turbo mode active.');
      } else if (checked) {
        setStatus(`Turbo mode fallback: ${result.reason || 'cross-origin isolation unavailable'}`);
      } else {
        setStatus('Turbo mode disabled.');
      }
      buildSettingsPanel();
    }
  });

  // ── AI / ML ──
  section('AI / ML');
  toggle('Saliency-aware Seams', 'saliencyEnabled');
  toggle('Blur-aware Stitching', 'blurAwareStitching');

  // ── Camera Model ──
  section('Camera Model');
  const cameraNote = document.createElement('div');
  cameraNote.style.cssText = 'font-size:11px; color:var(--text-dim); margin-bottom:10px; line-height:1.4;';
  cameraNote.textContent = 'Choose Same Camera or Mixed Settings from the top-bar workflow dropdowns before running optimization.';
  panel.appendChild(cameraNote);

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
    setState({
      settings: getPreset(resolvedMode || 'desktopHQ'),
      workflowSameCameraChoiceMade: false,
      workflowOptimized: false,
      workflowPreviewReady: false,
    });
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
    setState({
      images: [],
      keyImageId: null,
      workflowAlignmentMode: null,
      workflowAlignmentChoiceMade: false,
      workflowSameCameraChoiceMade: false,
      workflowOptimized: false,
      workflowPreviewReady: false,
    });
    setStatus('All images cleared.');
  });

  // Mode selector
  const modeSelect = $('mode-select') as HTMLSelectElement;
  modeSelect.addEventListener('change', () => {
    setState({
      userMode: modeSelect.value as any,
      workflowSameCameraChoiceMade: false,
      workflowOptimized: false,
      workflowPreviewReady: false,
    });
  });

  const alignSelect = $('workflow-step-align') as HTMLSelectElement;
  alignSelect.addEventListener('change', () => {
    if (alignSelect.value !== 'alignmentOnly' && alignSelect.value !== 'alignAndAdjust') return;
    setState({
      workflowAlignmentMode: alignSelect.value as 'alignmentOnly' | 'alignAndAdjust',
      workflowAlignmentChoiceMade: true,
      workflowSameCameraChoiceMade: false,
      workflowOptimized: false,
      workflowPreviewReady: false,
    });
    setStatus(
      alignSelect.value === 'alignmentOnly'
        ? 'Step 1 complete. Alignment-only workflow selected. Now choose Same Camera or Mixed Settings.'
        : 'Step 1 complete. Align-and-adjust workflow selected. Now choose Same Camera or Mixed Settings.',
    );
  });

  const cameraSelect = $('workflow-step-camera') as HTMLSelectElement;
  cameraSelect.addEventListener('change', () => {
    const { settings } = getState();
    if (!settings) return;
    if (cameraSelect.value === 'same') {
      setState({
        settings: { ...settings, sameCameraSettings: true },
        workflowSameCameraChoiceMade: true,
        workflowOptimized: false,
        workflowPreviewReady: false,
      });
      buildSettingsPanel();
      setStatus('Step 2 complete. Same-camera optimization selected. Open Step 3 to optimize.');
    } else if (cameraSelect.value === 'mixed') {
      setState({
        settings: { ...settings, sameCameraSettings: false },
        workflowSameCameraChoiceMade: true,
        workflowOptimized: false,
        workflowPreviewReady: false,
      });
      buildSettingsPanel();
      setStatus('Step 2 complete. Mixed-settings optimization selected. Open Step 3 to optimize.');
    }
  });

  // Mobile safe flag
  const msfFlag = $('mobile-safe-flag') as HTMLInputElement;
  msfFlag.addEventListener('change', () => {
    setState({
      mobileSafeFlag: msfFlag.checked,
      workflowSameCameraChoiceMade: false,
      workflowOptimized: false,
      workflowPreviewReady: false,
    });
  });

  // Subscribe to state changes
  subscribe(() => {
    renderImageList();
    updateCanvasPlaceholder();
    updateActionButtons();
    syncModeControls();
  });

  // Initial render
  renderImageList();
  updateCanvasPlaceholder();
  updateActionButtons();
  syncModeControls();
  buildSettingsPanel();
}

export { renderCapabilities, setStatus, startProgress, endProgress, updateProgress };
