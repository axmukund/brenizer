import { getState, setState, subscribe, type ImageEntry } from './appState';
import { capsSummary, type Capabilities } from './capabilities';

// ── helpers ──────────────────────────────────────────────
function $(id: string) { return document.getElementById(id)!; }

let nextId = 1;
function genId(): string { return `img-${nextId++}`; }

// ── file import ──────────────────────────────────────────
const VALID_TYPES = new Set(['image/jpeg', 'image/png']);

async function importFiles(files: FileList | File[]): Promise<void> {
  const st = getState();
  const maxImages = st.settings?.maxImages ?? 25;
  const arr = Array.from(files).filter(f => VALID_TYPES.has(f.type));

  if (arr.length === 0) {
    setStatus('No valid JPG/PNG files selected.');
    return;
  }

  const total = st.images.length + arr.length;
  if (total > maxImages) {
    setStatus(`Image cap is ${maxImages}. ${total - maxImages} file(s) skipped.`);
  }

  const toAdd = arr.slice(0, maxImages - st.images.length);
  const newEntries: ImageEntry[] = [];

  for (const file of toAdd) {
    try {
      const bmp = await createImageBitmap(file);
      const thumbUrl = makeThumb(bmp, 96);
      newEntries.push({
        id: genId(),
        file,
        name: file.name,
        width: bmp.width,
        height: bmp.height,
        thumbUrl,
        excluded: false,
      });
      bmp.close();
    } catch {
      console.warn('Failed to decode', file.name);
    }
  }

  setState({ images: [...st.images, ...newEntries] });
  setStatus(`${newEntries.length} image(s) added.`);
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
  const empty = $('empty-state');

  // Clear children except empty-state
  list.innerHTML = '';
  if (images.length === 0) {
    const el = document.createElement('div');
    el.className = 'empty-state';
    el.id = 'empty-state';
    el.textContent = 'Drop or upload JPG/PNG images to begin';
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

// ── status ───────────────────────────────────────────────
function setStatus(msg: string): void {
  $('status-bar').textContent = msg;
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
}

export { renderCapabilities, setStatus };
