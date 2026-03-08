export interface ExifMetadata {
  orientation: number;
  make?: string;
  model?: string;
  focalLengthMm?: number;
  focalLength35mm?: number;
  apertureFNumber?: number;
  exposureTimeSec?: number;
  iso?: number;
  whiteBalanceMode?: 'auto' | 'manual';
  exposureBiasEv?: number;
  capturedAtMs?: number;
}

function readAscii(view: DataView, offset: number, count: number): string {
  const chars: number[] = [];
  for (let i = 0; i < count; i++) {
    const c = view.getUint8(offset + i);
    if (c === 0) break;
    chars.push(c);
  }
  return String.fromCharCode(...chars).trim();
}

function parseExifDateTime(value: string): number | undefined {
  // EXIF canonical format: "YYYY:MM:DD HH:MM:SS"
  const m = value.match(/^(\d{4}):(\d{2}):(\d{2})[ T](\d{2}):(\d{2}):(\d{2})$/);
  if (!m) return undefined;
  const year = Number(m[1]);
  const month = Number(m[2]) - 1;
  const day = Number(m[3]);
  const hour = Number(m[4]);
  const minute = Number(m[5]);
  const second = Number(m[6]);
  if (!Number.isFinite(year) || !Number.isFinite(month) || !Number.isFinite(day)) return undefined;
  return Date.UTC(year, month, day, hour, minute, second);
}

interface IfdEntry {
  tag: number;
  type: number;
  count: number;
  valueOffset: number;
  entryOffset: number;
}

function readIfdEntries(view: DataView, ifdOffset: number, littleEndian: boolean): IfdEntry[] {
  if (ifdOffset < 0 || ifdOffset + 2 > view.byteLength) return [];
  const count = view.getUint16(ifdOffset, littleEndian);
  const out: IfdEntry[] = [];
  for (let i = 0; i < count; i++) {
    const off = ifdOffset + 2 + i * 12;
    if (off + 12 > view.byteLength) break;
    out.push({
      tag: view.getUint16(off, littleEndian),
      type: view.getUint16(off + 2, littleEndian),
      count: view.getUint32(off + 4, littleEndian),
      valueOffset: view.getUint32(off + 8, littleEndian),
      entryOffset: off,
    });
  }
  return out;
}

function getTypeUnitSize(type: number): number {
  switch (type) {
    case 1: // BYTE
    case 2: // ASCII
    case 7: // UNDEFINED
      return 1;
    case 3: // SHORT
      return 2;
    case 4: // LONG
    case 9: // SLONG
      return 4;
    case 5: // RATIONAL
    case 10: // SRATIONAL
      return 8;
    default:
      return 0;
  }
}

function getValueDataOffset(
  entry: IfdEntry,
  tiffStart: number,
  littleEndian: boolean,
  view: DataView,
): number | null {
  const unitSize = getTypeUnitSize(entry.type);
  if (unitSize <= 0) return null;
  const totalSize = unitSize * entry.count;
  if (totalSize <= 4) {
    // Value is stored inline at entryOffset+8.
    return entry.entryOffset + 8;
  }
  const off = tiffStart + entry.valueOffset;
  if (off < 0 || off + totalSize > view.byteLength) return null;
  return off;
}

function readShort(entry: IfdEntry, tiffStart: number, littleEndian: boolean, view: DataView): number | undefined {
  const off = getValueDataOffset(entry, tiffStart, littleEndian, view);
  if (off === null) return undefined;
  if (entry.type === 3) return view.getUint16(off, littleEndian);
  if (entry.type === 4) return view.getUint32(off, littleEndian);
  if (entry.type === 1) return view.getUint8(off);
  return undefined;
}

function readAsciiTag(entry: IfdEntry, tiffStart: number, littleEndian: boolean, view: DataView): string | undefined {
  if (entry.type !== 2 || entry.count <= 0) return undefined;
  const off = getValueDataOffset(entry, tiffStart, littleEndian, view);
  if (off === null) return undefined;
  return readAscii(view, off, entry.count);
}

function readRational(entry: IfdEntry, tiffStart: number, littleEndian: boolean, view: DataView): number | undefined {
  if (entry.count <= 0) return undefined;
  const off = getValueDataOffset(entry, tiffStart, littleEndian, view);
  if (off === null) return undefined;
  if (entry.type === 5) {
    const num = view.getUint32(off, littleEndian);
    const den = view.getUint32(off + 4, littleEndian);
    if (den === 0) return undefined;
    return num / den;
  }
  if (entry.type === 10) {
    const num = view.getInt32(off, littleEndian);
    const den = view.getInt32(off + 4, littleEndian);
    if (den === 0) return undefined;
    return num / den;
  }
  return undefined;
}

function parseExifFromJpegBuffer(buffer: ArrayBuffer): ExifMetadata {
  const view = new DataView(buffer);
  const meta: ExifMetadata = { orientation: 1 };

  if (view.byteLength < 4) return meta;
  if (view.getUint16(0, false) !== 0xFFD8) return meta; // SOI

  let offset = 2;
  while (offset + 4 <= view.byteLength) {
    if (view.getUint8(offset) !== 0xFF) break;
    const marker = view.getUint8(offset + 1);
    if (marker === 0xD9 || marker === 0xDA) break; // EOI / SOS
    const segmentLength = view.getUint16(offset + 2, false);
    if (segmentLength < 2) break;
    const segmentStart = offset + 4;
    const segmentDataLength = segmentLength - 2;
    const next = offset + 2 + segmentLength;
    if (next > view.byteLength) break;

    // APP1 EXIF
    if (marker === 0xE1 && segmentDataLength >= 8) {
      const exifHeader = readAscii(view, segmentStart, 6);
      if (exifHeader === 'Exif') {
        const tiffStart = segmentStart + 6;
        if (tiffStart + 8 > view.byteLength) break;
        const byteOrder = view.getUint16(tiffStart, false);
        const littleEndian = byteOrder === 0x4949;
        if (!littleEndian && byteOrder !== 0x4D4D) break;
        const magic = view.getUint16(tiffStart + 2, littleEndian);
        if (magic !== 42) break;
        const ifd0Rel = view.getUint32(tiffStart + 4, littleEndian);
        const ifd0 = tiffStart + ifd0Rel;
        const entries0 = readIfdEntries(view, ifd0, littleEndian);

        let exifIfdRel = 0;
        for (const e of entries0) {
          if (e.tag === 0x0112) {
            const v = readShort(e, tiffStart, littleEndian, view);
            if (v && v >= 1 && v <= 8) meta.orientation = v;
          } else if (e.tag === 0x010F) {
            const v = readAsciiTag(e, tiffStart, littleEndian, view);
            if (v) meta.make = v;
          } else if (e.tag === 0x0110) {
            const v = readAsciiTag(e, tiffStart, littleEndian, view);
            if (v) meta.model = v;
          } else if (e.tag === 0x0132) {
            const v = readAsciiTag(e, tiffStart, littleEndian, view);
            const ts = v ? parseExifDateTime(v) : undefined;
            if (ts !== undefined) meta.capturedAtMs = ts;
          } else if (e.tag === 0x8769) {
            const v = readShort(e, tiffStart, littleEndian, view);
            if (v) exifIfdRel = v;
          }
        }

        if (exifIfdRel > 0) {
          const exifIfd = tiffStart + exifIfdRel;
          const exifEntries = readIfdEntries(view, exifIfd, littleEndian);
          for (const e of exifEntries) {
            if (e.tag === 0x920A) {
              const v = readRational(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v)) meta.focalLengthMm = v;
            } else if (e.tag === 0xA405) {
              const v = readShort(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v)) meta.focalLength35mm = v;
            } else if (e.tag === 0x829D) {
              const v = readRational(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v) && v > 0) meta.apertureFNumber = v;
            } else if (e.tag === 0x829A) {
              const v = readRational(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v) && v > 0) meta.exposureTimeSec = v;
            } else if (e.tag === 0x8827 || e.tag === 0x8833) {
              const v = readShort(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v) && v > 0) meta.iso = v;
            } else if (e.tag === 0xA403) {
              const v = readShort(e, tiffStart, littleEndian, view);
              if (v === 0) meta.whiteBalanceMode = 'auto';
              else if (v === 1) meta.whiteBalanceMode = 'manual';
            } else if (e.tag === 0x9204) {
              const v = readRational(e, tiffStart, littleEndian, view);
              if (v !== undefined && Number.isFinite(v)) meta.exposureBiasEv = v;
            } else if (e.tag === 0x9003 || e.tag === 0x9004) {
              const v = readAsciiTag(e, tiffStart, littleEndian, view);
              const ts = v ? parseExifDateTime(v) : undefined;
              if (ts !== undefined) meta.capturedAtMs = ts;
            }
          }
        }

        break;
      }
    }

    offset = next;
  }

  return meta;
}

export async function parseExifMetadata(file: File): Promise<ExifMetadata> {
  const type = file.type.toLowerCase();
  const name = file.name.toLowerCase();
  // JPEG only for now; HEIC parsing is intentionally skipped.
  const isJpeg = type === 'image/jpeg' || name.endsWith('.jpg') || name.endsWith('.jpeg');
  if (!isJpeg) return { orientation: 1 };

  try {
    const buffer = await file.arrayBuffer();
    return parseExifFromJpegBuffer(buffer);
  } catch (err) {
    console.warn('EXIF parse failed for', file.name, err);
    return { orientation: 1 };
  }
}

let supportsImageOrientationNone: boolean | null = null;

async function createImageBitmapRaw(blob: Blob): Promise<ImageBitmap> {
  if (supportsImageOrientationNone !== false) {
    try {
      // Keep raw pixel order; we apply EXIF orientation ourselves.
      const bmp = await createImageBitmap(blob, { imageOrientation: 'none' });
      supportsImageOrientationNone = true;
      return bmp;
    } catch {
      supportsImageOrientationNone = false;
    }
  }
  return await createImageBitmap(blob);
}

function getOrientedDimensions(width: number, height: number, orientation: number): { width: number; height: number } {
  if (orientation >= 5 && orientation <= 8) return { width: height, height: width };
  return { width, height };
}

function applyOrientationTransform(
  ctx: CanvasRenderingContext2D,
  orientation: number,
  sourceW: number,
  sourceH: number,
): void {
  switch (orientation) {
    case 2:
      ctx.setTransform(-1, 0, 0, 1, sourceW, 0);
      break;
    case 3:
      ctx.setTransform(-1, 0, 0, -1, sourceW, sourceH);
      break;
    case 4:
      ctx.setTransform(1, 0, 0, -1, 0, sourceH);
      break;
    case 5:
      ctx.setTransform(0, 1, 1, 0, 0, 0);
      break;
    case 6:
      ctx.setTransform(0, 1, -1, 0, sourceH, 0);
      break;
    case 7:
      ctx.setTransform(0, -1, -1, 0, sourceH, sourceW);
      break;
    case 8:
      ctx.setTransform(0, -1, 1, 0, 0, sourceW);
      break;
    case 1:
    default:
      ctx.setTransform(1, 0, 0, 1, 0, 0);
      break;
  }
}

function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob((blob) => {
      if (blob) resolve(blob);
      else reject(new Error('Canvas toBlob failed'));
    }, type, quality);
  });
}

export async function normalizeImageBlobOrientation(
  sourceBlob: Blob,
  orientation: number,
): Promise<Blob> {
  const o = Number.isFinite(orientation) ? Math.round(orientation) : 1;
  if (o <= 1 || o > 8) return sourceBlob;

  const bmp = await createImageBitmapRaw(sourceBlob);
  try {
    const targetDims = getOrientedDimensions(bmp.width, bmp.height, o);
    const canvas = document.createElement('canvas');
    canvas.width = targetDims.width;
    canvas.height = targetDims.height;
    const ctx = canvas.getContext('2d');
    if (!ctx) return sourceBlob;
    applyOrientationTransform(ctx, o, bmp.width, bmp.height);
    ctx.drawImage(bmp, 0, 0);

    const srcType = sourceBlob.type && sourceBlob.type.startsWith('image/')
      ? sourceBlob.type
      : 'image/jpeg';
    const outType = srcType === 'image/png' ? 'image/png' : 'image/jpeg';
    return await canvasToBlob(canvas, outType, 0.98);
  } finally {
    bmp.close();
  }
}
