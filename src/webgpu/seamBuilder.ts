import type { MeshData } from '../gl';
import type { FaceRectComposite } from '../gl/composition';
import {
  buildCompactGraphFromSummaries,
  resolveCompactSummarySampleGrid,
  type CompactSeamGraphBuildResult,
  type SeamAccelerationTier,
  type SeamColorTransferStats,
} from '../gl/seam';

type GPUDeviceLike = any;
type GPUAdapterLike = any;

const GPUTextureUsageRef = (globalThis as any).GPUTextureUsage;
const GPUBufferUsageRef = (globalThis as any).GPUBufferUsage;
const GPUMapModeRef = (globalThis as any).GPUMapMode;

const WARP_VERT_WGSL = `
struct VertexIn {
  @location(0) clipPos : vec2<f32>,
  @location(1) uv : vec2<f32>,
};

struct VertexOut {
  @builtin(position) position : vec4<f32>,
  @location(0) uv : vec2<f32>,
};

@vertex
fn main(input : VertexIn) -> VertexOut {
  var out : VertexOut;
  out.position = vec4<f32>(input.clipPos, 0.0, 1.0);
  out.uv = input.uv;
  return out;
}
`;

const WARP_FRAG_WGSL = `
struct WarpParams {
  gainTone : vec4<f32>,
  vignette : vec4<f32>,
  colorGain : vec4<f32>,
  colorOffset : vec4<f32>,
};

@group(0) @binding(0) var u_sampler : sampler;
@group(0) @binding(1) var u_texture : texture_2d<f32>;
@group(0) @binding(2) var<uniform> u_params : WarpParams;

@fragment
fn main(@location(0) uv : vec2<f32>) -> @location(0) vec4<f32> {
  var c = textureSample(u_texture, u_sampler, uv);
  let centered = uv - vec2<f32>(0.5, 0.5);
  let r2 = dot(centered, centered) * 4.0;
  let r4 = r2 * r2;
  let r6 = r4 * r2;
  let vig = max(1.0 + u_params.vignette.x * r2 + u_params.vignette.y * r4 + u_params.vignette.z * r6, 0.1);
  c.rgb = c.rgb / vig;
  c.rgb = c.rgb * u_params.gainTone.xyz;
  if (u_params.gainTone.w > 0.5) {
    let white2 = 4.0;
    c.rgb = c.rgb * (1.0 + c.rgb / white2) / (1.0 + c.rgb);
  }
  if (u_params.colorGain.w > 0.5) {
    c.rgb = clamp(c.rgb * u_params.colorGain.xyz + u_params.colorOffset.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  }
  return c;
}
`;

const SUMMARY_COMP_WGSL = `
struct SummaryParams {
  dims : vec4<u32>,
  values : vec4<f32>,
};

@group(0) @binding(0) var u_texture : texture_2d<f32>;
@group(0) @binding(1) var<storage, read_write> u_mean : array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> u_sq : array<vec4<f32>>;
@group(0) @binding(3) var<uniform> u_params : SummaryParams;

@compute @workgroup_size(8, 8, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let gx = gid.x;
  let gy = gid.y;
  let width = u_params.dims.x;
  let height = u_params.dims.y;
  let gridW = u_params.dims.z;
  let gridH = u_params.dims.w;
  if (gx >= gridW || gy >= gridH) {
    return;
  }

  let blockSize = u_params.values.x;
  let alphaThreshold = u_params.values.y;
  let samples = max(1u, u32(u_params.values.z + 0.5));
  let totalSamples = f32(samples * samples);
  let blockOrigin = vec2<f32>(f32(gx) * blockSize, f32(gy) * blockSize);
  let maxX = max(0, i32(width) - 1);
  let maxY = max(0, i32(height) - 1);
  var sum = vec3<f32>(0.0, 0.0, 0.0);
  var sumSq = vec3<f32>(0.0, 0.0, 0.0);
  var count = 0.0;

  for (var sy = 0u; sy < samples; sy = sy + 1u) {
    for (var sx = 0u; sx < samples; sx = sx + 1u) {
      let offset = (vec2<f32>(f32(sx) + 0.5, f32(sy) + 0.5) / f32(samples)) * blockSize;
      let px = min(blockOrigin + offset, vec2<f32>(f32(width) - 1.0, f32(height) - 1.0));
      let ix = clamp(i32(px.x), 0, maxX);
      let iy = clamp(i32(px.y), 0, maxY);
      let sampleColor = textureLoad(u_texture, vec2<i32>(ix, iy), 0);
      if (sampleColor.a <= alphaThreshold) {
        continue;
      }
      sum += sampleColor.rgb;
      sumSq += sampleColor.rgb * sampleColor.rgb;
      count += 1.0;
    }
  }

  let index = gy * gridW + gx;
  if (count <= 0.0) {
    u_mean[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    u_sq[index] = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    return;
  }

  let coverage = count / totalSamples;
  u_mean[index] = vec4<f32>(sum / count, coverage);
  u_sq[index] = vec4<f32>(sumSq / count, coverage);
}
`;

export interface WebGPUSeamCompositeState {
  gridW: number;
  gridH: number;
  blockSize: number;
  sampleGrid: number;
  compMean: Float32Array;
  compSq: Float32Array;
}

export interface BuildWebGPUSeamGraphArgs {
  imageId: string;
  imageFile: File;
  sourceWidth: number;
  sourceHeight: number;
  width: number;
  height: number;
  blockSize: number;
  mesh: MeshData;
  viewMatrix: Float32Array;
  gain: [number, number, number];
  vignette?: { a: number; b: number; c: number };
  toneMap: boolean;
  faceRects?: FaceRectComposite[];
  saliencyGrid?: Float32Array | null;
  tier: SeamAccelerationTier;
  compositeState: WebGPUSeamCompositeState;
  colorTransfer?: Pick<SeamColorTransferStats, 'gain' | 'offset'> | null;
}

export interface WebGPUSeamBuilder {
  available(): boolean;
  buildCompactGraph(args: BuildWebGPUSeamGraphArgs): Promise<CompactSeamGraphBuildResult | null>;
  dispose(): void;
}

interface RenderSummariesArgs {
  imageId: string;
  imageFile: File;
  sourceWidth: number;
  sourceHeight: number;
  width: number;
  height: number;
  blockSize: number;
  mesh: MeshData;
  viewMatrix: Float32Array;
  gain: [number, number, number];
  vignette?: { a: number; b: number; c: number };
  toneMap: boolean;
  tier: SeamAccelerationTier;
  colorTransfer?: Pick<SeamColorTransferStats, 'gain' | 'offset'> | null;
}

interface WebGPUContext {
  device: GPUDeviceLike;
  sampler: any;
  warpPipeline: any;
  summaryPipeline: any;
  sourceCache: Map<string, any>;
}

function isWebGPUUsable(): boolean {
  return !!(navigator as Navigator & { gpu?: unknown }).gpu && !!GPUTextureUsageRef && !!GPUBufferUsageRef && !!GPUMapModeRef;
}

function buildSourceCacheKey(imageId: string, width: number, height: number): string {
  return `${imageId}:${width}x${height}`;
}

function buildWarpParamsArray(
  gain: [number, number, number],
  vignette: { a: number; b: number; c: number } | undefined,
  toneMap: boolean,
  colorTransfer: Pick<SeamColorTransferStats, 'gain' | 'offset'> | null | undefined,
): Float32Array {
  const colorGain = colorTransfer?.gain ?? [1, 1, 1];
  const colorOffset = colorTransfer?.offset ?? [0, 0, 0];
  return new Float32Array([
    gain[0], gain[1], gain[2], toneMap ? 1 : 0,
    vignette?.a ?? 0, vignette?.b ?? 0, vignette?.c ?? 0, 0,
    colorGain[0], colorGain[1], colorGain[2], colorTransfer ? 1 : 0,
    colorOffset[0], colorOffset[1], colorOffset[2], 0,
  ]);
}

function buildSummaryParamsBuffer(
  device: GPUDeviceLike,
  width: number,
  height: number,
  gridW: number,
  gridH: number,
  blockSize: number,
  sampleGrid: number,
): any {
  const raw = new ArrayBuffer(32);
  const u32 = new Uint32Array(raw, 0, 4);
  u32[0] = width;
  u32[1] = height;
  u32[2] = gridW;
  u32[3] = gridH;
  const f32 = new Float32Array(raw, 16, 4);
  f32[0] = blockSize;
  f32[1] = 10 / 255;
  f32[2] = sampleGrid;
  f32[3] = 0;
  const buffer = device.createBuffer({
    size: raw.byteLength,
    usage: GPUBufferUsageRef.UNIFORM | GPUBufferUsageRef.COPY_DST,
  });
  device.queue.writeBuffer(buffer, 0, raw);
  return buffer;
}

function buildClipVertexData(mesh: MeshData, viewMatrix: Float32Array): Float32Array {
  const vertexCount = mesh.positions.length >> 1;
  const out = new Float32Array(vertexCount * 4);
  for (let i = 0; i < vertexCount; i++) {
    const posOff = i * 2;
    const x = mesh.positions[posOff];
    const y = mesh.positions[posOff + 1];
    const clipX = viewMatrix[0] * x + viewMatrix[3] * y + viewMatrix[6];
    const clipY = viewMatrix[1] * x + viewMatrix[4] * y + viewMatrix[7];
    const outOff = i * 4;
    out[outOff] = clipX;
    out[outOff + 1] = clipY;
    out[outOff + 2] = mesh.uvs[posOff];
    out[outOff + 3] = mesh.uvs[posOff + 1];
  }
  return out;
}

async function createScaledSourceCanvas(file: File, width: number, height: number): Promise<OffscreenCanvas> {
  const bmp = await createImageBitmap(file);
  try {
    const canvas = new OffscreenCanvas(width, height);
    const ctx = canvas.getContext('2d');
    if (!ctx) throw new Error('Failed to create 2D canvas for WebGPU source scaling');
    ctx.drawImage(bmp, 0, 0, width, height);
    return canvas;
  } finally {
    bmp.close();
  }
}

async function mapFloat32Buffer(device: GPUDeviceLike, source: any, byteLength: number): Promise<Float32Array> {
  const staging = device.createBuffer({
    size: byteLength,
    usage: GPUBufferUsageRef.COPY_DST | GPUBufferUsageRef.MAP_READ,
  });
  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, staging, 0, byteLength);
  device.queue.submit([encoder.finish()]);
  await staging.mapAsync(GPUMapModeRef.READ);
  const mapped = staging.getMappedRange();
  const copy = new Float32Array(mapped.slice(0));
  staging.unmap();
  staging.destroy();
  return copy;
}

export function createWebGPUSeamBuilder(): WebGPUSeamBuilder {
  let initPromise: Promise<WebGPUContext | null> | null = null;

  async function ensureContext(): Promise<WebGPUContext | null> {
    if (initPromise) return initPromise;
    initPromise = (async () => {
      if (!isWebGPUUsable()) return null;
      const gpu = (navigator as Navigator & { gpu?: any }).gpu;
      const adapter: GPUAdapterLike | null = await gpu.requestAdapter();
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const sampler = device.createSampler({
        magFilter: 'linear',
        minFilter: 'linear',
        mipmapFilter: 'linear',
        addressModeU: 'clamp-to-edge',
        addressModeV: 'clamp-to-edge',
      });
      const warpPipeline = device.createRenderPipeline({
        layout: 'auto',
        vertex: {
          module: device.createShaderModule({ code: WARP_VERT_WGSL }),
          entryPoint: 'main',
          buffers: [
            {
              arrayStride: Float32Array.BYTES_PER_ELEMENT * 4,
              attributes: [
                { shaderLocation: 0, offset: 0, format: 'float32x2' },
                { shaderLocation: 1, offset: Float32Array.BYTES_PER_ELEMENT * 2, format: 'float32x2' },
              ],
            },
          ],
        },
        fragment: {
          module: device.createShaderModule({ code: WARP_FRAG_WGSL }),
          entryPoint: 'main',
          targets: [{ format: 'rgba8unorm' }],
        },
        primitive: {
          topology: 'triangle-list',
          cullMode: 'none',
        },
      });
      const summaryPipeline = device.createComputePipeline({
        layout: 'auto',
        compute: {
          module: device.createShaderModule({ code: SUMMARY_COMP_WGSL }),
          entryPoint: 'main',
        },
      });
      return {
        device,
        sampler,
        warpPipeline,
        summaryPipeline,
        sourceCache: new Map(),
      };
    })();
    return initPromise;
  }

  async function ensureSourceTexture(
    ctx: WebGPUContext,
    imageId: string,
    imageFile: File,
    width: number,
    height: number,
  ): Promise<any> {
    const key = buildSourceCacheKey(imageId, width, height);
    const cached = ctx.sourceCache.get(key);
    if (cached) return cached;
    const sourceCanvas = await createScaledSourceCanvas(imageFile, width, height);
    const texture = ctx.device.createTexture({
      size: [width, height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsageRef.TEXTURE_BINDING | GPUTextureUsageRef.COPY_DST | GPUTextureUsageRef.RENDER_ATTACHMENT,
    });
    ctx.device.queue.copyExternalImageToTexture(
      { source: sourceCanvas },
      { texture },
      [width, height],
    );
    ctx.sourceCache.set(key, texture);
    return texture;
  }

  async function renderCandidateSummaries(args: RenderSummariesArgs): Promise<{
    gridW: number;
    gridH: number;
    sampleGrid: number;
    mean: Float32Array;
    sq: Float32Array;
    summaryMs: number;
    readbackBytes: number;
  } | null> {
    const ctx = await ensureContext();
    if (!ctx) return null;

    const summaryStart = performance.now();
    const sampleGrid = resolveCompactSummarySampleGrid(args.tier);
    const gridW = Math.max(1, Math.ceil(args.width / args.blockSize));
    const gridH = Math.max(1, Math.ceil(args.height / args.blockSize));
    const sourceTex = await ensureSourceTexture(ctx, args.imageId, args.imageFile, args.sourceWidth, args.sourceHeight);
    const renderTex = ctx.device.createTexture({
      size: [args.width, args.height, 1],
      format: 'rgba8unorm',
      usage: GPUTextureUsageRef.RENDER_ATTACHMENT | GPUTextureUsageRef.TEXTURE_BINDING,
    });

    const vertexData = buildClipVertexData(args.mesh, args.viewMatrix);
    const vertexBuffer = ctx.device.createBuffer({
      size: vertexData.byteLength,
      usage: GPUBufferUsageRef.VERTEX | GPUBufferUsageRef.COPY_DST,
      mappedAtCreation: true,
    });
    new Float32Array(vertexBuffer.getMappedRange()).set(vertexData);
    vertexBuffer.unmap();

    const indexBuffer = ctx.device.createBuffer({
      size: args.mesh.indices.byteLength,
      usage: GPUBufferUsageRef.INDEX | GPUBufferUsageRef.COPY_DST,
      mappedAtCreation: true,
    });
    new Uint32Array(indexBuffer.getMappedRange()).set(args.mesh.indices);
    indexBuffer.unmap();

    const warpParams = buildWarpParamsArray(args.gain, args.vignette, args.toneMap, args.colorTransfer);
    const warpParamsBuffer = ctx.device.createBuffer({
      size: warpParams.byteLength,
      usage: GPUBufferUsageRef.UNIFORM | GPUBufferUsageRef.COPY_DST,
    });
    ctx.device.queue.writeBuffer(warpParamsBuffer, 0, warpParams);

    const renderBindGroup = ctx.device.createBindGroup({
      layout: ctx.warpPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: ctx.sampler },
        { binding: 1, resource: sourceTex.createView() },
        { binding: 2, resource: { buffer: warpParamsBuffer } },
      ],
    });

    const summaryBufferSize = gridW * gridH * 4 * Float32Array.BYTES_PER_ELEMENT;
    const meanBuffer = ctx.device.createBuffer({
      size: summaryBufferSize,
      usage: GPUBufferUsageRef.STORAGE | GPUBufferUsageRef.COPY_SRC,
    });
    const sqBuffer = ctx.device.createBuffer({
      size: summaryBufferSize,
      usage: GPUBufferUsageRef.STORAGE | GPUBufferUsageRef.COPY_SRC,
    });
    const summaryParamsBuffer = buildSummaryParamsBuffer(ctx.device, args.width, args.height, gridW, gridH, args.blockSize, sampleGrid);
    const summaryBindGroup = ctx.device.createBindGroup({
      layout: ctx.summaryPipeline.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: renderTex.createView() },
        { binding: 1, resource: { buffer: meanBuffer } },
        { binding: 2, resource: { buffer: sqBuffer } },
        { binding: 3, resource: { buffer: summaryParamsBuffer } },
      ],
    });

    const encoder = ctx.device.createCommandEncoder();
    const renderPass = encoder.beginRenderPass({
      colorAttachments: [
        {
          view: renderTex.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 0 },
          loadOp: 'clear',
          storeOp: 'store',
        },
      ],
    });
    renderPass.setPipeline(ctx.warpPipeline);
    renderPass.setBindGroup(0, renderBindGroup);
    renderPass.setVertexBuffer(0, vertexBuffer);
    renderPass.setIndexBuffer(indexBuffer, 'uint32');
    renderPass.drawIndexed(args.mesh.indices.length, 1, 0, 0, 0);
    renderPass.end();

    const computePass = encoder.beginComputePass();
    computePass.setPipeline(ctx.summaryPipeline);
    computePass.setBindGroup(0, summaryBindGroup);
    computePass.dispatchWorkgroups(Math.ceil(gridW / 8), Math.ceil(gridH / 8));
    computePass.end();

    ctx.device.queue.submit([encoder.finish()]);
    if (typeof ctx.device.queue.onSubmittedWorkDone === 'function') {
      await ctx.device.queue.onSubmittedWorkDone();
    }

    const mean = await mapFloat32Buffer(ctx.device, meanBuffer, summaryBufferSize);
    const sq = await mapFloat32Buffer(ctx.device, sqBuffer, summaryBufferSize);

    renderTex.destroy();
    vertexBuffer.destroy();
    indexBuffer.destroy();
    warpParamsBuffer.destroy();
    meanBuffer.destroy();
    sqBuffer.destroy();
    summaryParamsBuffer.destroy();

    return {
      gridW,
      gridH,
      sampleGrid,
      mean,
      sq,
      summaryMs: performance.now() - summaryStart,
      readbackBytes: mean.byteLength + sq.byteLength,
    };
  }

  return {
    available() {
      return isWebGPUUsable();
    },

    async buildCompactGraph(args) {
      const expectedGridW = Math.max(1, Math.ceil(args.width / args.blockSize));
      const expectedGridH = Math.max(1, Math.ceil(args.height / args.blockSize));
      if (
        args.compositeState.blockSize !== args.blockSize
        || args.compositeState.gridW !== expectedGridW
        || args.compositeState.gridH !== expectedGridH
      ) {
        return null;
      }
      const summaries = await renderCandidateSummaries(args);
      if (!summaries) return null;
      return buildCompactGraphFromSummaries({
        width: args.width,
        height: args.height,
        blockSize: args.blockSize,
        sampleGrid: summaries.sampleGrid,
        compMean: new Float32Array(args.compositeState.compMean),
        compSq: new Float32Array(args.compositeState.compSq),
        newMean: summaries.mean,
        newSq: summaries.sq,
        faceRects: args.faceRects,
        saliencyGrid: args.saliencyGrid,
        summaryMs: summaries.summaryMs,
        readbackBytes: summaries.readbackBytes,
        backendId: 'compact-webgpu-grid',
      });
    },

    dispose() {
      if (!initPromise) return;
      void initPromise.then((ctx) => {
        if (!ctx) return;
        for (const texture of ctx.sourceCache.values()) {
          texture.destroy();
        }
        ctx.sourceCache.clear();
      });
      initPromise = null;
    },
  };
}
