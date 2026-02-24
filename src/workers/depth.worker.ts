// depth.worker.ts - module worker using ONNX Runtime Web to run a MiDaS-like model
// Message contract:
// In: {type:'init', baseUrl, modelPath, preferWebGPU, targetSize}
// In: {type:'infer', imageId, rgbaBuffer, width, height}
// Out: {type:'progress', stage, done, total}
// Out: {type:'result', imageId, depthUint16Buffer, depthW, depthH, nearIsOne, epUsed}
// Out: {type:'error', imageId?, message}

let ort: any = null;
let session: any = null;
let epUsed: string | null = null;
let modelInputSize = 256;

/**
 * Dynamically import onnxruntime-web.
 * Uses string concatenation to prevent Vite/Rollup from resolving the import statically.
 */
async function loadOrt(variant: string): Promise<any> {
  const mod = 'onnxruntime-web';
  const path = variant ? `${mod}/dist/${variant}` : mod;
  // Use Function constructor to create a truly dynamic import that bundlers can't analyze
  const dynamicImport = new Function('p', 'return import(p)');
  return dynamicImport(path);
}

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const { baseUrl, modelPath, preferWebGPU, targetSize } = msg;
      modelInputSize = targetSize || modelInputSize;

      // Choose EP and import dynamically
      const useWebGPU = preferWebGPU && ('gpu' in navigator);
      if (useWebGPU) {
        try {
          ort = await loadOrt('ort-web.min.js');
          epUsed = 'webgpu';
        } catch { /* fall through to wasm */ }
      }
      if (!ort) {
        try {
          ort = await loadOrt('');
          epUsed = 'wasm';
        } catch {
          postMessage({ type: 'error', message: 'onnxruntime-web not available' });
          return;
        }
      }

      postMessage({ type: 'progress', stage: 'depth-init', done: 1, total: 1 });

      // Load model bytes
      const modelUrl = new URL(modelPath, baseUrl).toString();
      const resp = await fetch(modelUrl);
      if (!resp.ok) {
        postMessage({ type: 'error', message: `Model fetch failed: ${resp.status}` });
        return;
      }
      const modelArray = await resp.arrayBuffer();
      session = await ort.InferenceSession.create(modelArray, {
        executionProviders: [epUsed],
      });
      postMessage({
        type: 'progress',
        stage: 'depth-model-loaded',
        done: 1,
        total: 1,
        info: epUsed,
      });
      return;
    }

    if (msg.type === 'infer') {
      if (!ort || !session) {
        postMessage({ type: 'error', imageId: msg.imageId, message: 'Depth model not loaded' });
        return;
      }

      const { imageId, rgbaBuffer, width, height } = msg;
      const rgba = new Uint8ClampedArray(rgbaBuffer);

      // Resize to model input size
      const off = new OffscreenCanvas(modelInputSize, modelInputSize);
      const ctx = off.getContext('2d')!;
      if (width === modelInputSize && height === modelInputSize) {
        const im = new ImageData(rgba, width, height);
        ctx.putImageData(im, 0, 0);
      } else {
        const bmp = await createImageBitmap(
          new ImageData(new Uint8ClampedArray(rgba), width, height),
        );
        ctx.drawImage(bmp, 0, 0, modelInputSize, modelInputSize);
        bmp.close();
      }

      const imdata = ctx.getImageData(0, 0, modelInputSize, modelInputSize);

      // Convert to float NCHW
      const pixels = modelInputSize * modelInputSize;
      const floatData = new Float32Array(3 * pixels);
      for (let i = 0; i < pixels; i++) {
        floatData[i] = imdata.data[i * 4] / 255.0;                   // R
        floatData[pixels + i] = imdata.data[i * 4 + 1] / 255.0;     // G
        floatData[2 * pixels + i] = imdata.data[i * 4 + 2] / 255.0; // B
      }

      const tensor = new ort.Tensor('float32', floatData, [1, 3, modelInputSize, modelInputSize]);
      const inputName = session.inputNames?.[0] ?? 'input';
      const feeds: Record<string, any> = { [inputName]: tensor };
      const outputMap = await session.run(feeds);
      const outName = Object.keys(outputMap)[0];
      const depthData = outputMap[outName].data as Float32Array;

      // Normalize depth to Uint16 using min/max
      let min = Infinity;
      let max = -Infinity;
      for (const v of depthData) {
        if (v < min) min = v;
        if (v > max) max = v;
      }
      const range = Math.max(1e-6, max - min);
      const depthUint16 = new Uint16Array(depthData.length);
      for (let i = 0; i < depthData.length; i++) {
        depthUint16[i] = Math.round(((depthData[i] - min) / range) * 65535);
      }

      postMessage({
        type: 'result',
        imageId,
        depthUint16Buffer: depthUint16.buffer,
        depthW: modelInputSize,
        depthH: modelInputSize,
        nearIsOne: true,
        epUsed,
      });
      return;
    }

    postMessage({ type: 'error', message: 'unknown message' });
  } catch (err: any) {
    postMessage({
      type: 'error',
      imageId: msg && msg.imageId,
      message: err.message || err.toString(),
    });
  }
});