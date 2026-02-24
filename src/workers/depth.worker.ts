// depth.worker.ts - module worker using ONNX Runtime Web to run a MiDaS-like model
// Message contract:
// In: {type:'init', baseUrl, modelPath, preferWebGPU, targetSize}
// In: {type:'infer', imageId, rgbaBuffer, width, height}
// Out: {type:'progress', stage, done, total}
// Out: {type:'result', imageId, depthUint16Buffer, depthW, depthH, nearIsOne, epUsed}
// Out: {type:'error', imageId?, message}

import type { Tensor } from 'onnxruntime-web';

let ort: any = null;
let session: any = null;
let epUsed: string | null = null;
let modelInputSize = 256;

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const { baseUrl, modelPath, preferWebGPU, targetSize } = msg;
      modelInputSize = targetSize || modelInputSize;
      // Choose EP and import dynamically
      const useWebGPU = preferWebGPU && ('gpu' in navigator);
      if (useWebGPU) {
        // webgpu EP requires the webgpu build - in production host locally under /public/ort/
        ort = await import('onnxruntime-web/dist/ort-web.min.js').catch(()=>null);
        epUsed = 'webgpu';
      }
      if (!ort) {
        ort = await import('onnxruntime-web').catch(()=>null);
        epUsed = 'wasm';
      }
      postMessage({type:'progress', stage:'depth-init', done:1, total:1});
      // Load model bytes
      const modelUrl = new URL(modelPath, baseUrl).toString();
      const resp = await fetch(modelUrl);
      const modelArray = await resp.arrayBuffer();
      session = await ort.InferenceSession.create(modelArray, { executionProviders: [epUsed] });
      postMessage({type:'progress', stage:'depth-model-loaded', done:1, total:1, info: epUsed});
      return;
    }

    if (msg.type === 'infer') {
      const { imageId, rgbaBuffer, width, height } = msg;
      // rgbaBuffer expected to be Uint8ClampedArray pixels in RGBA order.
      const rgba = new Uint8ClampedArray(rgbaBuffer);
      // Create an OffscreenCanvas to resize the image to modelInputSize
      const off = new OffscreenCanvas(modelInputSize, modelInputSize);
      const ctx = off.getContext('2d');
      // Create ImageData from rgba if sizes match; otherwise caller should supply an ImageBitmap in future.
      if (width === modelInputSize && height === modelInputSize) {
        const im = new ImageData(rgba, width, height);
        ctx.putImageData(im, 0, 0);
      } else {
        // Draw via bitmap
        const blob = new Blob([rgba], { type: 'image/png' });
        const imgBitmap = await createImageBitmap(blob);
        ctx.drawImage(imgBitmap, 0, 0, modelInputSize, modelInputSize);
      }
      const imdata = ctx.getImageData(0,0,modelInputSize,modelInputSize);
      // Convert to float and normalize (model-specific -- adjust if needed)
      const floatData = new Float32Array(modelInputSize * modelInputSize * 3);
      let p=0;
      for (let i=0;i<imdata.data.length;i+=4) {
        floatData[p++] = imdata.data[i] / 255.0;
        floatData[p++] = imdata.data[i+1] / 255.0;
        floatData[p++] = imdata.data[i+2] / 255.0;
      }
      // Create tensor in NCHW layout if model expects that
      const tensor = new ort.Tensor('float32', floatData, [1,3,modelInputSize,modelInputSize]);
      const feeds: any = {}; // adapt name to actual model input
      // Attempt to detect input name
      const inputNames = session.inputNames || Object.keys(session.inputNames || {});
      const firstInput = session.inputNames && session.inputNames[0] ? session.inputNames[0] : null;
      if (firstInput) feeds[firstInput] = tensor;
      else feeds['input'] = tensor;
      const outputMap = await session.run(feeds);
      const outName = Object.keys(outputMap)[0];
      const depthData = outputMap[outName].data as Float32Array;
      // Normalize depth to 0..1 using percentiles (approximate with min/max here)
      let min=Infinity, max=-Infinity;
      for (let v of depthData) { if (v<min) min=v; if (v>max) max=v; }
      const range = Math.max(1e-6, max-min);
      const depthUint16 = new Uint16Array(depthData.length);
      for (let i=0;i<depthData.length;i++) {
        const n = (depthData[i]-min)/range;
        depthUint16[i] = Math.round(n * 65535);
      }
      postMessage({type:'result', imageId, depthUint16Buffer: depthUint16.buffer, depthW: modelInputSize, depthH: modelInputSize, nearIsOne: true, epUsed});
      return;
    }

    postMessage({type:'error', message: 'unknown message'});
  } catch (err) {
    postMessage({type:'error', imageId: msg && msg.imageId, message: err.message || err.toString()});
  }
});