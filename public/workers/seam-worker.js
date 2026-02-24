// seam-worker.js - classic worker that loads a WASM maxflow implementation and exposes a solve() API.
// Expected messages:
//  - {type:'init', baseUrl, maxflowPath}
//  - {type:'solve', jobId, gridW, gridH, dataCostsBuffer, edgeWeightsBuffer?, hardConstraintsBuffer, params}
// Replies:
//  - {type:'result', jobId, labelsBuffer}
//  - {type:'error', jobId?, message}

// This file is a skeleton. Implement the actual WASM loader and call patterns matching your chosen maxflow build.

let maxflowReady = false;
let Maxflow = null;

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const path = new URL(msg.maxflowPath, msg.baseUrl).toString();
      importScripts(path); // expects this script to register e.g. createMaxflowModule()
      // Wait for global factory
      const start = Date.now();
      while (typeof createMaxflowModule === 'undefined') {
        await new Promise(r => setTimeout(r, 50));
        if (Date.now() - start > 15000) throw new Error('maxflow wasm not found');
      }
      Maxflow = await createMaxflowModule();
      maxflowReady = true;
      postMessage({type:'progress', stage:'seam-init', percent:100, info:'maxflow ready'});
      return;
    }

    if (!maxflowReady) throw new Error('seam-worker not initialized');

    if (msg.type === 'solve') {
      const {jobId, gridW, gridH, dataCostsBuffer, hardConstraintsBuffer, params} = msg;
      // dataCostsBuffer expected to contain per-node cost info (e.g., colorDiff, gradDiff, depth)
      const costs = new Float32Array(dataCostsBuffer);
      const hard = hardConstraintsBuffer ? new Uint8Array(hardConstraintsBuffer) : null;

      // TODO: construct graph in wasm memory, run maxflow/mincut, read labels back into Uint8Array
      // For now, return a trivial label array (prefer existing composite)
      const labels = new Uint8Array(gridW * gridH);
      postMessage({type:'result', jobId, labelsBuffer: labels.buffer}, [labels.buffer]);
      return;
    }

    postMessage({type:'error', message:'unknown message'});
  } catch (err) {
    postMessage({type:'error', jobId: msg.jobId, message: err.message || err.toString()});
  }
}, false);
