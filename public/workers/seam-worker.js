/**
 * seam-worker.js — Graph-cut seam solver running in a Web Worker.
 *
 * Primary path:
 *  - compact seam graph IR from main thread
 *  - external WASM candidate when available
 *  - typed-array push-relabel maxflow fallback
 *
 * Recovery path:
 *  - legacy Edmonds-Karp solver for explicit fallback/debug use
 *
 * Messages in:
 *  - {type:'init', baseUrl, maxflowPath?, wasmPathSimd?, wasmPathThreads?, wasmWorkerPathThreads?}
 *  - {type:'solve', jobId, gridW, gridH, dataCostsBuffer, edgeWeightsHBuffer?, edgeWeightsVBuffer?, hardConstraintsBuffer, params}
 *
 * Messages out:
 *  - {type:'progress', stage, percent, info, backendId}
 *  - {type:'result', jobId, labelsBuffer, backendId, solverMs, pushes, relabels, globalRelabels}
 *  - {type:'error', jobId?, message}
 */

let ready = false;
let solverBackendId = 'js-push-relabel';
let solverInitDetail = 'using built-in JS solver';
let externalSolver = null;

function detectWasmSimd() {
  try {
    return WebAssembly.validate(new Uint8Array([
      0x00, 0x61, 0x73, 0x6d,
      0x01, 0x00, 0x00, 0x00,
      0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b,
      0x03, 0x02, 0x01, 0x00,
      0x0a, 0x0a, 0x01, 0x08, 0x00, 0x41, 0x00, 0xfd, 0x0f, 0x0b,
    ]));
  } catch {
    return false;
  }
}

function deriveWasmBinaryUrl(scriptUrl) {
  if (!scriptUrl) return '';
  return scriptUrl.replace(/\.js(\?.*)?$/, '.wasm$1');
}

async function tryLoadExternalWasmCandidate(kind, scriptUrl, workerUrl) {
  if (!scriptUrl) return { ok: false, reason: `${kind} candidate missing script URL` };

  try {
    importScripts(scriptUrl);
  } catch (err) {
    return { ok: false, reason: `${kind} loader import failed: ${err?.message || err}` };
  }

  const loaders = self.__brenizerMaxflowAssetLoaders || {};
  const loader = loaders[kind];
  if (typeof loader !== 'function') {
    return { ok: false, reason: `${kind} loader did not register a factory` };
  }

  try {
    const solver = await loader({
      scriptUrl,
      wasmUrl: deriveWasmBinaryUrl(scriptUrl),
      workerUrl,
      crossOriginIsolated: self.crossOriginIsolated === true,
    });
    if (!solver || typeof solver.solve !== 'function') {
      return { ok: false, reason: `${kind} loader returned no solver implementation` };
    }
    return {
      ok: true,
      solver,
      backendId: solver.backendId || `wasm-${kind}`,
      detail: solver.description || `loaded ${kind} candidate`,
    };
  } catch (err) {
    return { ok: false, reason: `${kind} candidate unavailable: ${err?.message || err}` };
  }
}

async function initializeSolverBackend(msg) {
  externalSolver = null;
  solverBackendId = 'js-push-relabel';
  solverInitDetail = 'using built-in JS solver';

  const simdScript = msg.wasmPathSimd || msg.maxflowPath || '';
  const threadsScript = msg.wasmPathThreads || '';
  const threadsWorkerScript = msg.wasmWorkerPathThreads || '';
  const attempts = [];

  if (self.crossOriginIsolated && typeof SharedArrayBuffer !== 'undefined' && threadsScript) {
    attempts.push(await tryLoadExternalWasmCandidate('threads', threadsScript, threadsWorkerScript));
  }
  if (detectWasmSimd() && simdScript) {
    attempts.push(await tryLoadExternalWasmCandidate('simd', simdScript, ''));
  }

  const loaded = attempts.find((attempt) => attempt.ok);
  if (loaded) {
    externalSolver = loaded.solver;
    solverBackendId = loaded.backendId;
    solverInitDetail = loaded.detail;
    return;
  }

  const failureReason = attempts.find((attempt) => attempt.reason)?.reason;
  if (failureReason) {
    solverInitDetail = `${solverInitDetail}; ${failureReason}`;
  }
}

async function solveMinCutExternal(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, onProgress, progressEveryMs) {
  if (!externalSolver || typeof externalSolver.solve !== 'function') {
    throw new Error('External solver requested without a loaded implementation');
  }

  onProgress('seam-solve-external', 55, `Using ${solverBackendId}`);
  const startedAt = performance.now();
  const result = await externalSolver.solve({
    gridW,
    gridH,
    dataCosts,
    edgeWeightsH,
    edgeWeightsV,
    hardConstraints: hard,
    progressEveryMs,
  }, onProgress);
  const labels = result?.labels instanceof Uint8Array
    ? result.labels
    : result?.labels
      ? new Uint8Array(result.labels)
      : null;
  if (!labels) {
    throw new Error('External solver returned no label buffer');
  }
  return {
    labels,
    backendId: result.backendId || solverBackendId,
    solverMs: Number(result.solverMs) || Math.round(performance.now() - startedAt),
    pushes: Number(result.pushes) || 0,
    relabels: Number(result.relabels) || 0,
    globalRelabels: Number(result.globalRelabels) || 0,
  };
}

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      await initializeSolverBackend(msg);
      ready = true;
      postMessage({
        type: 'progress',
        stage: 'seam-init',
        percent: 100,
        info: `maxflow ready (${solverBackendId}; ${solverInitDetail})`,
        backendId: solverBackendId,
      });
      return;
    }

    if (msg.type === 'solve') {
      const {
        jobId,
        gridW,
        gridH,
        dataCostsBuffer,
        edgeWeightsHBuffer,
        edgeWeightsVBuffer,
        edgeWeightsBuffer,
        hardConstraintsBuffer,
        params,
      } = msg;
      const solveParams = params || {};
      const progressEveryMs = Math.max(250, Number(solveParams.progressEveryMs) || 1200);
      const startedAt = performance.now();
      const preferLegacy = !!solveParams.forceLegacy;
      const activeBackendId = preferLegacy ? 'legacy-edmonds-karp' : solverBackendId;
      const emitProgress = (stage, percent, info, extra = {}) => {
        postMessage({
          type: 'progress',
          stage,
          percent,
          info,
          jobId,
          backendId: activeBackendId,
          elapsedMs: Math.round(performance.now() - startedAt),
          ...extra,
        });
      };

      emitProgress('seam-solve-start', 0, `Solving seam ${gridW}×${gridH}`);

      const dataCosts = new Float32Array(dataCostsBuffer);
      const nHorizontal = Math.max(0, (gridW - 1) * gridH);
      const edgeWeightsH = edgeWeightsHBuffer
        ? new Float32Array(edgeWeightsHBuffer)
        : edgeWeightsBuffer
          ? new Float32Array(edgeWeightsBuffer, 0, nHorizontal)
          : null;
      const edgeWeightsV = edgeWeightsVBuffer
        ? new Float32Array(edgeWeightsVBuffer)
        : edgeWeightsBuffer
          ? new Float32Array(
              edgeWeightsBuffer,
              Float32Array.BYTES_PER_ELEMENT * nHorizontal,
              Math.max(0, gridW * (gridH - 1)),
            )
          : null;
      const hard = hardConstraintsBuffer ? new Uint8Array(hardConstraintsBuffer) : null;

      const solveResult = preferLegacy
        ? solveMinCutLegacy(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, emitProgress, progressEveryMs)
        : externalSolver
          ? await solveMinCutExternal(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, emitProgress, progressEveryMs)
          : solveMinCutPushRelabel(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, emitProgress, progressEveryMs);

      emitProgress('seam-solve-done', 100, 'Seam solve complete', {
        solverMs: solveResult.solverMs,
        augments: solveResult.pushes,
      });
      postMessage({
        type: 'result',
        jobId,
        labelsBuffer: solveResult.labels.buffer,
        backendId: solveResult.backendId,
        solverMs: solveResult.solverMs,
        pushes: solveResult.pushes,
        relabels: solveResult.relabels,
        globalRelabels: solveResult.globalRelabels,
      }, [solveResult.labels.buffer]);
      return;
    }

    postMessage({ type: 'error', message: 'unknown message type: ' + msg.type });
  } catch (err) {
    postMessage({ type: 'error', jobId: msg.jobId, message: err.message || err.toString() });
  }
}, false);

function solveMinCutPushRelabel(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, onProgress, progressEveryMs) {
  const n = gridW * gridH;
  const S = n;
  const T = n + 1;
  const totalNodes = n + 2;
  const totalEdgesEstimate = Math.max(0, (gridW - 1) * gridH + gridW * (gridH - 1));
  let lastBuildProgressAt = performance.now();

  function maybeBuildProgress(stage, processed, total, startPercent, spanPercent) {
    const now = performance.now();
    if (now - lastBuildProgressAt >= progressEveryMs) {
      lastBuildProgressAt = now;
      const phasePct = total > 0 ? Math.min(1, processed / total) : 0;
      onProgress(stage, startPercent + spanPercent * phasePct, `Graph build ${Math.round(phasePct * 100)}%`, { augments: 0 });
    }
  }

  const maxStEdges = n * 2;
  const maxNLinkEdges = totalEdgesEstimate * 4;
  const maxEdges = Math.max(8, maxStEdges * 2 + maxNLinkEdges);
  const head = new Int32Array(totalNodes);
  head.fill(-1);
  const to = new Int32Array(maxEdges);
  const next = new Int32Array(maxEdges);
  const rev = new Int32Array(maxEdges);
  const cap = new Float32Array(maxEdges);
  let edgeCount = 0;

  function addEdge(from, target, capacity) {
    const fwd = edgeCount++;
    const back = edgeCount++;

    to[fwd] = target;
    cap[fwd] = capacity;
    next[fwd] = head[from];
    rev[fwd] = back;
    head[from] = fwd;

    to[back] = from;
    cap[back] = 0;
    next[back] = head[target];
    rev[back] = fwd;
    head[target] = back;
  }

  function addBiEdge(a, b, w) {
    addEdge(a, b, w);
    addEdge(b, a, w);
  }

  for (let i = 0; i < n; i++) {
    const costSource = dataCosts[i * 2];
    const costSink = dataCosts[i * 2 + 1];
    let capS = costSink;
    let capT = costSource;
    if (hard) {
      if (hard[i] === 1) { capS = 1e9; capT = 0; }
      else if (hard[i] === 2) { capS = 0; capT = 1e9; }
    }
    if (capS > 0) addEdge(S, i, capS);
    if (capT > 0) addEdge(i, T, capT);
    if ((i & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-data', i, n, 5, 25);
  }
  maybeBuildProgress('seam-solve-build-data', n, n, 5, 25);

  let processedEdges = 0;
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const a = y * gridW + x;
      const b = a + 1;
      const eIdx = y * (gridW - 1) + x;
      addBiEdge(a, b, edgeWeightsH ? edgeWeightsH[eIdx] : 1.0);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, 30, 25);
    }
  }
  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW; x++) {
      const a = y * gridW + x;
      const b = a + gridW;
      const eIdx = y * gridW + x;
      addBiEdge(a, b, edgeWeightsV ? edgeWeightsV[eIdx] : 1.0);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, 30, 25);
    }
  }
  maybeBuildProgress('seam-solve-build-edges', totalEdgesEstimate, totalEdgesEstimate, 30, 25);
  onProgress('seam-solve-graph', 55, `Graph ready (${n} nodes)`);

  const solverStartedAt = performance.now();
  const flowStats = pushRelabel(
    head,
    to,
    next,
    rev,
    cap,
    S,
    T,
    totalNodes,
    onProgress,
    progressEveryMs,
  );
  const solverMs = Math.round(performance.now() - solverStartedAt);

  const labels = extractLabelsFromResidual(head, to, next, cap, S, totalNodes, n);
  onProgress('seam-solve-labels', 98, `Labels built (${flowStats.pushes} pushes)`, {
    augments: flowStats.pushes,
    remainingMs: 0,
    solverMs,
  });

  return {
    labels,
    backendId: solverBackendId,
    solverMs,
    pushes: flowStats.pushes,
    relabels: flowStats.relabels,
    globalRelabels: flowStats.globalRelabels,
  };
}

function pushRelabel(head, to, next, rev, cap, S, T, totalNodes, onProgress, progressEveryMs) {
  const eps = 1e-6;
  const height = new Int32Array(totalNodes);
  const excess = new Float32Array(totalNodes);
  const current = new Int32Array(totalNodes);
  const active = new Uint8Array(totalNodes);
  const queue = new Int32Array(Math.max(8, totalNodes * 4));
  let qHead = 0;
  let qTail = 0;
  let pushes = 0;
  let relabels = 0;
  let globalRelabels = 0;
  let workUnits = 0;
  let lastProgressAt = performance.now();
  const startedAt = performance.now();

  function enqueue(v) {
    if (v === S || v === T || active[v] || excess[v] <= eps) return;
    active[v] = 1;
    queue[qTail++] = v;
  }

  function maybeProgress() {
    const now = performance.now();
    if (now - lastProgressAt >= progressEveryMs) {
      lastProgressAt = now;
      const elapsedMs = Math.max(1, now - startedAt);
      const progress = Math.min(0.98, workUnits / Math.max(1, totalNodes * 6));
      const remainingMs = progress > 0 ? Math.round((elapsedMs * (1 - progress)) / progress) : null;
      onProgress('seam-solve-maxflow', 55 + progress * 40, `Push-relabel ${Math.round(progress * 100)}% (${pushes} pushes, ${relabels} relabels)`, {
        augments: pushes,
        expectedAugments: Math.max(pushes + 1, totalNodes * 2),
        remainingMs,
        solverMs: Math.round(elapsedMs),
      });
    }
  }

  function relabel(v) {
    let minHeight = 0x7fffffff;
    for (let ei = head[v]; ei !== -1; ei = next[ei]) {
      if (cap[ei] <= eps) continue;
      const h = height[to[ei]];
      if (h < minHeight) minHeight = h;
    }
    if (minHeight < 0x7fffffff) {
      height[v] = minHeight + 1;
      relabels++;
    }
  }

  function globalRelabel() {
    const bfsQueue = new Int32Array(totalNodes);
    let bfsHead = 0;
    let bfsTail = 0;
    height.fill(totalNodes + 1);
    height[T] = 0;
    bfsQueue[bfsTail++] = T;
    while (bfsHead < bfsTail) {
      const u = bfsQueue[bfsHead++];
      for (let ei = head[u]; ei !== -1; ei = next[ei]) {
        const reverseEdge = rev[ei];
        const v = to[ei];
        if (cap[reverseEdge] > eps && height[v] > height[u] + 1) {
          height[v] = height[u] + 1;
          bfsQueue[bfsTail++] = v;
        }
      }
    }
    height[S] = totalNodes;
    globalRelabels++;
  }

  current.set(head);
  globalRelabel();

  for (let ei = head[S]; ei !== -1; ei = next[ei]) {
    const pushed = cap[ei];
    if (pushed <= eps) continue;
    cap[ei] = 0;
    cap[rev[ei]] += pushed;
    excess[to[ei]] += pushed;
    excess[S] -= pushed;
    pushes++;
    enqueue(to[ei]);
  }

  let sinceGlobalRelabel = 0;
  while (qHead < qTail) {
    const v = queue[qHead++];
    active[v] = 0;
    while (excess[v] > eps) {
      let advanced = false;
      for (let ei = current[v]; ei !== -1; ei = next[ei]) {
        current[v] = ei;
        const u = to[ei];
        if (cap[ei] <= eps || height[v] !== height[u] + 1) continue;
        const delta = Math.min(excess[v], cap[ei]);
        cap[ei] -= delta;
        cap[rev[ei]] += delta;
        excess[v] -= delta;
        excess[u] += delta;
        pushes++;
        enqueue(u);
        advanced = true;
        if (excess[v] <= eps) break;
      }
      if (excess[v] > eps) {
        relabel(v);
        current[v] = head[v];
        sinceGlobalRelabel++;
      }
      workUnits++;
      maybeProgress();
      if (!advanced && excess[v] > eps && current[v] === -1) {
        relabel(v);
        current[v] = head[v];
      }
      if (sinceGlobalRelabel >= totalNodes) {
        globalRelabel();
        current.set(head);
        sinceGlobalRelabel = 0;
      }
    }
    if (excess[v] > eps) enqueue(v);
  }

  onProgress('seam-solve-maxflow', 95, `Push-relabel done (${pushes} pushes, ${relabels} relabels)`, {
    augments: pushes,
    expectedAugments: Math.max(pushes, totalNodes * 2),
    remainingMs: 0,
    solverMs: Math.round(performance.now() - startedAt),
  });

  return { pushes, relabels, globalRelabels };
}

function extractLabelsFromResidual(head, to, next, cap, S, totalNodes, n) {
  const visited = new Uint8Array(totalNodes);
  const queue = new Int32Array(totalNodes);
  let qHead = 0;
  let qTail = 0;
  queue[qTail++] = S;
  visited[S] = 1;
  while (qHead < qTail) {
    const u = queue[qHead++];
    for (let ei = head[u]; ei !== -1; ei = next[ei]) {
      const v = to[ei];
      if (!visited[v] && cap[ei] > 1e-6) {
        visited[v] = 1;
        queue[qTail++] = v;
      }
    }
  }
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) labels[i] = visited[i] ? 0 : 1;
  return labels;
}

function solveMinCutLegacy(gridW, gridH, dataCosts, edgeWeightsH, edgeWeightsV, hard, onProgress, progressEveryMs) {
  const n = gridW * gridH;
  const S = n;
  const T = n + 1;
  const totalNodes = n + 2;
  const totalEdgesEstimate = Math.max(0, (gridW - 1) * gridH + gridW * (gridH - 1));
  let lastBuildProgressAt = performance.now();

  function maybeBuildProgress(stage, processed, total, startPercent, spanPercent) {
    const now = performance.now();
    if (now - lastBuildProgressAt >= progressEveryMs) {
      lastBuildProgressAt = now;
      const phasePct = total > 0 ? Math.min(1, processed / total) : 0;
      onProgress(stage, startPercent + spanPercent * phasePct, `Graph build ${Math.round(phasePct * 100)}%`, { augments: 0 });
    }
  }

  const adj = new Array(totalNodes);
  for (let i = 0; i < totalNodes; i++) adj[i] = [];

  function addEdge(from, to, cap) {
    adj[from].push({ to, cap, flow: 0, rev: adj[to].length });
    adj[to].push({ to: from, cap: 0, flow: 0, rev: adj[from].length - 1 });
  }

  function addEdgeBi(from, to, cap) {
    adj[from].push({ to, cap, flow: 0, rev: adj[to].length });
    adj[to].push({ to: from, cap, flow: 0, rev: adj[from].length - 1 });
  }

  for (let i = 0; i < n; i++) {
    const costSource = dataCosts[i * 2];
    const costSink = dataCosts[i * 2 + 1];
    let capS = costSink;
    let capT = costSource;
    if (hard) {
      if (hard[i] === 1) { capS = 1e9; capT = 0; }
      else if (hard[i] === 2) { capS = 0; capT = 1e9; }
    }
    if (capS > 0) addEdge(S, i, capS);
    if (capT > 0) addEdge(i, T, capT);
    if ((i & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-data', i, n, 5, 25);
  }
  maybeBuildProgress('seam-solve-build-data', n, n, 5, 25);

  let processedEdges = 0;
  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const a = y * gridW + x;
      const b = a + 1;
      const eIdx = y * (gridW - 1) + x;
      addEdgeBi(a, b, edgeWeightsH ? edgeWeightsH[eIdx] : 1.0);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, 30, 25);
    }
  }
  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW; x++) {
      const a = y * gridW + x;
      const b = a + gridW;
      const eIdx = y * gridW + x;
      addEdgeBi(a, b, edgeWeightsV ? edgeWeightsV[eIdx] : 1.0);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, 30, 25);
    }
  }
  maybeBuildProgress('seam-solve-build-edges', totalEdgesEstimate, totalEdgesEstimate, 30, 25);
  onProgress('seam-solve-graph', 55, `Graph ready (${n} nodes)`);

  const solverStartedAt = performance.now();
  const flowStats = maxflowLegacy(adj, S, T, totalNodes, n, onProgress, progressEveryMs);
  const solverMs = Math.round(performance.now() - solverStartedAt);

  const visited = new Uint8Array(totalNodes);
  const queue = [S];
  visited[S] = 1;
  let qi = 0;
  while (qi < queue.length) {
    const u = queue[qi++];
    for (const e of adj[u]) {
      if (!visited[e.to] && e.cap - e.flow > 1e-9) {
        visited[e.to] = 1;
        queue.push(e.to);
      }
    }
  }
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) labels[i] = visited[i] ? 0 : 1;

  onProgress('seam-solve-labels', 98, `Labels built (${flowStats.augments} augments)`, {
    augments: flowStats.augments,
    remainingMs: 0,
    solverMs,
  });

  return {
    labels,
    backendId: 'legacy-edmonds-karp',
    solverMs,
    pushes: flowStats.augments,
    relabels: 0,
    globalRelabels: 0,
  };
}

function maxflowLegacy(adj, S, T, totalNodes, gridNodes, onProgress, progressEveryMs) {
  let totalFlow = 0;
  const parent = new Int32Array(totalNodes);
  const parentEdge = new Int32Array(totalNodes);
  let augments = 0;
  let lastProgressAt = performance.now();
  const startedAt = performance.now();
  let expectedAugments = Math.max(32, Math.round(gridNodes * 0.35));
  const tickMask = 0x3ff;

  function maybeProgress() {
    const now = performance.now();
    if (now - lastProgressAt >= progressEveryMs) {
      lastProgressAt = now;
      if (augments >= expectedAugments * 0.9) {
        expectedAugments = Math.max(expectedAugments + 1, Math.round(expectedAugments * 1.35));
      }
      const elapsedMs = now - startedAt;
      const progress = Math.min(0.98, augments / Math.max(1, expectedAugments));
      const remainingMs = augments > 0
        ? Math.max(0, Math.round((elapsedMs / augments) * Math.max(0, expectedAugments - augments)))
        : null;
      onProgress('seam-solve-maxflow', 55 + progress * 40, `Maxflow ${Math.round(progress * 100)}% (${augments}/${expectedAugments} est augments)`, {
        augments,
        expectedAugments,
        remainingMs,
        solverMs: Math.round(elapsedMs),
      });
    }
  }

  while (true) {
    parent.fill(-1);
    parentEdge.fill(-1);
    parent[S] = S;

    const queue = new Int32Array(totalNodes);
    let qi = 0;
    let qj = 0;
    queue[qj++] = S;

    while (qi < qj && parent[T] === -1) {
      const u = queue[qi++];
      const edges = adj[u];
      for (let k = 0; k < edges.length; k++) {
        const e = edges[k];
        if (parent[e.to] !== -1) continue;
        if (e.cap - e.flow <= 1e-9) continue;
        parent[e.to] = u;
        parentEdge[e.to] = k;
        queue[qj++] = e.to;
        if (e.to === T) break;
      }
    }

    if (parent[T] === -1) break;

    let bottleneck = Infinity;
    let v = T;
    while (v !== S) {
      const u = parent[v];
      const e = adj[u][parentEdge[v]];
      bottleneck = Math.min(bottleneck, e.cap - e.flow);
      v = u;
    }

    v = T;
    while (v !== S) {
      const u = parent[v];
      const ei = parentEdge[v];
      const e = adj[u][ei];
      e.flow += bottleneck;
      adj[v][e.rev].flow -= bottleneck;
      v = u;
    }

    totalFlow += bottleneck;
    augments++;
    if ((augments & tickMask) === 0) maybeProgress();
  }

  onProgress('seam-solve-maxflow', 95, `Maxflow done (${augments} augments)`, {
    augments,
    expectedAugments,
    remainingMs: 0,
    solverMs: Math.round(performance.now() - startedAt),
  });

  return { totalFlow, augments };
}
