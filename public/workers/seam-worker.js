/**
 * seam-worker.js — Graph-cut seam solver running in a Web Worker.
 *
 * Solves binary min-cut on a coarse block grid using the Edmonds-Karp
 * maximum flow algorithm (BFS for shortest augmenting paths). Pure JS
 * implementation with no WASM dependency — works everywhere WebGL2 does.
 *
 * The grid graph represents a downsampled version of the overlap region
 * between the composite and a new image. Each node is a block of pixels;
 * the solver assigns each node to either "keep composite" (label 0) or
 * "take new image" (label 1) to minimise the total cut cost.
 *
 * Cost model (from composition.ts):
 *  - Data costs: distance-from-boundary + colour difference + saliency penalty
 *  - Edge weights: gradient-domain agreement + Brenizer blur discount
 *  - Hard constraints: blocks with only one source are clamped
 *  - Face penalty: keeps seams away from detected faces
 *
 * Messages in:
 *  - {type:'init', baseUrl, maxflowPath}
 *  - {type:'solve', jobId, gridW, gridH, dataCostsBuffer, edgeWeightsBuffer, hardConstraintsBuffer, params}
 *
 * Messages out:
 *  - {type:'progress', stage, percent, info}
 *  - {type:'result', jobId, labelsBuffer}
 *  - {type:'error', jobId?, message}
 */

let ready = false;
let rollingAugmentsPerNode = 0.35;

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      // No WASM needed — pure JS solver
      ready = true;
      postMessage({type:'progress', stage:'seam-init', percent:100, info:'maxflow ready (JS)'});
      return;
    }

    if (msg.type === 'solve') {
      const { jobId, gridW, gridH, dataCostsBuffer, edgeWeightsBuffer, hardConstraintsBuffer, params } = msg;
      const solveParams = params || {};
      const progressEveryMs = Math.max(250, Number(solveParams.progressEveryMs) || 1200);
      const startedAt = performance.now();
      const emitProgress = (stage, percent, info, extra = {}) => {
        postMessage({
          type: 'progress',
          stage,
          percent,
          info,
          jobId,
          elapsedMs: Math.round(performance.now() - startedAt),
          ...extra,
        });
      };
      emitProgress('seam-solve-start', 0, `Solving seam ${gridW}×${gridH}`);

      // dataCosts: Float32Array, 2 floats per node [cost_source, cost_sink]
      //   cost_source = cost if labelled 0 (keep composite) — i.e., penalty for NOT being source
      //   cost_sink   = cost if labelled 1 (take new)       — i.e., penalty for NOT being sink
      const dataCosts = new Float32Array(dataCostsBuffer);
      const nNodes = gridW * gridH;

      // edgeWeights: Float32Array, per edge (horizontal edges then vertical edges)
      //   horizontal: (gridW-1)*gridH edges, vertical: gridW*(gridH-1) edges
      const edgeWeights = edgeWeightsBuffer ? new Float32Array(edgeWeightsBuffer) : null;

      // hard constraints: Uint8Array per node: 0=free, 1=force source (composite), 2=force sink (new)
      const hard = hardConstraintsBuffer ? new Uint8Array(hardConstraintsBuffer) : null;

      const labels = solveMinCut(
        gridW,
        gridH,
        dataCosts,
        edgeWeights,
        hard,
        solveParams,
        (stage, percent, info, extra = {}) => emitProgress(stage, percent, info, extra),
        progressEveryMs,
      );

      emitProgress('seam-solve-done', 100, 'Seam solve complete');
      postMessage({type:'result', jobId, labelsBuffer: labels.buffer}, [labels.buffer]);
      return;
    }

    postMessage({type:'error', message:'unknown message type: ' + msg.type});
  } catch (err) {
    postMessage({type:'error', jobId: msg.jobId, message: err.message || err.toString()});
  }
}, false);

/**
 * Solve binary min-cut on a 2D grid graph using Edmonds-Karp maxflow.
 * Returns Uint8Array of labels: 0 = source (keep composite), 1 = sink (take new image).
 */
function solveMinCut(gridW, gridH, dataCosts, edgeWeights, hard, _params, onProgress, progressEveryMs) {
  const n = gridW * gridH;
  const S = n;     // virtual source node
  const T = n + 1; // virtual sink node
  const totalNodes = n + 2;
  const totalEdgesEstimate = (gridW - 1) * gridH + gridW * (gridH - 1);
  let lastBuildProgressAt = performance.now();
  const BUILD_DATA_START_PERCENT = 5;
  const BUILD_DATA_SPAN_PERCENT = 25;
  const BUILD_EDGES_START_PERCENT = BUILD_DATA_START_PERCENT + BUILD_DATA_SPAN_PERCENT;
  const BUILD_EDGES_SPAN_PERCENT = 25;
  const MAXFLOW_START_PERCENT = BUILD_EDGES_START_PERCENT + BUILD_EDGES_SPAN_PERCENT;
  const MAXFLOW_SPAN_PERCENT = 40;
  const LABELS_PERCENT = 98;

  function maybeBuildProgress(stage, processed, total, startPercent, spanPercent) {
    if (!onProgress) return;
    const now = performance.now();
    if (now - lastBuildProgressAt >= progressEveryMs) {
      lastBuildProgressAt = now;
      const phasePct = total > 0 ? Math.min(1, processed / total) : 0;
      const overallPercent = startPercent + spanPercent * phasePct;
      onProgress(stage, overallPercent, `Graph build ${Math.round(phasePct * 100)}%`, { augments: 0 });
    }
  }

  // Build adjacency list graph with capacities
  // Each edge is stored in both directions (forward + reverse)

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

  // Add s-t edges (data terms)
  for (let i = 0; i < n; i++) {
    const costSource = dataCosts[i * 2];     // penalty for label=0 (force toward source)
    const costSink = dataCosts[i * 2 + 1];   // penalty for NOT being sink

    let capS = costSink;  // capacity from S→node
    let capT = costSource; // capacity from node→T

    // Hard constraints
    if (hard) {
      if (hard[i] === 1) { capS = 1e9; capT = 0; }       // force source (label 0 = composite)
      else if (hard[i] === 2) { capS = 0; capT = 1e9; }   // force sink (label 1 = new)
    }

    if (capS > 0) addEdge(S, i, capS);
    if (capT > 0) addEdge(i, T, capT);
    if ((i & 0x7ff) === 0) {
      maybeBuildProgress('seam-solve-build-data', i, n, BUILD_DATA_START_PERCENT, BUILD_DATA_SPAN_PERCENT);
    }
  }
  maybeBuildProgress('seam-solve-build-data', n, n, BUILD_DATA_START_PERCENT, BUILD_DATA_SPAN_PERCENT);

  // Add n-links (smoothness between adjacent blocks)
  const nHorizontal = (gridW - 1) * gridH;
  let processedEdges = 0;

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const a = y * gridW + x;
      const b = y * gridW + x + 1;
      const eIdx = y * (gridW - 1) + x;
      const w = edgeWeights ? edgeWeights[eIdx] : 1.0;
      addEdgeBi(a, b, w);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) {
        maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, BUILD_EDGES_START_PERCENT, BUILD_EDGES_SPAN_PERCENT);
      }
    }
  }

  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW; x++) {
      const a = y * gridW + x;
      const b = (y + 1) * gridW + x;
      const eIdx = nHorizontal + y * gridW + x;
      const w = edgeWeights ? edgeWeights[eIdx] : 1.0;
      addEdgeBi(a, b, w);
      processedEdges++;
      if ((processedEdges & 0x7ff) === 0) {
        maybeBuildProgress('seam-solve-build-edges', processedEdges, totalEdgesEstimate, BUILD_EDGES_START_PERCENT, BUILD_EDGES_SPAN_PERCENT);
      }
    }
  }
  maybeBuildProgress('seam-solve-build-edges', totalEdgesEstimate, totalEdgesEstimate, BUILD_EDGES_START_PERCENT, BUILD_EDGES_SPAN_PERCENT);

  if (onProgress) onProgress('seam-solve-graph', MAXFLOW_START_PERCENT, `Graph ready (${n} nodes)`);

  // Edmonds-Karp maxflow
  const flowStats = maxflow(
    adj,
    S,
    T,
    totalNodes,
    n,
    onProgress
      ? (stage, phasePct, info, extra = {}) => {
          const overallPercent = MAXFLOW_START_PERCENT + MAXFLOW_SPAN_PERCENT * phasePct;
          onProgress(stage, overallPercent, info, extra);
        }
      : null,
    progressEveryMs,
  );

  // Find min-cut: BFS from S on residual graph
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

  // Labels: nodes reachable from S = source side (label 0), else sink side (label 1)
  const labels = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    labels[i] = visited[i] ? 0 : 1;
  }

  if (onProgress) onProgress('seam-solve-labels', LABELS_PERCENT, `Labels built (${flowStats.augments} augments)`, {
    augments: flowStats.augments,
    remainingMs: 0,
  });

  return labels;
}

/**
 * Edmonds-Karp maxflow algorithm (BFS for shortest augmenting path).
 */
function maxflow(adj, S, T, totalNodes, gridNodes, onProgress, progressEveryMs) {
  let totalFlow = 0;
  const parent = new Int32Array(totalNodes);
  const parentEdge = new Int32Array(totalNodes);
  let augments = 0;
  let visitedNodes = 0;
  let lastProgressAt = performance.now();
  const startedAt = performance.now();
  let expectedAugments = Math.max(32, Math.round(gridNodes * rollingAugmentsPerNode));
  const tickMask = 0x3ff;

  function maybeProgress(stage) {
    if (!onProgress) return;
    const now = performance.now();
    if (now - lastProgressAt >= progressEveryMs) {
      lastProgressAt = now;
      while (augments >= expectedAugments * 0.9) {
        expectedAugments = Math.max(expectedAugments + 1, Math.round(expectedAugments * 1.35));
      }
      const phasePct = expectedAugments > 0 ? Math.min(0.995, augments / expectedAugments) : 0;
      const elapsedMs = now - startedAt;
      const augmentsPerSec = elapsedMs > 0 ? augments / (elapsedMs / 1000) : 0;
      const remainingAugments = Math.max(0, expectedAugments - augments);
      const remainingMs = augmentsPerSec > 0.05 ? Math.round((remainingAugments / augmentsPerSec) * 1000) : undefined;
      onProgress(
        stage,
        phasePct,
        `Maxflow ${Math.round(phasePct * 100)}% (${augments}/${expectedAugments} est augments)`,
        { augments, remainingMs, expectedAugments },
      );
    }
  }

  while (true) {
    parent.fill(-1);
    parentEdge.fill(-1);
    parent[S] = S;
    const queue = [S];
    let qi = 0;

    while (qi < queue.length && parent[T] === -1) {
      const u = queue[qi++];
      visitedNodes++;
      if ((visitedNodes & tickMask) === 0) maybeProgress('seam-solve-search');
      const edges = adj[u];
      for (let k = 0; k < edges.length; k++) {
        const e = edges[k];
        if (parent[e.to] === -1 && e.cap - e.flow > 1e-9) {
          parent[e.to] = u;
          parentEdge[e.to] = k;
          if (e.to === T) break;
          queue.push(e.to);
        }
      }
    }

    if (parent[T] === -1) break;

    // Find bottleneck
    let bottleneck = Infinity;
    let v = T;
    while (v !== S) {
      const u = parent[v];
      const e = adj[u][parentEdge[v]];
      bottleneck = Math.min(bottleneck, e.cap - e.flow);
      v = u;
    }

    // Update flows
    v = T;
    while (v !== S) {
      const u = parent[v];
      const e = adj[u][parentEdge[v]];
      e.flow += bottleneck;
      adj[v][e.rev].flow -= bottleneck;
      v = u;
    }

    totalFlow += bottleneck;
    augments++;
    if ((augments & 0x1f) === 0) maybeProgress('seam-solve-augment');
  }

  const augmentsPerNode = augments / Math.max(1, gridNodes);
  const clampedAugmentsPerNode = Math.min(8, Math.max(0.02, augmentsPerNode));
  rollingAugmentsPerNode = rollingAugmentsPerNode * 0.85 + clampedAugmentsPerNode * 0.15;
  if (onProgress) {
    onProgress('seam-solve-maxflow', 1, `Maxflow done (${augments} augments)`, {
      augments,
      remainingMs: 0,
      expectedAugments: Math.max(expectedAugments, augments),
    });
  }
  return { totalFlow, augments };
}
