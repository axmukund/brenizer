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
//  - {type:'init', baseUrl, maxflowPath}
//  - {type:'solve', jobId, gridW, gridH, dataCostsBuffer, edgeWeightsBuffer, hardConstraintsBuffer, params}
// Messages out:
//  - {type:'progress', stage, percent, info}
//  - {type:'result', jobId, labelsBuffer}
//  - {type:'error', jobId?, message}

let ready = false;

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

      const labels = solveMinCut(gridW, gridH, dataCosts, edgeWeights, hard, params || {});

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
function solveMinCut(gridW, gridH, dataCosts, edgeWeights, hard, params) {
  const n = gridW * gridH;
  const S = n;     // virtual source node
  const T = n + 1; // virtual sink node
  const totalNodes = n + 2;

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
  }

  // Add n-links (smoothness between adjacent blocks)
  const nHorizontal = (gridW - 1) * gridH;

  for (let y = 0; y < gridH; y++) {
    for (let x = 0; x < gridW - 1; x++) {
      const a = y * gridW + x;
      const b = y * gridW + x + 1;
      const eIdx = y * (gridW - 1) + x;
      const w = edgeWeights ? edgeWeights[eIdx] : 1.0;
      addEdgeBi(a, b, w);
    }
  }

  for (let y = 0; y < gridH - 1; y++) {
    for (let x = 0; x < gridW; x++) {
      const a = y * gridW + x;
      const b = (y + 1) * gridW + x;
      const eIdx = nHorizontal + y * gridW + x;
      const w = edgeWeights ? edgeWeights[eIdx] : 1.0;
      addEdgeBi(a, b, w);
    }
  }

  // Edmonds-Karp maxflow
  maxflow(adj, S, T, totalNodes);

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

  return labels;
}

/**
 * Edmonds-Karp maxflow algorithm (BFS for shortest augmenting path).
 */
function maxflow(adj, S, T, totalNodes) {
  let totalFlow = 0;
  const parent = new Int32Array(totalNodes);
  const parentEdge = new Int32Array(totalNodes);

  while (true) {
    parent.fill(-1);
    parentEdge.fill(-1);
    parent[S] = S;
    const queue = [S];
    let qi = 0;

    while (qi < queue.length && parent[T] === -1) {
      const u = queue[qi++];
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
  }

  return totalFlow;
}
