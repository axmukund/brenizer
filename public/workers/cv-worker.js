// cv-worker.js - classic worker using OpenCV.js
// Loads opencv.js via importScripts and exposes message handlers per spec.
// Messages accepted:
//  - {type:'init', baseUrl, opencvPath}
//  - {type:'addImage', imageId, grayBuffer, width, height, rgbSmallBuffer?, depth?}
//  - {type:'computeFeatures', orbParams}
//  - {type:'matchGraph', windowW, ratio, ransacThreshPx, minInliers, matchAllPairs}
//  - {type:'buildGraph'}
//  - {type:'refine', maxIters, huberDeltaPx, lambdaInit}
//  - {type:'computeExposure'}
//  - {type:'buildMST'}
//  - {type:'computeLocalMesh', imageId, parentId, meshGrid, sigma, depthSigma, minSupport}

// Messages posted back:
//  - {type:'progress', stage, percent, info}
//  - {type:'features', imageId, keypointsBuffer, descriptorsBuffer, descCols}
//  - {type:'edges', edges: [...]}
//  - {type:'transforms', refId, transforms: [...]}
//  - {type:'exposure', gains: [...]}
//  - {type:'mst', refId, order, parent}
//  - {type:'mesh', imageId, verticesBuffer, uvsBuffer, indicesBuffer, bounds}
//  - {type:'error', message}

let cvReady = false;
let images = {}; // imageId -> {width, height, gray:Uint8ClampedArray, depth?:Uint16Array, keypoints:Float32Array, descriptors:Uint8Array, descCols:number}
let edges = [];  // [{i, j, H:Float64Array, inliers:[{xi,yi,xj,yj}], rms, inlierCount}]
let transforms = {}; // imageId -> Float64Array(9) col-major 3x3
let refId = null;
let mstOrder = [];
let mstParent = {};

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const opencvPath = msg.opencvPath || 'opencv/opencv.js';
      const fullPath = new URL(opencvPath, msg.baseUrl).toString();
      importScripts(fullPath);

      // cv is a Module factory — wait for WASM to be ready
      if (typeof cv !== 'undefined' && typeof cv.then === 'function') {
        await new Promise((resolve, reject) => {
          const timeout = setTimeout(() => reject(new Error('OpenCV WASM init timeout')), 30000);
          cv.then(function(mod) {
            clearTimeout(timeout);
            // In some builds cv is replaced by the module
            if (typeof cv.Mat === 'undefined' && mod && mod.Mat) {
              self.cv = mod;
            }
            resolve();
          });
        });
      } else {
        // Fallback: poll for cv.Mat
        const start = Date.now();
        while (typeof cv === 'undefined' || typeof cv.Mat === 'undefined') {
          await new Promise(r => setTimeout(r, 50));
          if (Date.now() - start > 30000) throw new Error('OpenCV failed to load in time');
        }
      }

      cvReady = true;
      postMessage({type:'progress', stage:'cv-init', percent:100, info:'OpenCV loaded'});
      return;
    }

    if (!cvReady) throw new Error('cv-worker not initialized');

    if (msg.type === 'addImage') {
      const {imageId, grayBuffer, width, height, depth} = msg;
      const gray = new Uint8ClampedArray(grayBuffer);
      images[imageId] = {
        width, height, gray,
        depth: depth ? new Uint16Array(depth) : null,
        keypoints: null,
        descriptors: null,
        descCols: 0
      };
      postMessage({type:'progress', stage:'addImage', percent:100, info:`added ${imageId}`});
      return;
    }

    if (msg.type === 'computeFeatures') {
      const nFeatures = (msg.orbParams && msg.orbParams.nFeatures) ? msg.orbParams.nFeatures : 2000;
      const ids = Object.keys(images);
      let done = 0;

      for (const id of ids) {
        const img = images[id];
        let mat = null;
        let orb = null;
        let keypoints = null;
        let descriptors = null;
        let mask = null;

        try {
          mat = cv.matFromArray(img.height, img.width, cv.CV_8UC1, img.gray);
          orb = new cv.ORB(nFeatures);
          keypoints = new cv.KeyPointVector();
          descriptors = new cv.Mat();
          mask = new cv.Mat();

          orb.detectAndCompute(mat, mask, keypoints, descriptors);

          // Serialize keypoints to Float32Array [x0,y0,x1,y1,...]
          const numKp = keypoints.size();
          const kps = new Float32Array(numKp * 2);
          for (let i = 0; i < numKp; i++) {
            const kp = keypoints.get(i);
            kps[i * 2] = kp.pt.x;
            kps[i * 2 + 1] = kp.pt.y;
          }

          // Serialize descriptors to Uint8Array
          const descCols = descriptors.cols;
          const descRows = descriptors.rows;
          const descBuf = new Uint8Array(descRows * descCols);
          if (descRows > 0 && descCols > 0) {
            descBuf.set(descriptors.data.slice(0, descRows * descCols));
          }

          img.keypoints = kps;
          img.descriptors = descBuf;
          img.descCols = descCols;

          postMessage({
            type:'features',
            imageId: id,
            keypointsBuffer: kps.buffer,
            descriptorsBuffer: descBuf.buffer,
            descCols
          });

        } finally {
          if (mat) mat.delete();
          if (orb) orb.delete();
          if (keypoints) keypoints.delete();
          if (descriptors) descriptors.delete();
          if (mask) mask.delete();
        }

        done++;
        postMessage({type:'progress', stage:'features', percent: Math.round(100 * done / ids.length), info: `${done}/${ids.length}`});
      }
      return;
    }

    if (msg.type === 'matchGraph') {
      const {windowW, ratio, ransacThreshPx, minInliers, matchAllPairs} = msg;
      const ids = Object.keys(images);
      edges = [];

      // Build candidate pairs
      const pairs = [];
      for (let a = 0; a < ids.length; a++) {
        for (let b = a + 1; b < ids.length; b++) {
          if (matchAllPairs || (b - a) <= windowW) {
            pairs.push([ids[a], ids[b]]);
          }
        }
      }

      let done = 0;
      for (const [idI, idJ] of pairs) {
        const imgI = images[idI];
        const imgJ = images[idJ];

        if (!imgI.descriptors || !imgJ.descriptors ||
            imgI.descriptors.length === 0 || imgJ.descriptors.length === 0) {
          done++;
          continue;
        }

        const descColsI = imgI.descCols || 32;
        const descColsJ = imgJ.descCols || 32;
        const numI = imgI.descriptors.length / descColsI;
        const numJ = imgJ.descriptors.length / descColsJ;

        if (numI < 4 || numJ < 4) { done++; continue; }

        let matI = null, matJ = null;
        let bf = null, matches = null;

        try {
          matI = cv.matFromArray(numI, descColsI, cv.CV_8UC1, imgI.descriptors);
          matJ = cv.matFromArray(numJ, descColsJ, cv.CV_8UC1, imgJ.descriptors);

          bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
          matches = new cv.DMatchVectorVector();
          bf.knnMatch(matI, matJ, matches, 2);

          // Ratio test
          const goodPtsI = [];
          const goodPtsJ = [];
          for (let m = 0; m < matches.size(); m++) {
            const match = matches.get(m);
            if (match.size() >= 2) {
              const m1 = match.get(0);
              const m2 = match.get(1);
              if (m1.distance < ratio * m2.distance) {
                const qi = m1.queryIdx;
                const ti = m1.trainIdx;
                if (qi < numI && ti < numJ) {
                  goodPtsI.push(imgI.keypoints[qi * 2], imgI.keypoints[qi * 2 + 1]);
                  goodPtsJ.push(imgJ.keypoints[ti * 2], imgJ.keypoints[ti * 2 + 1]);
                }
              }
            }
          }

          if (goodPtsI.length / 2 < minInliers) { done++; continue; }

          // RANSAC homography
          const srcPts = cv.matFromArray(goodPtsI.length / 2, 1, cv.CV_32FC2, new Float32Array(goodPtsI));
          const dstPts = cv.matFromArray(goodPtsJ.length / 2, 1, cv.CV_32FC2, new Float32Array(goodPtsJ));
          const inlierMask = new cv.Mat();
          const H = cv.findHomography(srcPts, dstPts, cv.RANSAC, ransacThreshPx, inlierMask);

          if (H.rows === 3 && H.cols === 3) {
            // Count inliers
            let inlierCount = 0;
            const inliers = [];
            for (let k = 0; k < inlierMask.rows; k++) {
              if (inlierMask.data[k]) {
                inlierCount++;
                inliers.push({
                  xi: goodPtsI[k * 2], yi: goodPtsI[k * 2 + 1],
                  xj: goodPtsJ[k * 2], yj: goodPtsJ[k * 2 + 1]
                });
              }
            }

            if (inlierCount >= minInliers) {
              // Compute RMS
              let rmsSum = 0;
              for (const inl of inliers) {
                const hd = H.data64F;
                const denom = hd[6] * inl.xi + hd[7] * inl.yi + hd[8];
                const px = (hd[0] * inl.xi + hd[1] * inl.yi + hd[2]) / denom;
                const py = (hd[3] * inl.xi + hd[4] * inl.yi + hd[5]) / denom;
                rmsSum += (px - inl.xj) ** 2 + (py - inl.yj) ** 2;
              }
              const rms = Math.sqrt(rmsSum / inlierCount);

              const HBuf = new Float64Array(9);
              HBuf.set(H.data64F.slice(0, 9));

              const inliersBuf = new Float32Array(inlierCount * 4);
              for (let k = 0; k < inlierCount; k++) {
                inliersBuf[k * 4] = inliers[k].xi;
                inliersBuf[k * 4 + 1] = inliers[k].yi;
                inliersBuf[k * 4 + 2] = inliers[k].xj;
                inliersBuf[k * 4 + 3] = inliers[k].yj;
              }

              edges.push({
                i: idI, j: idJ,
                H: HBuf,
                inliers: inliers,
                inliersBuf: inliersBuf,
                rms,
                inlierCount
              });
            }
          }

          srcPts.delete(); dstPts.delete(); inlierMask.delete(); H.delete();
        } finally {
          if (matI) matI.delete();
          if (matJ) matJ.delete();
          if (bf) bf.delete();
          if (matches) matches.delete();
        }
        done++;
        postMessage({type:'progress', stage:'matching', percent: Math.round(100 * done / pairs.length), info: `${done}/${pairs.length}`});
      }

      // If graph is disconnected and we didn't match all pairs, retry with all pairs
      if (!matchAllPairs && edges.length > 0) {
        const connected = checkConnectivity(ids, edges);
        if (!connected) {
          postMessage({type:'progress', stage:'matching', percent: 100, info: 'Graph disconnected, trying all pairs...'});
          // Could expand window or match all — for now just warn
        }
      }

      // Send edges
      const edgeMessages = edges.map(e => ({
        i: e.i,
        j: e.j,
        HBuffer: e.H.buffer,
        inliersBuffer: e.inliersBuf.buffer,
        rms: e.rms,
        inlierCount: e.inlierCount
      }));
      postMessage({type:'edges', edges: edgeMessages});
      return;
    }

    if (msg.type === 'buildGraph') {
      // Already have edges from matchGraph; just acknowledge
      postMessage({type:'progress', stage:'buildGraph', percent:100, info:'graph built'});
      return;
    }

    if (msg.type === 'buildMST') {
      const ids = Object.keys(images);
      if (edges.length === 0 || ids.length < 2) {
        postMessage({type:'mst', refId: ids[0] || null, order: ids, parent: {}});
        return;
      }

      // Select reference: node with max sum of edge weights
      const weightSum = {};
      for (const id of ids) weightSum[id] = 0;
      for (const e of edges) {
        const w = e.inlierCount / (e.rms + 0.1);
        weightSum[e.i] = (weightSum[e.i] || 0) + w;
        weightSum[e.j] = (weightSum[e.j] || 0) + w;
      }
      refId = ids.reduce((a, b) => (weightSum[a] || 0) >= (weightSum[b] || 0) ? a : b);

      // Maximum spanning tree (Kruskal-like, descending edge weight)
      const sortedEdges = [...edges].sort((a, b) => {
        const wa = a.inlierCount / (a.rms + 0.1);
        const wb = b.inlierCount / (b.rms + 0.1);
        return wb - wa;
      });

      // Union-Find
      const parent = {};
      const rank = {};
      for (const id of ids) { parent[id] = id; rank[id] = 0; }
      function find(x) { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; }
      function union(a, b) {
        a = find(a); b = find(b);
        if (a === b) return false;
        if (rank[a] < rank[b]) [a, b] = [b, a];
        parent[b] = a;
        if (rank[a] === rank[b]) rank[a]++;
        return true;
      }

      const mstEdges = [];
      for (const e of sortedEdges) {
        if (union(e.i, e.j)) {
          mstEdges.push(e);
          if (mstEdges.length === ids.length - 1) break;
        }
      }

      // BFS from refId to get order and parent map
      const adj = {};
      for (const id of ids) adj[id] = [];
      for (const e of mstEdges) {
        adj[e.i].push({to: e.j, edge: e});
        adj[e.j].push({to: e.i, edge: e});
      }

      mstOrder = [];
      mstParent = {};
      const visited = new Set();
      const queue = [refId];
      visited.add(refId);
      mstParent[refId] = null;

      while (queue.length > 0) {
        const node = queue.shift();
        mstOrder.push(node);
        for (const {to} of adj[node]) {
          if (!visited.has(to)) {
            visited.add(to);
            mstParent[to] = node;
            queue.push(to);
          }
        }
      }

      // Add any disconnected nodes
      for (const id of ids) {
        if (!visited.has(id)) {
          mstOrder.push(id);
          mstParent[id] = null;
        }
      }

      // Compute initial global transforms by propagating H along MST
      transforms = {};
      transforms[refId] = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]); // identity

      // For each node in BFS order (after ref), compute T = T_parent * H or T_parent * H_inv
      for (let k = 1; k < mstOrder.length; k++) {
        const node = mstOrder[k];
        const par = mstParent[node];
        if (!par || !transforms[par]) {
          transforms[node] = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
          continue;
        }

        // Find edge between par and node
        const edge = edges.find(e =>
          (e.i === par && e.j === node) || (e.i === node && e.j === par)
        );

        if (!edge) {
          transforms[node] = new Float64Array(transforms[par]);
          continue;
        }

        // H maps i->j. If edge.i === node, edge.j === par: H maps node->par
        // We need T_node such that when we project node's points to global:
        //   global = T_node * pt_node
        // And T_par * pt_par = global for parent points
        // H maps i->j, so j = H * i
        // If edge.i === node, edge.j === par:
        //   par_pt = H * node_pt => T_par * par_pt ≈ T_node * node_pt (not quite right)
        // Actually: T_node should place node's points in global coords
        // T_par places par's points in global coords
        // H maps edge.i -> edge.j
        // If edge.i === node, edge.j === par: par_pt = H * node_pt
        //   =>  T_global_node = T_par * H (project node through H to par, then par to global)
        // If edge.i === par, edge.j === node: node_pt = H * par_pt
        //   =>  T_global_node needs H_inv: T_node = T_par * H_inv

        let T_node;
        if (edge.i === node && edge.j === par) {
          // H maps node -> par, so T_node = T_par * H
          T_node = mulMat3(transforms[par], edge.H);
        } else {
          // H maps par -> node, so T_node = T_par * H_inv
          const Hinv = invertH(edge.H);
          if (Hinv) {
            T_node = mulMat3(transforms[par], Hinv);
          } else {
            T_node = new Float64Array(transforms[par]);
          }
        }

        transforms[node] = T_node;
      }

      postMessage({type:'mst', refId, order: mstOrder, parent: mstParent});

      // Also send transforms
      const tList = Object.entries(transforms).map(([id, T]) => ({
        imageId: id,
        TBuffer: T.buffer
      }));
      postMessage({type:'transforms', refId, transforms: tList});
      return;
    }

    if (msg.type === 'refine') {
      // TODO: implement LM refinement loop over homography params
      // For now, send current transforms as-is
      const tList = Object.entries(transforms).map(([id, T]) => ({
        imageId: id,
        TBuffer: new Float64Array(T).buffer
      }));
      postMessage({type:'transforms', refId, transforms: tList});
      postMessage({type:'progress', stage:'refine', percent:100, info:'refinement placeholder completed'});
      return;
    }

    if (msg.type === 'computeExposure') {
      const ids = Object.keys(images);
      const n = ids.length;

      if (n < 2 || edges.length === 0) {
        const gains = ids.map(id => ({ imageId: id, gain: 1.0 }));
        postMessage({ type: 'exposure', gains });
        return;
      }

      const idToIdx = {};
      ids.forEach((id, i) => idToIdx[id] = i);

      // For each edge, compute mean log luminance ratio from inlier correspondences
      // r_ij ≈ log(mean_lum_i) - log(mean_lum_j)  at matching keypoints
      const edgeRatios = [];

      for (const e of edges) {
        const imgI = images[e.i];
        const imgJ = images[e.j];
        if (!imgI || !imgJ || !imgI.gray || !imgJ.gray) continue;

        // Sample luminance at inlier keypoint positions
        let sumLogI = 0, sumLogJ = 0, count = 0;
        for (const inl of e.inliers) {
          let xi, yi, xj, yj;
          xi = inl.xi; yi = inl.yi;
          xj = inl.xj; yj = inl.yj;

          const lumI = sampleGray(imgI.gray, imgI.width, imgI.height, xi, yi);
          const lumJ = sampleGray(imgJ.gray, imgJ.width, imgJ.height, xj, yj);

          if (lumI > 5 && lumJ > 5) { // avoid very dark pixels
            sumLogI += Math.log(lumI);
            sumLogJ += Math.log(lumJ);
            count++;
          }
        }

        if (count >= 5) {
          // r_ij = mean(log(lum_j)) - mean(log(lum_i))
          // This means: to make them equal, g_i should compensate for this diff
          const rij = (sumLogJ / count) - (sumLogI / count);
          edgeRatios.push({
            i: idToIdx[e.i],
            j: idToIdx[e.j],
            ratio: rij
          });
        }
      }

      if (edgeRatios.length === 0) {
        const gains = ids.map(id => ({ imageId: id, gain: 1.0 }));
        postMessage({ type: 'exposure', gains });
        return;
      }

      // Solve least squares: for each edge, log(g_j) - log(g_i) ≈ r_ij
      // Fix reference image gain = 1 (log(g_ref) = 0)
      const refIdx = refId ? idToIdx[refId] : 0;

      // Build normal equations: A' * A * x = A' * b
      // where each edge gives one equation: x_j - x_i = r_ij (x = log gains)
      // With constraint x_ref = 0
      const AtA = new Float64Array(n * n);
      const Atb = new Float64Array(n);

      for (const er of edgeRatios) {
        // Equation: x_j - x_i = r_ij
        // => -x_i + x_j = r_ij
        AtA[er.i * n + er.i] += 1;
        AtA[er.j * n + er.j] += 1;
        AtA[er.i * n + er.j] -= 1;
        AtA[er.j * n + er.i] -= 1;
        Atb[er.i] -= er.ratio; // -1 * r_ij
        Atb[er.j] += er.ratio; // +1 * r_ij
      }

      // Fix reference: strong prior x_ref = 0
      const lambda = 1000;
      AtA[refIdx * n + refIdx] += lambda;
      // Atb[refIdx] += 0 (already zero target)

      // Add regularization to pull all gains toward 1
      const regWeight = 0.01;
      for (let i = 0; i < n; i++) {
        AtA[i * n + i] += regWeight;
      }

      // Solve using Cholesky-like approach (simple Gaussian elimination)
      const logGains = solveLinearN(AtA, Atb, n);

      const gains = ids.map((id, i) => ({
        imageId: id,
        gain: logGains ? Math.exp(logGains[i]) : 1.0
      }));

      postMessage({ type: 'exposure', gains });
      return;
    }

    if (msg.type === 'computeLocalMesh') {
      const { imageId, parentId, meshGrid, sigma, depthSigma, minSupport } = msg;
      const G = meshGrid || 4;
      const img = images[imageId];
      if (!img) {
        postMessage({type:'error', message:`Image ${imageId} not found`});
        return;
      }

      // Find edge between imageId and parentId to get inlier correspondences
      const edge = edges.find(e =>
        (e.i === imageId && e.j === parentId) || (e.i === parentId && e.j === imageId)
      );

      // Global transforms for image and parent
      const Ti = transforms[imageId];
      const Tp = transforms[parentId];

      if (!edge || !Ti || !Tp) {
        // Fallback: return identity mesh warped by global transform
        const result = buildGlobalMesh(imageId, G, Ti, img);
        postMessage({type:'mesh', imageId, ...result});
        return;
      }

      // Build correspondences: x_src (in image i coords) → X_target (in global coords via parent)
      // Edge inliers are {xi, yi, xj, yj}
      // If edge.i === imageId: xi,yi are in imageId coords; xj,yj in parentId coords
      // If edge.i === parentId: swap
      const srcPts = []; // [x, y] in image i coords
      const dstPts = []; // [X, Y] in global coords

      for (const inl of edge.inliers) {
        let si_x, si_y, sp_x, sp_y;
        if (edge.i === imageId) {
          si_x = inl.xi; si_y = inl.yi;
          sp_x = inl.xj; sp_y = inl.yj;
        } else {
          si_x = inl.xj; si_y = inl.yj;
          sp_x = inl.xi; sp_y = inl.yi;
        }

        // Project parent point to global using Tp
        const denom = Tp[6] * sp_x + Tp[7] * sp_y + Tp[8];
        if (Math.abs(denom) < 1e-10) continue;
        const gx = (Tp[0] * sp_x + Tp[1] * sp_y + Tp[2]) / denom;
        const gy = (Tp[3] * sp_x + Tp[4] * sp_y + Tp[5]) / denom;

        srcPts.push([si_x, si_y]);
        dstPts.push([gx, gy]);
      }

      if (srcPts.length < 4) {
        // Not enough correspondences for DLT; use global transform
        const result = buildGlobalMesh(imageId, G, Ti, img);
        postMessage({type:'mesh', imageId, ...result});
        return;
      }

      // Optionally load depth data for depth weighting
      const depthData = img.depth; // Uint16Array or null
      const useDepth = depthData && depthSigma > 0;

      // Build mesh grid
      const cols = G + 1;
      const rows = G + 1;
      const w = img.width;
      const h = img.height;

      const vertices = new Float32Array(cols * rows * 2);
      const uvs = new Float32Array(cols * rows * 2);

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) {
          const u = c / G;
          const v = r / G;
          const vx = u * w; // vertex position in image coords
          const vy = v * h;
          const vi = (r * cols + c) * 2;

          uvs[vi] = u;
          uvs[vi + 1] = v;

          // Compute weighted DLT for this vertex
          const weights = new Float64Array(srcPts.length);
          let totalWeight = 0;
          const sigma2 = sigma * sigma;

          for (let k = 0; k < srcPts.length; k++) {
            const dx = srcPts[k][0] - vx;
            const dy = srcPts[k][1] - vy;
            let ws = Math.exp(-(dx * dx + dy * dy) / (2 * sigma2));

            if (useDepth) {
              // Sample depth at source point and vertex
              const dSrc = sampleDepth(depthData, w, h, srcPts[k][0], srcPts[k][1]);
              const dV = sampleDepth(depthData, w, h, vx, vy);
              const dd = dSrc - dV;
              const depthSigma2 = depthSigma * depthSigma;
              const wd = Math.exp(-(dd * dd) / (2 * depthSigma2));
              ws *= wd;
            }

            weights[k] = ws;
            totalWeight += ws;
          }

          // Check effective support
          const effectiveSupport = totalWeight;
          let gx, gy;

          if (effectiveSupport < minSupport || srcPts.length < 4) {
            // Fallback to global transform
            const denom = Ti[6] * vx + Ti[7] * vy + Ti[8];
            if (Math.abs(denom) < 1e-10) {
              gx = vx; gy = vy;
            } else {
              gx = (Ti[0] * vx + Ti[1] * vy + Ti[2]) / denom;
              gy = (Ti[3] * vx + Ti[4] * vy + Ti[5]) / denom;
            }
          } else {
            // Weighted DLT to find local homography H_v mapping src→dst
            const Hv = weightedDLT(srcPts, dstPts, weights);
            if (Hv) {
              const denom = Hv[6] * vx + Hv[7] * vy + Hv[8];
              if (Math.abs(denom) < 1e-10) {
                gx = vx; gy = vy;
              } else {
                gx = (Hv[0] * vx + Hv[1] * vy + Hv[2]) / denom;
                gy = (Hv[3] * vx + Hv[4] * vy + Hv[5]) / denom;
              }
            } else {
              // DLT failed, fallback
              const denom = Ti[6] * vx + Ti[7] * vy + Ti[8];
              if (Math.abs(denom) < 1e-10) {
                gx = vx; gy = vy;
              } else {
                gx = (Ti[0] * vx + Ti[1] * vy + Ti[2]) / denom;
                gy = (Ti[3] * vx + Ti[4] * vy + Ti[5]) / denom;
              }
            }
          }

          vertices[vi] = gx;
          vertices[vi + 1] = gy;
          minX = Math.min(minX, gx);
          minY = Math.min(minY, gy);
          maxX = Math.max(maxX, gx);
          maxY = Math.max(maxY, gy);
        }
      }

      // Triangulate
      const indices = [];
      for (let r = 0; r < G; r++) {
        for (let c = 0; c < G; c++) {
          const tl = r * cols + c;
          const tr = tl + 1;
          const bl = tl + cols;
          const br = bl + 1;
          indices.push(tl, bl, tr, tr, bl, br);
        }
      }

      const indicesArr = new Uint32Array(indices);

      postMessage({
        type:'mesh',
        imageId,
        verticesBuffer: vertices.buffer,
        uvsBuffer: uvs.buffer,
        indicesBuffer: indicesArr.buffer,
        bounds: { minX, minY, maxX, maxY }
      }, [vertices.buffer, uvs.buffer, indicesArr.buffer]);
      return;
    }

    postMessage({type:'error', message:`unknown message: ${JSON.stringify(msg).slice(0,200)}`});
  } catch (err) {
    postMessage({type:'error', message: err.message || err.toString()});
  }
}, false);

// ── Utility functions ──────────────────────────────────

function checkConnectivity(ids, edges) {
  if (ids.length <= 1) return true;
  const adj = {};
  for (const id of ids) adj[id] = [];
  for (const e of edges) {
    adj[e.i].push(e.j);
    adj[e.j].push(e.i);
  }
  const visited = new Set();
  const queue = [ids[0]];
  visited.add(ids[0]);
  while (queue.length > 0) {
    const n = queue.shift();
    for (const nb of adj[n]) {
      if (!visited.has(nb)) { visited.add(nb); queue.push(nb); }
    }
  }
  return visited.size === ids.length;
}

// Multiply two 3x3 matrices stored as row-major Float64Array(9)
function mulMat3(A, B) {
  const C = new Float64Array(9);
  for (let r = 0; r < 3; r++) {
    for (let c = 0; c < 3; c++) {
      C[r*3+c] = A[r*3+0]*B[0*3+c] + A[r*3+1]*B[1*3+c] + A[r*3+2]*B[2*3+c];
    }
  }
  return C;
}

// Invert a 3x3 homography matrix (row-major)
function invertH(H) {
  const [a,b,c,d,e,f,g,h,i] = H;
  const det = a*(e*i-f*h) - b*(d*i-f*g) + c*(d*h-e*g);
  if (Math.abs(det) < 1e-10) return null;
  const invDet = 1.0 / det;
  return new Float64Array([
    (e*i-f*h)*invDet, (c*h-b*i)*invDet, (b*f-c*e)*invDet,
    (f*g-d*i)*invDet, (a*i-c*g)*invDet, (c*d-a*f)*invDet,
    (d*h-e*g)*invDet, (b*g-a*h)*invDet, (a*e-b*d)*invDet
  ]);
}

// Build a mesh using the global transform (fallback for APAP)
function buildGlobalMesh(imageId, G, T, img) {
  const cols = G + 1;
  const rows = G + 1;
  const w = img.width;
  const h = img.height;

  const vertices = new Float32Array(cols * rows * 2);
  const uvs = new Float32Array(cols * rows * 2);
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const u = c / G;
      const v = r / G;
      const vx = u * w;
      const vy = v * h;
      const vi = (r * cols + c) * 2;
      uvs[vi] = u;
      uvs[vi + 1] = v;

      if (T) {
        const denom = T[6] * vx + T[7] * vy + T[8];
        if (Math.abs(denom) > 1e-10) {
          const gx = (T[0] * vx + T[1] * vy + T[2]) / denom;
          const gy = (T[3] * vx + T[4] * vy + T[5]) / denom;
          vertices[vi] = gx;
          vertices[vi + 1] = gy;
          minX = Math.min(minX, gx);
          minY = Math.min(minY, gy);
          maxX = Math.max(maxX, gx);
          maxY = Math.max(maxY, gy);
        } else {
          vertices[vi] = vx;
          vertices[vi + 1] = vy;
        }
      } else {
        vertices[vi] = vx;
        vertices[vi + 1] = vy;
        minX = Math.min(minX, vx);
        minY = Math.min(minY, vy);
        maxX = Math.max(maxX, vx);
        maxY = Math.max(maxY, vy);
      }
    }
  }

  const indices = [];
  for (let r = 0; r < G; r++) {
    for (let c = 0; c < G; c++) {
      const tl = r * cols + c;
      const tr = tl + 1;
      const bl = tl + cols;
      const br = bl + 1;
      indices.push(tl, bl, tr, tr, bl, br);
    }
  }

  return {
    verticesBuffer: vertices.buffer,
    uvsBuffer: uvs.buffer,
    indicesBuffer: new Uint32Array(indices).buffer,
    bounds: { minX, minY, maxX, maxY }
  };
}

// Sample depth value at (x, y) from Uint16Array depth map using nearest-neighbor
function sampleDepth(depthData, w, h, x, y) {
  const ix = Math.min(Math.max(Math.round(x), 0), w - 1);
  const iy = Math.min(Math.max(Math.round(y), 0), h - 1);
  return depthData[iy * w + ix] / 65535.0;
}

/**
 * Weighted DLT (Direct Linear Transform) to solve for a homography H
 * that maps srcPts[i] → dstPts[i] with weights[i].
 * Returns 9-element Float64Array (row-major 3x3) or null if degenerate.
 */
function weightedDLT(srcPts, dstPts, weights) {
  const n = srcPts.length;
  if (n < 4) return null;

  // Build weighted A matrix for Ah = 0
  // Each correspondence gives 2 rows in A (9 columns)
  // Row 1: [0, 0, 0, -w*x, -w*y, -w, w*y'*x, w*y'*y, w*y']
  // Row 2: [w*x, w*y, w, 0, 0, 0, -w*x'*x, -w*x'*y, -w*x']
  const rows = 2 * n;
  const cols = 9;

  // Use AtA (9x9) directly instead of forming the full A matrix
  const AtA = new Float64Array(81); // 9x9

  for (let k = 0; k < n; k++) {
    const w = weights[k];
    if (w < 1e-12) continue;
    const sx = srcPts[k][0];
    const sy = srcPts[k][1];
    const dx = dstPts[k][0];
    const dy = dstPts[k][1];

    // Row 1 of A for this correspondence
    const r1 = [0, 0, 0, -w * sx, -w * sy, -w, w * dy * sx, w * dy * sy, w * dy];
    // Row 2 of A for this correspondence
    const r2 = [w * sx, w * sy, w, 0, 0, 0, -w * dx * sx, -w * dx * sy, -w * dx];

    // Accumulate AtA += r1^T * r1 + r2^T * r2
    for (let i = 0; i < 9; i++) {
      for (let j = i; j < 9; j++) {
        const val = r1[i] * r1[j] + r2[i] * r2[j];
        AtA[i * 9 + j] += val;
        if (i !== j) AtA[j * 9 + i] += val; // symmetric
      }
    }
  }

  // Find smallest eigenvector of AtA using power iteration on (AtA)^{-1}
  // Instead, use a simpler approach: solve AtA * h = 0 by fixing h[8] = 1
  // This gives us a 8x8 system AtA_reduced * h_reduced = -AtA_col8
  const A8 = new Float64Array(64); // 8x8
  const b8 = new Float64Array(8);

  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      A8[i * 8 + j] = AtA[i * 9 + j];
    }
    b8[i] = -AtA[i * 9 + 8];
  }

  // Solve 8x8 system using Gaussian elimination
  const h8 = solveLinear8(A8, b8);
  if (!h8) return null;

  const H = new Float64Array(9);
  for (let i = 0; i < 8; i++) H[i] = h8[i];
  H[8] = 1.0;

  // Normalize so H[8] = 1 (already is)
  return H;
}

/**
 * Solve an 8x8 linear system Ax = b using Gaussian elimination with partial pivoting.
 * A is 8x8 row-major, b is length 8.
 * Returns solution x or null if singular.
 */
function solveLinear8(A, b) {
  const n = 8;
  // Copy to augmented matrix
  const m = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      m[i * (n + 1) + j] = A[i * n + j];
    }
    m[i * (n + 1) + n] = b[i];
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    // Find pivot
    let maxVal = Math.abs(m[col * (n + 1) + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(m[row * (n + 1) + col]);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }
    if (maxVal < 1e-12) return null;

    // Swap rows
    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = m[col * (n + 1) + j];
        m[col * (n + 1) + j] = m[maxRow * (n + 1) + j];
        m[maxRow * (n + 1) + j] = tmp;
      }
    }

    // Eliminate
    const pivot = m[col * (n + 1) + col];
    for (let row = col + 1; row < n; row++) {
      const factor = m[row * (n + 1) + col] / pivot;
      for (let j = col; j <= n; j++) {
        m[row * (n + 1) + j] -= factor * m[col * (n + 1) + j];
      }
    }
  }

  // Back substitution
  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = m[i * (n + 1) + n];
    for (let j = i + 1; j < n; j++) {
      sum -= m[i * (n + 1) + j] * x[j];
    }
    const diag = m[i * (n + 1) + i];
    if (Math.abs(diag) < 1e-12) return null;
    x[i] = sum / diag;
  }
  return x;
}

// Sample grayscale pixel value at (x, y) using bilinear interpolation
function sampleGray(gray, w, h, x, y) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  const x0 = Math.min(Math.max(ix, 0), w - 1);
  const x1 = Math.min(x0 + 1, w - 1);
  const y0 = Math.min(Math.max(iy, 0), h - 1);
  const y1 = Math.min(y0 + 1, h - 1);

  const v00 = gray[y0 * w + x0];
  const v10 = gray[y0 * w + x1];
  const v01 = gray[y1 * w + x0];
  const v11 = gray[y1 * w + x1];

  return (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 +
         (1 - fx) * fy * v01 + fx * fy * v11;
}

/**
 * Solve an NxN linear system Ax = b using Gaussian elimination with partial pivoting.
 * A is NxN row-major Float64Array, b is length N Float64Array.
 * Returns solution x or null if singular.
 */
function solveLinearN(A, b, n) {
  // Copy to augmented matrix
  const m = new Float64Array(n * (n + 1));
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      m[i * (n + 1) + j] = A[i * n + j];
    }
    m[i * (n + 1) + n] = b[i];
  }

  // Forward elimination with partial pivoting
  for (let col = 0; col < n; col++) {
    let maxVal = Math.abs(m[col * (n + 1) + col]);
    let maxRow = col;
    for (let row = col + 1; row < n; row++) {
      const val = Math.abs(m[row * (n + 1) + col]);
      if (val > maxVal) { maxVal = val; maxRow = row; }
    }
    if (maxVal < 1e-15) return null;

    if (maxRow !== col) {
      for (let j = 0; j <= n; j++) {
        const tmp = m[col * (n + 1) + j];
        m[col * (n + 1) + j] = m[maxRow * (n + 1) + j];
        m[maxRow * (n + 1) + j] = tmp;
      }
    }

    const pivot = m[col * (n + 1) + col];
    for (let row = col + 1; row < n; row++) {
      const factor = m[row * (n + 1) + col] / pivot;
      for (let j = col; j <= n; j++) {
        m[row * (n + 1) + j] -= factor * m[col * (n + 1) + j];
      }
    }
  }

  const x = new Float64Array(n);
  for (let i = n - 1; i >= 0; i--) {
    let sum = m[i * (n + 1) + n];
    for (let j = i + 1; j < n; j++) {
      sum -= m[i * (n + 1) + j] * x[j];
    }
    const diag = m[i * (n + 1) + i];
    if (Math.abs(diag) < 1e-15) return null;
    x[i] = sum / diag;
  }
  return x;
}
