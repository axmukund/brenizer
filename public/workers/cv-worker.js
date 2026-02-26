/**
 * cv-worker.js — Core computer vision worker using OpenCV.js
 *
 * Runs in a Web Worker context. Loads OpenCV.js via importScripts and
 * exposes message handlers for each pipeline stage. All OpenCV Mat objects
 * are carefully managed with try/finally to prevent WASM memory leaks.
 *
 * Pipeline stages implemented:
 *  - Feature extraction: ORB with CLAHE preprocessing for low-texture regions
 *  - Saliency: gradient magnitude (Sobel) + colour distinctness (Achanta) + focus (Laplacian)
 *  - Vignetting: radial polynomial V(r) = 1 + ar² + br⁴ estimated from luminance falloff
 *  - Matching: cross-checked kNN with MAGSAC++ σ-consensus scoring
 *  - MST: maximum spanning tree with BFS transform propagation + perspective validation
 *  - Refinement: Levenberg-Marquardt bundle adjustment with Huber loss
 *  - Exposure: per-channel RGB gain compensation (L2 + Huber options)
 *  - APAP mesh: Tikhonov-regularized weighted DLT with depth/face weighting
 *
 * Messages accepted:
 *  - {type:'init', baseUrl, opencvPath}
 *  - {type:'addImage', imageId, grayBuffer, width, height, rgbSmallBuffer?, depth?}
 *  - {type:'computeFeatures', orbParams}
 *  - {type:'computeSaliency'}
 *  - {type:'matchGraph', windowW, ratio, ransacThreshPx, minInliers, matchAllPairs}
 *  - {type:'buildGraph'}
 *  - {type:'refine', maxIters, huberDeltaPx, lambdaInit}
 *  - {type:'computeExposure', robustHuber?}
 *  - {type:'buildMST'}
 *  - {type:'computeLocalMesh', imageId, parentId, meshGrid, sigma, depthSigma, minSupport}
 *  - {type:'computeVignetting'}
 *
 * Messages posted back:
 *  - {type:'progress', stage, percent, info}
 *  - {type:'features', imageId, keypointsBuffer, descriptorsBuffer, descCols}
 *  - {type:'saliency', imageId, saliencyBuffer, width, height}
 *  - {type:'edges', edges: [...]}
 *  - {type:'transforms', refId, transforms: [...]}
 *  - {type:'exposure', gains: [...]}
 *  - {type:'vignetting', imageId, vignetteParams}
 *  - {type:'mst', refId, order, parent}
 *  - {type:'mesh', imageId, verticesBuffer, uvsBuffer, indicesBuffer, bounds}
 *  - {type:'error', message}
 */

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

    if (msg.type === 'clearImages') {
      images = {};
      edges = [];
      mstOrder = [];
      mstParent = {};
      transforms = {};
      postMessage({type:'progress', stage:'clearImages', percent:100, info:'cleared'});
      return;
    }

    if (msg.type === 'addImage') {
      const {imageId, grayBuffer, width, height, depth, rgbSmallBuffer} = msg;
      const gray = new Uint8ClampedArray(grayBuffer);
      const rgbSmall = rgbSmallBuffer ? new Uint8ClampedArray(rgbSmallBuffer) : null;
      if (images[imageId]) {
        // Merge: only overwrite gray if the buffer is non-trivial
        if (grayBuffer.byteLength > 1) {
          images[imageId].gray = gray;
          images[imageId].width = width;
          images[imageId].height = height;
        }
        if (rgbSmall) {
          images[imageId].rgbSmall = rgbSmall;
        }
        if (depth) {
          images[imageId].depth = new Uint16Array(depth);
        }
      } else {
        images[imageId] = {
          width, height, gray,
          rgbSmall: rgbSmall,
          depth: depth ? new Uint16Array(depth) : null,
          keypoints: null,
          descriptors: null,
          descCols: 0
        };
      }
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
        let clahe = null;
        let enhancedMat = null;
        let orb = null;
        let keypoints = null;
        let descriptors = null;
        let mask = null;

        try {
          mat = cv.matFromArray(img.height, img.width, cv.CV_8UC1, img.gray);

          // ── CLAHE (Contrast Limited Adaptive Histogram Equalization) ──
          // Apply CLAHE before ORB to improve feature detection in
          // low-texture regions (sky, fog, snow, uniform walls).
          // CLAHE enhances local contrast in 8×8 tile grid without
          // amplifying noise excessively (clipLimit = 3.0).
          // This is critical for grid-tile stitching where some tiles
          // may contain mostly sky or other featureless areas.
          try {
            clahe = new cv.CLAHE(3.0, new cv.Size(8, 8));
            enhancedMat = new cv.Mat();
            clahe.apply(mat, enhancedMat);
          } catch (claheErr) {
            // If CLAHE fails (shouldn't happen, but be safe), use raw image
            console.warn('CLAHE failed, using raw grayscale:', claheErr);
            enhancedMat = mat.clone();
          }

          orb = new cv.ORB(nFeatures);
          keypoints = new cv.KeyPointVector();
          descriptors = new cv.Mat();
          mask = new cv.Mat();

          // Run ORB on CLAHE-enhanced image for better keypoints
          orb.detectAndCompute(enhancedMat, mask, keypoints, descriptors);

          // ── Sub-pixel refinement ──────────────────────────────────
          // Refine keypoint locations to sub-pixel accuracy using OpenCV's
          // cornerSubPix.  This typically improves alignment RMS by 10-30%
          // for well-textured scenes.  We extract positions into a cv.Mat,
          // run cornerSubPix, then read back sub-pixel positions.
          const numKpRaw = keypoints.size();
          if (numKpRaw > 0 && typeof cv.cornerSubPix === 'function') {
            try {
              const cornersMat = new cv.Mat(numKpRaw, 1, cv.CV_32FC2);
              for (let i = 0; i < numKpRaw; i++) {
                const kp = keypoints.get(i);
                cornersMat.floatAt(i * 2) = kp.pt.x;
                cornersMat.floatAt(i * 2 + 1) = kp.pt.y;
              }
              const winSize = new cv.Size(3, 3);
              const zeroZone = new cv.Size(-1, -1);
              const criteria = new cv.TermCriteria(
                cv.TermCriteria_EPS + cv.TermCriteria_MAX_ITER, 20, 0.01
              );
              cv.cornerSubPix(enhancedMat, cornersMat, winSize, zeroZone, criteria);
              // Write back refined positions to keypoints
              for (let i = 0; i < numKpRaw; i++) {
                const kp = keypoints.get(i);
                kp.pt.x = cornersMat.floatAt(i * 2);
                kp.pt.y = cornersMat.floatAt(i * 2 + 1);
              }
              cornersMat.delete();
            } catch (subPixErr) {
              // Sub-pixel refinement is best-effort; ORB features may not
              // be true corners, so cornerSubPix can sometimes fail.
              // Fall back to original integer positions.
            }
          }

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

          // Store copies on the image record (worker-side reference)
          img.keypoints = new Float32Array(kps);
          img.descriptors = new Uint8Array(descBuf);
          img.descCols = descCols;

          // Transfer (zero-copy) the original buffers to the main thread
          postMessage({
            type:'features',
            imageId: id,
            keypointsBuffer: kps.buffer,
            descriptorsBuffer: descBuf.buffer,
            descCols
          }, [kps.buffer, descBuf.buffer]);

        } finally {
          if (mat) mat.delete();
          if (clahe) clahe.delete();
          if (enhancedMat) enhancedMat.delete();
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

    // ── Saliency map computation ──────────────────────────────────────
    // Computes a per-pixel importance score using:
    //   1. Gradient magnitude (Sobel-like) — edges and textures
    //   2. Colour distinctness — unusual colours relative to image mean
    //   3. Focus measure (Laplacian variance) — sharp regions score higher
    // The saliency map is used for:
    //   - Weighted feature matching (features on salient objects matter more)
    //   - Saliency-aware seam placement (avoid cutting through salient objects)
    //   - Blur detection (defocused regions get lower saliency)
    // Inspired by Itti-Koch-Niebur visual attention model (1998) and
    // frequency-tuned saliency (Achanta et al., CVPR 2009).
    if (msg.type === 'computeSaliency') {
      const ids = Object.keys(images);
      let done = 0;
      for (const id of ids) {
        const img = images[id];
        const w = img.width, h = img.height;
        const gray = img.gray;
        const saliency = new Float32Array(w * h);

        // 1. Gradient magnitude via Sobel 3x3 approximation
        const gradMag = new Float32Array(w * h);
        let maxGrad = 0;
        for (let y = 1; y < h - 1; y++) {
          for (let x = 1; x < w - 1; x++) {
            const idx = y * w + x;
            // Sobel X: [-1 0 1; -2 0 2; -1 0 1]
            const gx = -gray[(y-1)*w+(x-1)] - 2*gray[y*w+(x-1)] - gray[(y+1)*w+(x-1)]
                       +gray[(y-1)*w+(x+1)] + 2*gray[y*w+(x+1)] + gray[(y+1)*w+(x+1)];
            // Sobel Y: [-1 -2 -1; 0 0 0; 1 2 1]
            const gy = -gray[(y-1)*w+(x-1)] - 2*gray[(y-1)*w+x] - gray[(y-1)*w+(x+1)]
                       +gray[(y+1)*w+(x-1)] + 2*gray[(y+1)*w+x] + gray[(y+1)*w+(x+1)];
            const mag = Math.sqrt(gx * gx + gy * gy);
            gradMag[idx] = mag;
            if (mag > maxGrad) maxGrad = mag;
          }
        }
        // Normalize gradient to [0,1]
        if (maxGrad > 0) {
          for (let i = 0; i < gradMag.length; i++) gradMag[i] /= maxGrad;
        }

        // 2. Colour distinctness (if RGB available)
        const colDist = new Float32Array(w * h);
        if (img.rgbSmall) {
          const rgb = img.rgbSmall;
          // Compute image mean colour
          let meanR = 0, meanG = 0, meanB = 0;
          const n = w * h;
          for (let i = 0; i < n; i++) {
            meanR += rgb[i * 3]; meanG += rgb[i * 3 + 1]; meanB += rgb[i * 3 + 2];
          }
          meanR /= n; meanG /= n; meanB /= n;
          // Distance from mean (Achanta frequency-tuned approach)
          let maxDist = 0;
          for (let i = 0; i < n; i++) {
            const dr = rgb[i * 3] - meanR;
            const dg = rgb[i * 3 + 1] - meanG;
            const db = rgb[i * 3 + 2] - meanB;
            const dist = Math.sqrt(dr * dr + dg * dg + db * db);
            colDist[i] = dist;
            if (dist > maxDist) maxDist = dist;
          }
          if (maxDist > 0) {
            for (let i = 0; i < n; i++) colDist[i] /= maxDist;
          }
        }

        // 3. Focus measure via local Laplacian variance (blur detection)
        // High variance = sharp = in focus; low variance = blurred (Brenizer bokeh)
        const focusMap = new Float32Array(w * h);
        const blockR = 4; // half-block radius for local variance
        let maxFocus = 0;
        for (let y = blockR; y < h - blockR; y++) {
          for (let x = blockR; x < w - blockR; x++) {
            // Compute Laplacian at this pixel
            const lap = 4 * gray[y * w + x] - gray[(y-1)*w+x] - gray[(y+1)*w+x]
                        - gray[y*w+(x-1)] - gray[y*w+(x+1)];
            const lapSq = lap * lap;
            focusMap[y * w + x] = lapSq;
            if (lapSq > maxFocus) maxFocus = lapSq;
          }
        }
        if (maxFocus > 0) {
          for (let i = 0; i < focusMap.length; i++) focusMap[i] /= maxFocus;
        }

        // Combine: saliency = 0.4 * gradient + 0.3 * colorDistinctness + 0.3 * focus
        for (let i = 0; i < w * h; i++) {
          saliency[i] = 0.4 * gradMag[i] + 0.3 * colDist[i] + 0.3 * focusMap[i];
        }

        // Store on image for later use in matching and seam placement
        img.saliency = saliency;
        img.focusMap = focusMap;

        // Compute per-image blur score (mean focus measure) — used for
        // blur-aware feature weighting in Brenizer composites
        let focusSum = 0, focusCount = 0;
        for (let i = 0; i < focusMap.length; i++) {
          if (focusMap[i] > 0) { focusSum += focusMap[i]; focusCount++; }
        }
        img.blurScore = focusCount > 0 ? focusSum / focusCount : 0;

        // Transfer a *copy* of the saliency buffer to main thread —
        // the original stays on the image for later seam-placement use.
        const saliencyCopy = new Float32Array(saliency);
        postMessage({
          type: 'saliency',
          imageId: id,
          saliencyBuffer: saliencyCopy.buffer,
          width: w,
          height: h,
          blurScore: img.blurScore,
        }, [saliencyCopy.buffer]);

        done++;
        postMessage({type:'progress', stage:'saliency', percent: Math.round(100 * done / ids.length), info: `${done}/${ids.length}`});
      }
      return;
    }

    // ── Vignetting estimation ─────────────────────────────────────────
    // Estimates radial vignetting from the image luminance distribution.
    // Models vignetting as V(r) = 1 + a*r² + b*r⁴ where r = normalized
    // distance from image center (0 at center, 1 at corners).
    // The coefficients are estimated by comparing the mean luminance at
    // different radial distances — natural images are assumed to have
    // roughly uniform expected brightness at all radii.
    // This follows the approach in PTGui / Autopano (polynomial radial model).
    if (msg.type === 'computeVignetting') {
      const ids = Object.keys(images);
      const pooled = msg.pooled || false; // If true, estimate single model for all images

      if (pooled && ids.length >= 2) {
        // ── Pooled vignetting estimation ─────────────────────────
        // When all images share the same lens+aperture (Brenizer method),
        // pool radial luminance samples from ALL images for a single,
        // more robust vignetting model.  N images = N× more data,
        // greatly reducing noise in the polynomial fit.
        const nBins = 10;
        const binSum = new Float64Array(nBins);
        const binCount = new Float64Array(nBins);
        const step = 8;

        for (const id of ids) {
          const img = images[id];
          const w = img.width, h = img.height;
          const gray = img.gray;
          const cx = w / 2, cy = h / 2;
          const maxR = Math.sqrt(cx * cx + cy * cy);
          for (let y = 0; y < h; y += step) {
            for (let x = 0; x < w; x += step) {
              const dx = x - cx, dy = y - cy;
              const r = Math.sqrt(dx * dx + dy * dy) / maxR;
              const bin = Math.min(Math.floor(r * nBins), nBins - 1);
              binSum[bin] += gray[y * w + x];
              binCount[bin]++;
            }
          }
        }

        const binMean = new Float64Array(nBins);
        const centerMean = binCount[0] > 0 ? binSum[0] / binCount[0] : 128;
        for (let i = 0; i < nBins; i++) {
          binMean[i] = binCount[i] > 0 ? binSum[i] / binCount[i] : centerMean;
        }

        let sumR2R2 = 0, sumR2R4 = 0, sumR4R4 = 0;
        let sumR2T = 0, sumR4T = 0;
        for (let i = 1; i < nBins; i++) {
          const r = (i + 0.5) / nBins;
          const r2 = r * r, r4 = r2 * r2;
          const target = (centerMean > 10 && binMean[i] > 10)
            ? (centerMean / binMean[i] - 1.0) : 0;
          sumR2R2 += r2 * r2; sumR2R4 += r2 * r4; sumR4R4 += r4 * r4;
          sumR2T += r2 * target; sumR4T += r4 * target;
        }
        const det = sumR2R2 * sumR4R4 - sumR2R4 * sumR2R4;
        let a = 0, b = 0;
        if (Math.abs(det) > 1e-10) {
          a = (sumR4R4 * sumR2T - sumR2R4 * sumR4T) / det;
          b = (sumR2R2 * sumR4T - sumR2R4 * sumR2T) / det;
        }
        a = Math.max(-2, Math.min(2, a));
        b = Math.max(-2, Math.min(2, b));

        // Broadcast same params to all images
        for (const id of ids) {
          images[id].vignetteParams = { a, b };
          postMessage({ type: 'vignetting', imageId: id, vignetteParams: { a, b } });
        }
        postMessage({type:'progress', stage:'vignetting', percent:100, info:`pooled: a=${a.toFixed(4)}, b=${b.toFixed(4)}`});
        return;
      }

      // Per-image vignetting (original path)
      for (const id of ids) {
        const img = images[id];
        const w = img.width, h = img.height;
        const gray = img.gray;
        const cx = w / 2, cy = h / 2;
        const maxR = Math.sqrt(cx * cx + cy * cy);

        // Bin luminance by radial distance (10 bins)
        const nBins = 10;
        const binSum = new Float64Array(nBins);
        const binCount = new Float64Array(nBins);
        const step = 8; // sample every 8th pixel for speed
        for (let y = 0; y < h; y += step) {
          for (let x = 0; x < w; x += step) {
            const dx = x - cx, dy = y - cy;
            const r = Math.sqrt(dx * dx + dy * dy) / maxR;
            const bin = Math.min(Math.floor(r * nBins), nBins - 1);
            binSum[bin] += gray[y * w + x];
            binCount[bin]++;
          }
        }

        // Compute mean luminance per bin
        const binMean = new Float64Array(nBins);
        const centerMean = binCount[0] > 0 ? binSum[0] / binCount[0] : 128;
        for (let i = 0; i < nBins; i++) {
          binMean[i] = binCount[i] > 0 ? binSum[i] / binCount[i] : centerMean;
        }

        // Fit polynomial: V(r) = 1 + a*r² + b*r⁴
        // We want V(r) * observed(r) ≈ centerMean → V(r) ≈ centerMean / observed(r)
        // Least-squares fit: minimize Σ_i (V(r_i) - target_i)²
        // where target_i = centerMean / binMean[i], r_i = bin center
        let sumR2R2 = 0, sumR2R4 = 0, sumR4R4 = 0;
        let sumR2T = 0, sumR4T = 0;
        for (let i = 1; i < nBins; i++) { // skip center bin
          const r = (i + 0.5) / nBins;
          const r2 = r * r, r4 = r2 * r2;
          const target = (centerMean > 10 && binMean[i] > 10)
            ? (centerMean / binMean[i] - 1.0) : 0;
          sumR2R2 += r2 * r2; sumR2R4 += r2 * r4; sumR4R4 += r4 * r4;
          sumR2T += r2 * target; sumR4T += r4 * target;
        }
        const det = sumR2R2 * sumR4R4 - sumR2R4 * sumR2R4;
        let a = 0, b = 0;
        if (Math.abs(det) > 1e-10) {
          a = (sumR4R4 * sumR2T - sumR2R4 * sumR4T) / det;
          b = (sumR2R2 * sumR4T - sumR2R4 * sumR2T) / det;
        }
        // Clamp to physically reasonable range
        a = Math.max(-2, Math.min(2, a));
        b = Math.max(-2, Math.min(2, b));

        img.vignetteParams = { a, b };
        postMessage({ type: 'vignetting', imageId: id, vignetteParams: { a, b } });
      }
      postMessage({type:'progress', stage:'vignetting', percent:100, info:'done'});
      return;
    }

    // ── Cylindrical projection (PTGui-style) ───────────────────────────
    // Projects image points into cylindrical coordinates for wide-FoV stitching.
    // Used by PTGui, AutoStitch, and other professional panorama tools.
    // x_cyl = f * atan2(x - cx, f)
    // y_cyl = f * (y - cy) / sqrt((x - cx)² + f²)
    // where f = estimated focal length in pixels, (cx,cy) = image centre.
    if (msg.type === 'cylindricalProject') {
      const ids = Object.keys(images);
      for (const id of ids) {
        const img = images[id];
        const w = img.width, h = img.height;
        // Estimate focal length from image width (assume ~50mm on 35mm sensor ≈ FoV ~40°)
        const fov = msg.fovDegrees || 40;
        const f = (w / 2) / Math.tan((fov * Math.PI / 180) / 2);
        const cx = w / 2, cy = h / 2;

        // Project all keypoints to cylindrical
        if (img.keypoints && img.keypoints.length > 0) {
          const n = img.keypoints.length / 2;
          const cylKp = new Float32Array(n * 2);
          for (let i = 0; i < n; i++) {
            const x = img.keypoints[i * 2] - cx;
            const y = img.keypoints[i * 2 + 1] - cy;
            cylKp[i * 2] = f * Math.atan2(x, f) + cx;
            cylKp[i * 2 + 1] = f * y / Math.sqrt(x * x + f * f) + cy;
          }
          img.cylindricalKeypoints = cylKp;
        }

        // Store focal length estimate
        img.focalLength = f;
      }
      postMessage({type:'progress', stage:'cylindrical', percent:100, info:'done'});
      return;
    }

    // ── Lens distortion estimation (Brown-Conrady model) ───────────────
    // Estimates radial distortion from homography residuals.
    // Models distortion as: r_d = r(1 + k1*r² + k2*r⁴)
    // where r = normalised distance from principal point.
    // Uses matched point reprojection errors after homography fitting:
    // if systematic radial pattern exists in residuals, we can estimate k1, k2.
    // This is the same model used by PTGui, OpenCV, and Lensfun.
    if (msg.type === 'estimateLensDistortion') {
      const ids = Object.keys(images);
      for (const id of ids) {
        const img = images[id];
        const w = img.width, h = img.height;
        const cx = w / 2, cy = h / 2;
        const maxR = Math.sqrt(cx * cx + cy * cy);

        // Collect reprojection residuals from all edges involving this image
        let sumK1Num = 0, sumK1Den = 0;
        for (const edge of edges) {
          if (edge.i !== id && edge.j !== id) continue;
          const H = edge.H;
          if (!H || H.length < 9) continue;

          // Use inlier correspondences
          const inliers = edge.inliers;
          const n = inliers.length / 4;
          for (let k = 0; k < n; k++) {
            const xi = inliers[k * 4], yi = inliers[k * 4 + 1];
            const xj = inliers[k * 4 + 2], yj = inliers[k * 4 + 3];

            // Determine which point belongs to this image
            const isI = (edge.i === id);
            const px = isI ? xi : xj, py = isI ? yi : yj;
            const qx = isI ? xj : xi, qy = isI ? yj : yi;

            // Normalised radius
            const dx = px - cx, dy = py - cy;
            const r = Math.sqrt(dx * dx + dy * dy) / maxR;
            const r2 = r * r;

            // Compute homography-predicted position
            const d = H[6] * px + H[7] * py + H[8];
            if (Math.abs(d) < 1e-10) continue;
            const predX = (H[0] * px + H[1] * py + H[2]) / d;
            const predY = (H[3] * px + H[4] * py + H[5]) / d;

            // Radial component of residual (dot product with radial direction)
            const resX = qx - predX, resY = qy - predY;
            const radDir = r > 1e-6 ? [dx / (r * maxR), dy / (r * maxR)] : [0, 0];
            const radialRes = resX * radDir[0] + resY * radDir[1];

            // Least squares: k1 ≈ sum(radialRes * r²) / sum(r⁴)
            sumK1Num += radialRes * r2;
            sumK1Den += r2 * r2;
          }
        }

        const k1 = sumK1Den > 1e-10 ? sumK1Num / sumK1Den : 0;
        // Clamp to reasonable range for typical camera lenses
        img.distortionK1 = Math.max(-0.5, Math.min(0.5, k1));
        img.distortionK2 = 0; // Higher-order term requires more data

        postMessage({
          type: 'distortion',
          imageId: id,
          k1: img.distortionK1,
          k2: img.distortionK2,
        });
      }
      postMessage({type:'progress', stage:'distortion', percent:100, info:'done'});
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
        let matchesRev = null;

        try {
          matI = cv.matFromArray(numI, descColsI, cv.CV_8UC1, imgI.descriptors);
          matJ = cv.matFromArray(numJ, descColsJ, cv.CV_8UC1, imgJ.descriptors);

          bf = new cv.BFMatcher(cv.NORM_HAMMING, false);
          matches = new cv.DMatchVectorVector();
          bf.knnMatch(matI, matJ, matches, 2);

          // Forward ratio test: I → J
          const fwdBest = new Map(); // queryIdx → {trainIdx, distance}
          for (let m = 0; m < matches.size(); m++) {
            const match = matches.get(m);
            if (match.size() >= 2) {
              const m1 = match.get(0);
              const m2 = match.get(1);
              if (m1.distance < ratio * m2.distance) {
                const qi = m1.queryIdx;
                const ti = m1.trainIdx;
                if (qi < numI && ti < numJ) {
                  fwdBest.set(qi, {trainIdx: ti, distance: m1.distance});
                }
              }
            }
          }

          // Reverse match: J → I for cross-check
          matchesRev = new cv.DMatchVectorVector();
          bf.knnMatch(matJ, matI, matchesRev, 2);
          const revBest = new Map(); // queryIdx(J) → trainIdx(I)
          for (let m = 0; m < matchesRev.size(); m++) {
            const match = matchesRev.get(m);
            if (match.size() >= 2) {
              const m1 = match.get(0);
              const m2 = match.get(1);
              if (m1.distance < ratio * m2.distance) {
                revBest.set(m1.queryIdx, m1.trainIdx);
              }
            }
          }
          matchesRev.delete();
          matchesRev = null;

          // Cross-check filter: keep only mutual best matches
          const goodPtsI = [];
          const goodPtsJ = [];
          for (const [qi, fwd] of fwdBest) {
            const revTrain = revBest.get(fwd.trainIdx);
            if (revTrain === qi) {
              // Mutual best match confirmed
              goodPtsI.push(imgI.keypoints[qi * 2], imgI.keypoints[qi * 2 + 1]);
              goodPtsJ.push(imgJ.keypoints[fwd.trainIdx * 2], imgJ.keypoints[fwd.trainIdx * 2 + 1]);
            }
          }

          if (goodPtsI.length / 2 < minInliers) { done++; continue; }

          // ── MAGSAC++-style RANSAC with marginalized scoring ──────────
          // Run OpenCV RANSAC to get an initial homography, then re-score
          // every correspondence using a σ-consensus score that marginalizes
          // over unknown noise scale σ ∈ (0, σ_max]. Points with lower
          // reprojection error contribute more, giving a soft inlier count
          // that is more discriminative than a fixed-threshold hard count.
          // Reference: Baráth & Matas, "MAGSAC++", CVPR 2020.
          let srcPts = null, dstPts = null, inlierMask = null, H = null;
          try {
          srcPts = cv.matFromArray(goodPtsI.length / 2, 1, cv.CV_32FC2, new Float32Array(goodPtsI));
          dstPts = cv.matFromArray(goodPtsJ.length / 2, 1, cv.CV_32FC2, new Float32Array(goodPtsJ));
          inlierMask = new cv.Mat();
          H = cv.findHomography(srcPts, dstPts, cv.RANSAC, ransacThreshPx, inlierMask);

          if (H.rows === 3 && H.cols === 3) {
            const hd = H.data64F;
            const nPts = goodPtsI.length / 2;

            // Compute per-point reprojection errors² for MAGSAC scoring
            const reproj2 = new Float64Array(nPts);
            for (let k = 0; k < nPts; k++) {
              const xi = goodPtsI[k * 2], yi = goodPtsI[k * 2 + 1];
              const xj = goodPtsJ[k * 2], yj = goodPtsJ[k * 2 + 1];
              const d = hd[6] * xi + hd[7] * yi + hd[8];
              if (Math.abs(d) < 1e-10) { reproj2[k] = 1e6; continue; }
              const px = (hd[0] * xi + hd[1] * yi + hd[2]) / d;
              const py = (hd[3] * xi + hd[4] * yi + hd[5]) / d;
              reproj2[k] = (px - xj) ** 2 + (py - yj) ** 2;
            }

            // MAGSAC++ σ-consensus score: for each point, integrate the
            // probability of being an inlier over σ ∈ (0, σ_max].
            // Weight_k = max(0, 1 − (err_k² / σ_max²))^2  (Epanechnikov-like kernel)
            // This gives soft weights ∈ [0,1] — no hard threshold needed.
            // σ_max = 3× RANSAC threshold: empirically captures 99%+ of true inliers
            // while still rejecting gross outliers (Baráth & Matas, CVPR 2020).
            const sigmaMax = ransacThreshPx * 3; // σ_max ≈ 3× RANSAC threshold
            const sigmaMax2 = sigmaMax * sigmaMax;
            const magsacWeights = new Float64Array(nPts);
            let magsacInlierCount = 0;
            const inliers = [];

            for (let k = 0; k < nPts; k++) {
              const u = reproj2[k] / sigmaMax2;
              if (u >= 1.0) {
                magsacWeights[k] = 0;
                continue; // hard outlier beyond σ_max
              }
              // Epanechnikov kernel: w = (1 − u)²
              const w = (1.0 - u) * (1.0 - u);
              magsacWeights[k] = w;
              // Collect inliers at 2× RANSAC threshold (4× in squared space) for
              // backward-compat with LM and APAP which expect a discrete inlier set.
              if (reproj2[k] <= ransacThreshPx * ransacThreshPx * 4) {
                magsacInlierCount++;
                inliers.push({
                  xi: goodPtsI[k * 2], yi: goodPtsI[k * 2 + 1],
                  xj: goodPtsJ[k * 2], yj: goodPtsJ[k * 2 + 1],
                  magsacWeight: w,
                  origIdx: k  // original index into reproj2 for correct RMS lookup
                });
              }
            }

            if (magsacInlierCount >= minInliers) {
              // Compute weighted RMS using MAGSAC weights (better quality metric)
              let wRmsSum = 0, wSum = 0;
              for (const inl of inliers) {
                wRmsSum += inl.magsacWeight * reproj2[inl.origIdx];
                wSum += inl.magsacWeight;
              }
              // Fallback to unweighted if weights sum to ~0
              let rms;
              if (wSum > 1e-6) {
                rms = Math.sqrt(wRmsSum / wSum);
              } else {
                let rmsSum = 0;
                for (const inl of inliers) {
                  const d = hd[6] * inl.xi + hd[7] * inl.yi + hd[8];
                  if (Math.abs(d) < 1e-10) continue; // skip degenerate point
                  const px = (hd[0] * inl.xi + hd[1] * inl.yi + hd[2]) / d;
                  const py = (hd[3] * inl.xi + hd[4] * inl.yi + hd[5]) / d;
                  rmsSum += (px - inl.xj) ** 2 + (py - inl.yj) ** 2;
                }
                rms = magsacInlierCount > 0 ? Math.sqrt(rmsSum / magsacInlierCount) : 1e6;
              }

              const HBuf = new Float64Array(9);
              HBuf.set(hd.slice(0, 9));

              // CRITICAL: Validate homography for physical plausibility
              if (!isHomographyValid(HBuf, imgI.width, imgI.height)) {
                console.warn(`  Rejected pair: ${idI} ↔ ${idJ} (${magsacInlierCount} inliers, RMS=${rms.toFixed(2)})`);
                done++; continue;
              }

              // Detect near-duplicates: images that overlap almost completely with
              // minimal translation are likely duplicate shots and should be flagged.
              // Thresholds: RMS < 2px (very tight alignment), >80% inliers (most
              // points match), translation < 5% of image size (nearly stationary).
              const inlierRatio = magsacInlierCount / nPts;
              const tx = HBuf[2], ty = HBuf[5];
              const maxDim = Math.max(imgI.width, imgI.height);
              const translation = Math.sqrt(tx * tx + ty * ty);
              const translationRatio = translation / maxDim;
              const isDuplicate = (rms < 2.0 && inlierRatio > 0.8 && translationRatio < 0.05);

              const inliersBuf = new Float32Array(magsacInlierCount * 4);
              for (let k = 0; k < magsacInlierCount; k++) {
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
                inlierCount: magsacInlierCount,
                isDuplicate
              });
            }
          }
          } finally {
            if (srcPts) srcPts.delete();
            if (dstPts) dstPts.delete();
            if (inlierMask) inlierMask.delete();
            if (H) H.delete();
          }
        } finally {
          if (matI) matI.delete();
          if (matJ) matJ.delete();
          if (bf) bf.delete();
          if (matches) matches.delete();
          if (matchesRev) matchesRev.delete();
        }
        done++;
        postMessage({type:'progress', stage:'matching', percent: Math.round(100 * done / pairs.length), info: `${done}/${pairs.length}`});
      }

      // If graph is disconnected and we didn't match all pairs, retry with all pairs
      if (!matchAllPairs && edges.length > 0) {
        const connected = checkConnectivity(ids, edges);
        if (!connected) {
          postMessage({type:'progress', stage:'matching', percent: 100, info: 'Graph disconnected — some images may not align'});
          console.warn('Match graph is disconnected. Some images are not connected to the reference.');
        }
      }

      // Detect and report near-duplicates
      const duplicatePairs = edges.filter(e => e.isDuplicate).map(e => [e.i, e.j]);
      if (duplicatePairs.length > 0) {
        postMessage({type:'progress', stage:'matching', percent:100, info:`Found ${duplicatePairs.length} near-duplicate pair(s)`});
      }

      // Send edges
      const edgeMessages = edges.map(e => ({
        i: e.i,
        j: e.j,
        HBuffer: e.H.buffer,
        inliersBuffer: e.inliersBuf.buffer,
        rms: e.rms,
        inlierCount: e.inlierCount,
        isDuplicate: e.isDuplicate || false
      }));
      postMessage({type:'edges', edges: edgeMessages, duplicatePairs});
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

      // Select reference: combine edge weight sum with graph centrality.
      // PTGui selects the most "connected" image that also minimizes the
      // maximum transformation chain to any other image.  We use a score
      // that combines: (1) total edge quality, (2) number of edges (degree),
      // and (3) centrality approximated from pairwise connections.
      const weightSum = {};
      const degree = {};
      for (const id of ids) { weightSum[id] = 0; degree[id] = 0; }
      for (const e of edges) {
        const w = e.inlierCount / (e.rms + 0.1);
        weightSum[e.i] = (weightSum[e.i] || 0) + w;
        weightSum[e.j] = (weightSum[e.j] || 0) + w;
        degree[e.i] = (degree[e.i] || 0) + 1;
        degree[e.j] = (degree[e.j] || 0) + 1;
      }
      // Centrality bonus: prefer images connected to many others.
      // The combined score: weightSum * sqrt(degree) penalizes leaf nodes
      // and promotes images with many high-quality connections.
      refId = ids.reduce((a, b) => {
        const scoreA = (weightSum[a] || 0) * Math.sqrt(degree[a] || 1);
        const scoreB = (weightSum[b] || 0) * Math.sqrt(degree[b] || 1);
        return scoreA >= scoreB ? a : b;
      });

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

      // Also build a full adjacency from all edges (for fallback path searches)
      const fullAdj = {};
      for (const id of ids) fullAdj[id] = [];
      for (const e of edges) {
        fullAdj[e.i].push({to: e.j, edge: e});
        fullAdj[e.j].push({to: e.i, edge: e});
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

        // Normalize so T[8] = 1 to keep perspective terms well-scaled
        if (Math.abs(T_node[8]) > 1e-10) {
          const inv = 1.0 / T_node[8];
          for (let k = 0; k < 9; k++) T_node[k] *= inv;
        }

        // Validate: check that all 4 corners of the image project with positive denom.
        // If any corner is behind the camera, the transform has extreme perspective.
        const nodeImg = images[node];
        if (nodeImg) {
          const nw = nodeImg.width, nh = nodeImg.height;
          const corners = [[0,0], [nw,0], [nw,nh], [0,nh]];
          let hasExtremePerspective = false;
          for (const [cx, cy] of corners) {
            const denom = T_node[6] * cx + T_node[7] * cy + T_node[8];
            if (denom < 0.05) { hasExtremePerspective = true; break; }
          }
          if (hasExtremePerspective) {
            console.warn(`Transform for ${node} has extreme perspective (corner behind camera)`);
            // Try to find a better path: look for alternative parent edges
            let found = false;
            for (const altAdj of (fullAdj[node] || [])) {
              const altPar = altAdj.to;
              if (altPar === par || !transforms[altPar]) continue;
              const altEdge = altAdj.edge;
              let T_alt;
              if (altEdge.i === node && altEdge.j === altPar) {
                T_alt = mulMat3(transforms[altPar], altEdge.H);
              } else {
                const Hinv2 = invertH(altEdge.H);
                if (!Hinv2) continue;
                T_alt = mulMat3(transforms[altPar], Hinv2);
              }
              if (Math.abs(T_alt[8]) > 1e-10) {
                const inv2 = 1.0 / T_alt[8];
                for (let k = 0; k < 9; k++) T_alt[k] *= inv2;
              }
              // Re-check perspective
              let altOk = true;
              for (const [cx, cy] of corners) {
                const denom = T_alt[6] * cx + T_alt[7] * cy + T_alt[8];
                if (denom < 0.05) { altOk = false; break; }
              }
              if (altOk) {
                console.warn(`  → Using alternative parent ${altPar} for ${node}`);
                T_node = T_alt;
                found = true;
                break;
              }
            }
            if (!found) {
              console.warn(`  → No alternative path found; keeping original transform`);
            }
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
      // ── Levenberg-Marquardt bundle adjustment over homography parameters ──
      // Jointly refine all global transforms to minimise reprojection error across
      // all inlier correspondences. Reference image is fixed (identity).
      //
      // Parameterisation: each non-reference image has 8 free parameters
      // (H[0..7], with H[8]=1). Residuals are (px - xj, py - yj) for each
      // inlier in every edge, where (px,py) = T_j^{-1} * T_i * (xi,yi).

      const maxIters = msg.maxIters || 30;
      const huberDelta = msg.huberDeltaPx || 2.0;
      const lambdaInit = msg.lambdaInit || 0.01;

      const ids = Object.keys(transforms);
      const nImages = ids.length;
      const refIdx = ids.indexOf(refId);

      // Collect all inlier correspondences from edges
      const residualObs = []; // {idxI, idxJ, xi, yi, xj, yj}
      for (const e of edges) {
        const ii = ids.indexOf(e.i);
        const ji = ids.indexOf(e.j);
        if (ii < 0 || ji < 0) continue;
        for (const inl of e.inliers) {
          residualObs.push({
            idxI: ii, idxJ: ji,
            xi: inl.xi, yi: inl.yi,
            xj: inl.xj, yj: inl.yj
          });
        }
      }

      if (residualObs.length < 4 || nImages < 2) {
        // Not enough data for refinement
        const tList = Object.entries(transforms).map(([id, T]) => ({
          imageId: id,
          TBuffer: new Float64Array(T).buffer
        }));
        postMessage({type:'transforms', refId, transforms: tList});
        postMessage({type:'progress', stage:'refine', percent:100, info:'skipped (insufficient data)'});
        return;
      }

      // ── Shared-intrinsics rotation-model BA ─────────────────────────────
      // When all images share same camera (aperture, focal length, ISO),
      // parameterise each global transform as T_i = K R_i K^{-1} where
      //   K = [[f,0,cx],[0,f,cy],[0,0,1]]   (shared intrinsics)
      //   R_i = Rodrigues(r_i)               (per-image rotation, 3 params)
      // Total parameters: 3(N-1) + 1 vs 8(N-1), ~2.6× fewer.
      // More constrained → better conditioned → more robust.
      if (msg.sameCameraSettings && nImages >= 2) {
        const refImg = images[refId];
        const cx = refImg ? refImg.width / 2 : 0;
        const cy = refImg ? refImg.height / 2 : 0;

        // ── Estimate shared focal length from existing homographies ──
        // Centre each transform and use column orthogonality constraint:
        //   f² = −(Tc[0]·Tc[1] + Tc[3]·Tc[4]) / (Tc[6]·Tc[7])
        const fEstimates = [];
        const C = [1,0,-cx, 0,1,-cy, 0,0,1];
        const Cinv = [1,0,cx, 0,1,cy, 0,0,1];
        for (let i = 0; i < nImages; i++) {
          if (i === refIdx) continue;
          const T = transforms[ids[i]];
          const Tc = mulMat3(mulMat3(C, T), Cinv);
          // Orthogonality constraint
          const d1 = Tc[6] * Tc[7];
          if (Math.abs(d1) > 1e-12) {
            const f2 = -(Tc[0]*Tc[1] + Tc[3]*Tc[4]) / d1;
            if (f2 > 100) fEstimates.push(Math.sqrt(f2));
          }
          // Equal-norm constraint
          const d2 = Tc[7]*Tc[7] - Tc[6]*Tc[6];
          if (Math.abs(d2) > 1e-12) {
            const f2 = (Tc[0]*Tc[0] + Tc[3]*Tc[3] - Tc[1]*Tc[1] - Tc[4]*Tc[4]) / d2;
            if (f2 > 100) fEstimates.push(Math.sqrt(f2));
          }
        }
        let f0;
        if (fEstimates.length >= 1) {
          fEstimates.sort((a, b) => a - b);
          f0 = fEstimates[Math.floor(fEstimates.length / 2)];
        } else {
          f0 = (refImg ? refImg.width : 800) / (2 * Math.tan(25 * Math.PI / 180));
        }
        // Clamp to plausible range (10–20000 px)
        f0 = Math.max(10, Math.min(20000, f0));

        // ── Decompose each transform → Rodrigues vector ──
        function extractRodrigues(R) {
          const tr = R[0] + R[4] + R[8];
          const cosT = Math.max(-1, Math.min(1, (tr - 1) / 2));
          const theta = Math.acos(cosT);
          if (theta < 1e-6) return [0, 0, 0];
          const s = theta / (2 * Math.sin(theta));
          return [s * (R[7]-R[5]), s * (R[2]-R[6]), s * (R[3]-R[1])];
        }
        function rodriguesToR(rx, ry, rz) {
          const theta = Math.sqrt(rx*rx + ry*ry + rz*rz);
          if (theta < 1e-10) return [1,0,0, 0,1,0, 0,0,1];
          const kx=rx/theta, ky=ry/theta, kz=rz/theta;
          const c=Math.cos(theta), s=Math.sin(theta), v=1-c;
          return [
            kx*kx*v+c,    kx*ky*v-kz*s, kx*kz*v+ky*s,
            kx*ky*v+kz*s, ky*ky*v+c,    ky*kz*v-kx*s,
            kx*kz*v-ky*s, ky*kz*v+kx*s, kz*kz*v+c
          ];
        }

        // Parameter vector: [f, r0_x, r0_y, r0_z, r1_x, ...]  (skip ref)
        const siPC = 1 + (nImages - 1) * 3;
        const siP = new Float64Array(siPC);
        siP[0] = f0;
        const siOff = new Int32Array(nImages);
        let soff = 1;
        for (let i = 0; i < nImages; i++) {
          if (i === refIdx) { siOff[i] = -1; continue; }
          siOff[i] = soff;
          const T = transforms[ids[i]];
          const K = [f0,0,cx, 0,f0,cy, 0,0,1];
          const Ki = [1/f0,0,-cx/f0, 0,1/f0,-cy/f0, 0,0,1];
          const R = mulMat3(mulMat3(Ki, T), K);
          const rv = extractRodrigues(R);
          siP[soff] = rv[0]; siP[soff+1] = rv[1]; siP[soff+2] = rv[2];
          soff += 3;
        }

        const _siIdent = [1,0,0, 0,1,0, 0,0,1];
        function siGetT(idx) {
          if (idx === refIdx) return _siIdent;
          const f = siP[0], o = siOff[idx];
          const R = rodriguesToR(siP[o], siP[o+1], siP[o+2]);
          const K = [f,0,cx, 0,f,cy, 0,0,1];
          const Ki = [1/f,0,-cx/f, 0,1/f,-cy/f, 0,0,1];
          return mulMat3(mulMat3(K, R), Ki);
        }

        function siCost() {
          let c = 0;
          for (const obs of residualObs) {
            const Ti = siGetT(obs.idxI);
            const di = Ti[6]*obs.xi + Ti[7]*obs.yi + Ti[8];
            if (Math.abs(di) < 1e-10) { c += huberDelta*huberDelta; continue; }
            const pix = (Ti[0]*obs.xi + Ti[1]*obs.yi + Ti[2]) / di;
            const piy = (Ti[3]*obs.xi + Ti[4]*obs.yi + Ti[5]) / di;
            const Tj = siGetT(obs.idxJ);
            const dj = Tj[6]*obs.xj + Tj[7]*obs.yj + Tj[8];
            if (Math.abs(dj) < 1e-10) { c += huberDelta*huberDelta; continue; }
            const pjx = (Tj[0]*obs.xj + Tj[1]*obs.yj + Tj[2]) / dj;
            const pjy = (Tj[3]*obs.xj + Tj[4]*obs.yj + Tj[5]) / dj;
            const dx = pix-pjx, dy = piy-pjy, r2 = dx*dx+dy*dy;
            c += (r2 <= huberDelta*huberDelta) ? r2 : 2*huberDelta*Math.sqrt(r2) - huberDelta*huberDelta;
          }
          return c;
        }

        // ── LM iteration ──
        const siEps = 1e-6;
        let siLambda = lambdaInit;
        let siPrev = siCost();
        const siJtJ = new Float64Array(siPC * siPC);
        const siJtr = new Float64Array(siPC);
        const siJR = new Float64Array(siPC * 2);

        for (let iter = 0; iter < maxIters; iter++) {
          siJtJ.fill(0); siJtr.fill(0);

          for (const obs of residualObs) {
            siJR.fill(0);
            // Current residual
            const Ti = siGetT(obs.idxI);
            const di = Ti[6]*obs.xi + Ti[7]*obs.yi + Ti[8];
            if (Math.abs(di) < 1e-10) continue;
            const pix = (Ti[0]*obs.xi + Ti[1]*obs.yi + Ti[2]) / di;
            const piy = (Ti[3]*obs.xi + Ti[4]*obs.yi + Ti[5]) / di;
            const Tj = siGetT(obs.idxJ);
            const dj = Tj[6]*obs.xj + Tj[7]*obs.yj + Tj[8];
            if (Math.abs(dj) < 1e-10) continue;
            const pjx = (Tj[0]*obs.xj + Tj[1]*obs.yj + Tj[2]) / dj;
            const pjy = (Tj[3]*obs.xj + Tj[4]*obs.yj + Tj[5]) / dj;
            const rx = pix-pjx, ry = piy-pjy, r2 = rx*rx+ry*ry;
            let w = 1.0;
            if (r2 > huberDelta*huberDelta) w = huberDelta / Math.sqrt(r2);

            // ∂r/∂f — perturb shared focal length (affects both images)
            {
              const old = siP[0];
              siP[0] = old + siEps;
              const Tip = siGetT(obs.idxI), Tjp = siGetT(obs.idxJ);
              siP[0] = old - siEps;
              const Tim = siGetT(obs.idxI), Tjm = siGetT(obs.idxJ);
              siP[0] = old;
              const dip = Tip[6]*obs.xi+Tip[7]*obs.yi+Tip[8];
              const dim = Tim[6]*obs.xi+Tim[7]*obs.yi+Tim[8];
              const djp = Tjp[6]*obs.xj+Tjp[7]*obs.yj+Tjp[8];
              const djm = Tjm[6]*obs.xj+Tjm[7]*obs.yj+Tjm[8];
              if (Math.abs(dip)>1e-10 && Math.abs(dim)>1e-10 &&
                  Math.abs(djp)>1e-10 && Math.abs(djm)>1e-10) {
                const rpx = (Tip[0]*obs.xi+Tip[1]*obs.yi+Tip[2])/dip - (Tjp[0]*obs.xj+Tjp[1]*obs.yj+Tjp[2])/djp;
                const rpy = (Tip[3]*obs.xi+Tip[4]*obs.yi+Tip[5])/dip - (Tjp[3]*obs.xj+Tjp[4]*obs.yj+Tjp[5])/djp;
                const rmx = (Tim[0]*obs.xi+Tim[1]*obs.yi+Tim[2])/dim - (Tjm[0]*obs.xj+Tjm[1]*obs.yj+Tjm[2])/djm;
                const rmy = (Tim[3]*obs.xi+Tim[4]*obs.yi+Tim[5])/dim - (Tjm[3]*obs.xj+Tjm[4]*obs.yj+Tjm[5])/djm;
                siJR[0] = (rpx - rmx) / (2*siEps);
                siJR[siPC] = (rpy - rmy) / (2*siEps);
              }
            }

            // ∂r/∂r_i — rotation params of image I
            if (obs.idxI !== refIdx) {
              const o = siOff[obs.idxI];
              for (let p = 0; p < 3; p++) {
                const old = siP[o+p];
                siP[o+p] = old + siEps;
                const Tp = siGetT(obs.idxI);
                siP[o+p] = old - siEps;
                const Tm = siGetT(obs.idxI);
                siP[o+p] = old;
                const dpp = Tp[6]*obs.xi+Tp[7]*obs.yi+Tp[8];
                const dpm = Tm[6]*obs.xi+Tm[7]*obs.yi+Tm[8];
                if (Math.abs(dpp)>1e-10 && Math.abs(dpm)>1e-10) {
                  siJR[o+p] = ((Tp[0]*obs.xi+Tp[1]*obs.yi+Tp[2])/dpp - (Tm[0]*obs.xi+Tm[1]*obs.yi+Tm[2])/dpm) / (2*siEps);
                  siJR[siPC+o+p] = ((Tp[3]*obs.xi+Tp[4]*obs.yi+Tp[5])/dpp - (Tm[3]*obs.xi+Tm[4]*obs.yi+Tm[5])/dpm) / (2*siEps);
                }
              }
            }

            // ∂r/∂r_j — rotation params of image J (negated: residual = pi − pj)
            if (obs.idxJ !== refIdx) {
              const o = siOff[obs.idxJ];
              for (let p = 0; p < 3; p++) {
                const old = siP[o+p];
                siP[o+p] = old + siEps;
                const Tp = siGetT(obs.idxJ);
                siP[o+p] = old - siEps;
                const Tm = siGetT(obs.idxJ);
                siP[o+p] = old;
                const dpp = Tp[6]*obs.xj+Tp[7]*obs.yj+Tp[8];
                const dpm = Tm[6]*obs.xj+Tm[7]*obs.yj+Tm[8];
                if (Math.abs(dpp)>1e-10 && Math.abs(dpm)>1e-10) {
                  siJR[o+p] -= ((Tp[0]*obs.xj+Tp[1]*obs.yj+Tp[2])/dpp - (Tm[0]*obs.xj+Tm[1]*obs.yj+Tm[2])/dpm) / (2*siEps);
                  siJR[siPC+o+p] -= ((Tp[3]*obs.xj+Tp[4]*obs.yj+Tp[5])/dpp - (Tm[3]*obs.xj+Tm[4]*obs.yj+Tm[5])/dpm) / (2*siEps);
                }
              }
            }

            // Accumulate J^T J and J^T r
            for (let rr = 0; rr < 2; rr++) {
              const res = rr === 0 ? rx : ry;
              const ro = rr * siPC;
              for (let i = 0; i < siPC; i++) {
                const ji = siJR[ro + i];
                if (Math.abs(ji) < 1e-15) continue;
                siJtr[i] += w * ji * res;
                for (let j = i; j < siPC; j++) {
                  const jj = siJR[ro + j];
                  if (Math.abs(jj) < 1e-15) continue;
                  const v = w * ji * jj;
                  siJtJ[i*siPC+j] += v;
                  if (i !== j) siJtJ[j*siPC+i] += v;
                }
              }
            }
          }

          // Damping
          for (let i = 0; i < siPC; i++) {
            siJtJ[i*siPC+i] *= (1 + siLambda);
            if (Math.abs(siJtJ[i*siPC+i]) < 1e-12) siJtJ[i*siPC+i] = siLambda;
          }
          const negJ = new Float64Array(siPC);
          for (let i = 0; i < siPC; i++) negJ[i] = -siJtr[i];
          const delta = solveLinearN(siJtJ, negJ, siPC);
          if (!delta) { siLambda *= 10; continue; }
          let bad = false;
          for (let i = 0; i < delta.length; i++) if (isNaN(delta[i])) { bad = true; break; }
          if (bad) { siLambda *= 10; continue; }

          const snap = new Float64Array(siP);
          for (let i = 0; i < siPC; i++) siP[i] += delta[i];
          if (siP[0] < 10) siP[0] = 10; // clamp focal length

          const nc = siCost();
          if (nc < siPrev) {
            siLambda = Math.max(siLambda * 0.3, 1e-7);
            siPrev = nc;
            let mx = 0;
            for (let i = 0; i < siPC; i++) if (Math.abs(delta[i]) > mx) mx = Math.abs(delta[i]);
            if (mx < 1e-8) break;
          } else {
            for (let i = 0; i < siPC; i++) siP[i] = snap[i];
            siLambda = Math.min(siLambda * 10, 1e8);
          }
          postMessage({type:'progress', stage:'refine', percent: Math.round(100*(iter+1)/maxIters),
            info:`shared-intrinsics iter ${iter+1}/${maxIters}, f=${siP[0].toFixed(1)}px, cost=${siPrev.toFixed(2)}`});
        }

        // Reconstruct transforms from optimised params
        for (let i = 0; i < nImages; i++) {
          if (i === refIdx) continue;
          transforms[ids[i]] = new Float64Array(siGetT(i));
        }
        const tList = Object.entries(transforms).map(([id, T]) => ({
          imageId: id, TBuffer: new Float64Array(T).buffer
        }));
        postMessage({type:'transforms', refId, transforms: tList});
        postMessage({type:'progress', stage:'refine', percent:100,
          info:`shared-intrinsics LM done, f=${siP[0].toFixed(1)}px, cost=${siPrev.toFixed(2)}`});
        return;
      }

      // Flatten current transforms into parameter vector
      // Each image except ref has 8 params: H[0..7] (H[8] always 1)
      const paramCount = (nImages - 1) * 8;
      const params = new Float64Array(paramCount);
      const imgParamOffset = new Int32Array(nImages); // offset into params, -1 for ref
      let off = 0;
      for (let i = 0; i < nImages; i++) {
        if (i === refIdx) {
          imgParamOffset[i] = -1;
          continue;
        }
        imgParamOffset[i] = off;
        const T = transforms[ids[i]];
        for (let k = 0; k < 8; k++) params[off + k] = T[k];
        off += 8;
      }

      // Helper: reconstruct transform from params (pre-allocated buffer)
      const _Tbuf = new Float64Array(9);
      const _TrefBuf = new Float64Array([1, 0, 0, 0, 1, 0, 0, 0, 1]);
      function getTransform(imgIdx) {
        if (imgIdx === refIdx) return _TrefBuf;
        const o = imgParamOffset[imgIdx];
        for (let k = 0; k < 8; k++) _Tbuf[k] = params[o + k];
        _Tbuf[8] = 1.0;
        return _Tbuf;
      }
      // Note: getTransform returns a shared buffer — callers must consume before next call.

      // Compute residuals and cost
      function computeCost() {
        let cost = 0;
        for (const obs of residualObs) {
          const Ti = getTransform(obs.idxI);
          // Copy Ti since getTransform returns shared buffer
          const pi_x_num = Ti[0] * obs.xi + Ti[1] * obs.yi + Ti[2];
          const pi_y_num = Ti[3] * obs.xi + Ti[4] * obs.yi + Ti[5];
          const pi_den = Ti[6] * obs.xi + Ti[7] * obs.yi + Ti[8];
          if (Math.abs(pi_den) < 1e-10) { cost += huberDelta * huberDelta; continue; }
          const pix = pi_x_num / pi_den;
          const piy = pi_y_num / pi_den;

          const Tj = getTransform(obs.idxJ);
          const pj_den = Tj[6] * obs.xj + Tj[7] * obs.yj + Tj[8];
          if (Math.abs(pj_den) < 1e-10) { cost += huberDelta * huberDelta; continue; }
          const pjx = (Tj[0] * obs.xj + Tj[1] * obs.yj + Tj[2]) / pj_den;
          const pjy = (Tj[3] * obs.xj + Tj[4] * obs.yj + Tj[5]) / pj_den;

          const dx = pix - pjx;
          const dy = piy - pjy;
          const r2 = dx * dx + dy * dy;
          // Huber loss
          if (r2 <= huberDelta * huberDelta) {
            cost += r2;
          } else {
            cost += 2 * huberDelta * Math.sqrt(r2) - huberDelta * huberDelta;
          }
        }
        return cost;
      }

      // Numeric Jacobian computation via central differences
      const eps = 1e-6;
      // LM damping parameter λ: balances between gradient descent (high λ)
      // and Gauss-Newton (low λ). λ_init = 0.01 is a typical starting value.
      // On accept: λ *= 0.3 (move toward Gauss-Newton), floor 1e-7.
      // On reject: λ *= 10 (more gradient descent), cap 1e8.
      let lambda = lambdaInit;
      let prevCost = computeCost();

      // Pre-allocate LM buffers outside the iteration loop to avoid GC churn
      const JtJ = new Float64Array(paramCount * paramCount);
      const Jtr = new Float64Array(paramCount);
      const jacobRow = new Float64Array(paramCount * 2);

      for (let iter = 0; iter < maxIters; iter++) {
        // Build J^T J and J^T r using numeric Jacobian
        JtJ.fill(0);
        Jtr.fill(0);
        jacobRow.fill(0);

        for (let obs_k = 0; obs_k < residualObs.length; obs_k++) {
          const obs = residualObs[obs_k];

          // Compute residuals inline (getTransform/project use shared buffers)
          let Ti = getTransform(obs.idxI);
          const pi_den = Ti[6] * obs.xi + Ti[7] * obs.yi + Ti[8];
          if (Math.abs(pi_den) < 1e-10) continue;
          const pix = (Ti[0] * obs.xi + Ti[1] * obs.yi + Ti[2]) / pi_den;
          const piy = (Ti[3] * obs.xi + Ti[4] * obs.yi + Ti[5]) / pi_den;

          const Tj = getTransform(obs.idxJ);
          const pj_den = Tj[6] * obs.xj + Tj[7] * obs.yj + Tj[8];
          if (Math.abs(pj_den) < 1e-10) continue;
          const pjx = (Tj[0] * obs.xj + Tj[1] * obs.yj + Tj[2]) / pj_den;
          const pjy = (Tj[3] * obs.xj + Tj[4] * obs.yj + Tj[5]) / pj_den;

          const rx = pix - pjx;
          const ry = piy - pjy;
          const r2 = rx * rx + ry * ry;

          // Huber weight
          let w = 1.0;
          if (r2 > huberDelta * huberDelta) {
            w = huberDelta / Math.sqrt(r2);
          }

          // Clear jacobian row
          jacobRow.fill(0);

          // Derivatives w.r.t. image I params (central differences)
          if (obs.idxI !== refIdx) {
            const o = imgParamOffset[obs.idxI];
            for (let p = 0; p < 8; p++) {
              const old = params[o + p];
              params[o + p] = old + eps;
              Ti = getTransform(obs.idxI);
              const d1 = Ti[6] * obs.xi + Ti[7] * obs.yi + Ti[8];
              let p1x = 0, p1y = 0, ok1 = Math.abs(d1) > 1e-10;
              if (ok1) { p1x = (Ti[0] * obs.xi + Ti[1] * obs.yi + Ti[2]) / d1; p1y = (Ti[3] * obs.xi + Ti[4] * obs.yi + Ti[5]) / d1; }

              params[o + p] = old - eps;
              Ti = getTransform(obs.idxI);
              const d2 = Ti[6] * obs.xi + Ti[7] * obs.yi + Ti[8];
              let p2x = 0, p2y = 0, ok2 = Math.abs(d2) > 1e-10;
              if (ok2) { p2x = (Ti[0] * obs.xi + Ti[1] * obs.yi + Ti[2]) / d2; p2y = (Ti[3] * obs.xi + Ti[4] * obs.yi + Ti[5]) / d2; }

              params[o + p] = old;
              if (ok1 && ok2) {
                jacobRow[o + p] = (p1x - p2x) / (2 * eps);
                jacobRow[paramCount + o + p] = (p1y - p2y) / (2 * eps);
              }
            }
          }
          // Derivatives w.r.t. image J params
          if (obs.idxJ !== refIdx) {
            const o = imgParamOffset[obs.idxJ];
            for (let p = 0; p < 8; p++) {
              const old = params[o + p];
              params[o + p] = old + eps;
              let Tj2 = getTransform(obs.idxJ);
              const d1 = Tj2[6] * obs.xj + Tj2[7] * obs.yj + Tj2[8];
              let p1x = 0, p1y = 0, ok1 = Math.abs(d1) > 1e-10;
              if (ok1) { p1x = (Tj2[0] * obs.xj + Tj2[1] * obs.yj + Tj2[2]) / d1; p1y = (Tj2[3] * obs.xj + Tj2[4] * obs.yj + Tj2[5]) / d1; }

              params[o + p] = old - eps;
              Tj2 = getTransform(obs.idxJ);
              const d2 = Tj2[6] * obs.xj + Tj2[7] * obs.yj + Tj2[8];
              let p2x = 0, p2y = 0, ok2 = Math.abs(d2) > 1e-10;
              if (ok2) { p2x = (Tj2[0] * obs.xj + Tj2[1] * obs.yj + Tj2[2]) / d2; p2y = (Tj2[3] * obs.xj + Tj2[4] * obs.yj + Tj2[5]) / d2; }

              params[o + p] = old;
              if (ok1 && ok2) {
                // Residual = pi - pj, so d/dpj = -dpj/dp
                jacobRow[o + p] = -(p1x - p2x) / (2 * eps);
                jacobRow[paramCount + o + p] = -(p1y - p2y) / (2 * eps);
              }
            }
          }

          // Accumulate JtJ and Jtr with Huber weighting
          for (let rr = 0; rr < 2; rr++) {
            const res_rr = rr === 0 ? rx : ry;
            const rowOff = rr * paramCount;
            for (let i = 0; i < paramCount; i++) {
              const ji = jacobRow[rowOff + i];
              if (Math.abs(ji) < 1e-15) continue;
              Jtr[i] += w * ji * res_rr;
              for (let j = i; j < paramCount; j++) {
                const jj = jacobRow[rowOff + j];
                if (Math.abs(jj) < 1e-15) continue;
                const val = w * ji * jj;
                JtJ[i * paramCount + j] += val;
                if (i !== j) JtJ[j * paramCount + i] += val;
              }
            }
          }
        }

        // Damping: JtJ += lambda * diag(JtJ)
        for (let i = 0; i < paramCount; i++) {
          JtJ[i * paramCount + i] *= (1 + lambda);
          // Ensure diagonal isn't zero
          if (Math.abs(JtJ[i * paramCount + i]) < 1e-12) {
            JtJ[i * paramCount + i] = lambda;
          }
        }

        // Solve (JtJ) * delta = -Jtr
        const negJtr = new Float64Array(paramCount);
        for (let i = 0; i < paramCount; i++) negJtr[i] = -Jtr[i];
        const delta = solveLinearN(JtJ, negJtr, paramCount);

        if (!delta) {
          // Singular — increase damping
          lambda *= 10;
          continue;
        }

        // Check for NaN in delta (degenerate numeric solve)
        let hasNaN = false;
        for (let i = 0; i < delta.length; i++) { if (isNaN(delta[i])) { hasNaN = true; break; } }
        if (hasNaN) {
          lambda *= 10;
          continue;
        }

        // Trial step
        const oldParams = new Float64Array(params);
        for (let i = 0; i < paramCount; i++) {
          params[i] += delta[i];
        }

        const newCost = computeCost();
        if (newCost < prevCost) {
          // Accept step, decrease lambda
          lambda = Math.max(lambda * 0.3, 1e-7);
          prevCost = newCost;

          // Early termination if accepted delta is tiny
          let maxDelta = 0;
          for (let i = 0; i < paramCount; i++) {
            if (Math.abs(delta[i]) > maxDelta) maxDelta = Math.abs(delta[i]);
          }
          if (maxDelta < 1e-8) break;
        } else {
          // Reject step, increase lambda
          for (let i = 0; i < paramCount; i++) params[i] = oldParams[i];
          lambda = Math.min(lambda * 10, 1e8);
        }

        postMessage({type:'progress', stage:'refine', percent: Math.round(100 * (iter + 1) / maxIters), info: `iter ${iter+1}/${maxIters}, cost=${prevCost.toFixed(2)}, λ=${lambda.toExponential(1)}`});
      }

      // Write back refined parameters to transforms
      for (let i = 0; i < nImages; i++) {
        if (i === refIdx) continue;
        const T = new Float64Array(9);
        const o = imgParamOffset[i];
        for (let k = 0; k < 8; k++) T[k] = params[o + k];
        T[8] = 1.0;
        transforms[ids[i]] = T;
      }

      const tList = Object.entries(transforms).map(([id, T]) => ({
        imageId: id,
        TBuffer: new Float64Array(T).buffer
      }));
      postMessage({type:'transforms', refId, transforms: tList});
      postMessage({type:'progress', stage:'refine', percent:100, info:`LM complete, final cost=${prevCost.toFixed(2)}`});
      return;
    }

    // ── Post-BA quality assessment ─────────────────────────────────────
    // After bundle adjustment, compute per-image alignment quality metrics.
    // Detects:
    //  1. High mean reprojection error → badly aligned image
    //  2. Few functional edges (after BA most inliers became outliers)
    //  3. Extreme perspective (large perspective terms in global transform)
    // Returns per-image quality scores and a list of images recommended
    // for exclusion. The pipeline controller can then re-run BA without them.
    if (msg.type === 'qualityAssessment') {
      const ids = Object.keys(images);
      const threshold = msg.threshold || 5.0; // px — exclude if mean reproj > threshold
      const perImageErrors = {};
      const perImageEdgeCount = {};

      for (const id of ids) {
        perImageErrors[id] = [];
        perImageEdgeCount[id] = 0;
      }

      // Compute per-image reprojection errors from ALL edges
      for (const e of edges) {
        const Ti = transforms[e.i];
        const Tj = transforms[e.j];
        if (!Ti || !Tj) continue;

        let edgeErrors_i = [];
        let edgeErrors_j = [];

        for (const inl of e.inliers) {
          // Project both points to global
          const di = Ti[6] * inl.xi + Ti[7] * inl.yi + Ti[8];
          const dj = Tj[6] * inl.xj + Tj[7] * inl.yj + Tj[8];
          if (Math.abs(di) < 1e-10 || Math.abs(dj) < 1e-10) continue;
          const gix = (Ti[0] * inl.xi + Ti[1] * inl.yi + Ti[2]) / di;
          const giy = (Ti[3] * inl.xi + Ti[4] * inl.yi + Ti[5]) / di;
          const gjx = (Tj[0] * inl.xj + Tj[1] * inl.yj + Tj[2]) / dj;
          const gjy = (Tj[3] * inl.xj + Tj[4] * inl.yj + Tj[5]) / dj;
          const err = Math.sqrt((gix - gjx) ** 2 + (giy - gjy) ** 2);
          edgeErrors_i.push(err);
          edgeErrors_j.push(err);
        }

        if (edgeErrors_i.length > 0) {
          perImageErrors[e.i].push(...edgeErrors_i);
          perImageErrors[e.j].push(...edgeErrors_j);
          perImageEdgeCount[e.i]++;
          perImageEdgeCount[e.j]++;
        }
      }

      // Compute per-image quality metrics
      const quality = [];
      const allMedianErrors = [];

      for (const id of ids) {
        const errs = perImageErrors[id];
        if (errs.length === 0) {
          quality.push({ imageId: id, meanError: Infinity, medianError: Infinity,
                        edgeCount: 0, perspectiveMag: 0, isOutlier: true, reason: 'no edges' });
          continue;
        }

        errs.sort((a, b) => a - b);
        const mean = errs.reduce((s, v) => s + v, 0) / errs.length;
        const median = errs[Math.floor(errs.length / 2)];

        // Perspective magnitude: ||[T[6], T[7]]|| — large values indicate
        // extreme perspective that might cause artifacts
        const T = transforms[id];
        const perspMag = T ? Math.sqrt(T[6] * T[6] + T[7] * T[7]) : 0;

        quality.push({
          imageId: id,
          meanError: mean,
          medianError: median,
          edgeCount: perImageEdgeCount[id],
          perspectiveMag: perspMag,
          isOutlier: false,
          reason: ''
        });
        allMedianErrors.push(median);
      }

      // Compute global median error for relative threshold
      allMedianErrors.sort((a, b) => a - b);
      const globalMedian = allMedianErrors.length > 0
        ? allMedianErrors[Math.floor(allMedianErrors.length / 2)] : 0;

      // Adaptive threshold: max(absolute threshold, 3× global median)
      // This handles both tight-alignment cases (low median) and looser cases
      const adaptiveThreshold = Math.max(threshold, globalMedian * 3);

      // Mark outliers
      const excludeIds = [];
      for (const q of quality) {
        if (q.imageId === refId) continue; // never exclude reference
        if (q.meanError > adaptiveThreshold) {
          q.isOutlier = true;
          q.reason = `mean reproj ${q.meanError.toFixed(2)}px > ${adaptiveThreshold.toFixed(1)}px`;
          excludeIds.push(q.imageId);
        } else if (q.perspectiveMag > 0.01) {
          // PTGui-style: flag extreme perspective (vanishing line near image)
          q.isOutlier = true;
          q.reason = `extreme perspective (${q.perspectiveMag.toExponential(2)})`;
          excludeIds.push(q.imageId);
        }
      }

      postMessage({
        type: 'qualityAssessment',
        quality,
        excludeIds,
        globalMedianError: globalMedian,
        adaptiveThreshold
      });
      return;
    }

    if (msg.type === 'computeExposure') {
      // ── Per-channel RGB gain compensation ──────────────────────────
      // Instead of a single scalar gain, solve for independent R, G, B
      // multiplicative gains per image: g_i = [gR, gG, gB]. This corrects
      // both exposure differences AND white-balance shifts between images.
      // For grayscale-only images we fall back to scalar gain.
      //
      // For each edge (i,j), at each inlier correspondence we sample the
      // grayscale luminance and solve: log(g_j) − log(g_i) ≈ r_ij
      // in a global least-squares system with a reference image fixed at 1.
      //
      // If RGB small buffers are available (rgbSmall), we solve 3 independent
      // systems for R, G, B channels. Otherwise, we use the grayscale channel
      // for a single gain and replicate it across channels.
      //
      // When sameCameraSettings is true, we use much stronger regularization
      // toward gain=1 (same aperture/ISO/shutter → expect near-equal exposure).
      const sameCam = msg.sameCameraSettings || false;
      const ids = Object.keys(images);
      const n = ids.length;

      if (n < 2 || edges.length === 0) {
        const gains = ids.map(id => ({ imageId: id, gain: 1.0, gainR: 1.0, gainG: 1.0, gainB: 1.0 }));
        postMessage({ type: 'exposure', gains });
        return;
      }

      const idToIdx = {};
      ids.forEach((id, i) => idToIdx[id] = i);

      // Check if any image has RGB data; if not, fall back to scalar
      const hasRGB = Object.values(images).some(img => img.rgbSmall);

      // For each edge, compute per-channel log ratios from inlier correspondences
      const edgeRatios = []; // {i, j, ratioGray, ratioR?, ratioG?, ratioB?}

      for (const e of edges) {
        const imgI = images[e.i];
        const imgJ = images[e.j];
        if (!imgI || !imgJ || !imgI.gray || !imgJ.gray) continue;

        let sumLogI = 0, sumLogJ = 0, count = 0;
        let sumLogR_I = 0, sumLogR_J = 0;
        let sumLogG_I = 0, sumLogG_J = 0;
        let sumLogB_I = 0, sumLogB_J = 0;
        let rgbCount = 0;
        // Reusable output buffers for sampleRGB to avoid per-call allocation
        const _rgbBufI = [0, 0, 0];
        const _rgbBufJ = [0, 0, 0];

        for (const inl of e.inliers) {
          const xi = inl.xi, yi = inl.yi;
          const xj = inl.xj, yj = inl.yj;

          const lumI = sampleGray(imgI.gray, imgI.width, imgI.height, xi, yi);
          const lumJ = sampleGray(imgJ.gray, imgJ.width, imgJ.height, xj, yj);

          if (lumI > 5 && lumJ > 5) {
            sumLogI += Math.log(lumI);
            sumLogJ += Math.log(lumJ);
            count++;
          }

          // Sample RGB if available (rgbSmall is a Uint8Array of w*h*3)
          if (imgI.rgbSmall && imgJ.rgbSmall) {
            const rgbI = sampleRGB(imgI.rgbSmall, imgI.width, imgI.height, xi, yi, _rgbBufI);
            const rgbJ = sampleRGB(imgJ.rgbSmall, imgJ.width, imgJ.height, xj, yj, _rgbBufJ);
            if (rgbI && rgbJ && rgbI[0] > 5 && rgbJ[0] > 5 &&
                rgbI[1] > 5 && rgbJ[1] > 5 && rgbI[2] > 5 && rgbJ[2] > 5) {
              sumLogR_I += Math.log(rgbI[0]); sumLogR_J += Math.log(rgbJ[0]);
              sumLogG_I += Math.log(rgbI[1]); sumLogG_J += Math.log(rgbJ[1]);
              sumLogB_I += Math.log(rgbI[2]); sumLogB_J += Math.log(rgbJ[2]);
              rgbCount++;
            }
          }
        }

        if (count >= 5) {
          const entry = {
            i: idToIdx[e.i],
            j: idToIdx[e.j],
            ratioGray: (sumLogJ / count) - (sumLogI / count)
          };
          if (rgbCount >= 5) {
            entry.ratioR = (sumLogR_J / rgbCount) - (sumLogR_I / rgbCount);
            entry.ratioG = (sumLogG_J / rgbCount) - (sumLogG_I / rgbCount);
            entry.ratioB = (sumLogB_J / rgbCount) - (sumLogB_I / rgbCount);
          }
          edgeRatios.push(entry);
        }
      }

      if (edgeRatios.length === 0) {
        const gains = ids.map(id => ({ imageId: id, gain: 1.0, gainR: 1.0, gainG: 1.0, gainB: 1.0 }));
        postMessage({ type: 'exposure', gains });
        return;
      }

      // Validate refIdx: ensure reference image exists in the index map
      const refIdx = (refId && idToIdx[refId] !== undefined) ? idToIdx[refId] : 0;

      // Robust IRLS (Iteratively Reweighted Least Squares) gain solver.
      // Uses Huber loss to down-weight extreme exposure ratios from outlier
      // correspondences or saturated pixels. This is critical for Brenizer
      // mosaics with wide aperture where bokeh boundaries create unreliable
      // matches. Falls back to standard least squares when IRLS fails.
      // Huber threshold δ = 0.5 log-units (corresponds to ~1.65× gain factor).
      const HUBER_DELTA = 0.5;
      const IRLS_ITERS = 5;

      function solveGainSystem(ratioKey) {
        const irlsWeights = new Float64Array(edgeRatios.length);
        irlsWeights.fill(1.0);
        let solution = null;

        for (let iter = 0; iter < IRLS_ITERS; iter++) {
          const AtA = new Float64Array(n * n);
          const Atb = new Float64Array(n);
          let hasData = false;

          for (let ei = 0; ei < edgeRatios.length; ei++) {
            const er = edgeRatios[ei];
            const r = er[ratioKey];
            if (r === undefined) continue;
            hasData = true;
            const w = irlsWeights[ei];
            AtA[er.i * n + er.i] += w;
            AtA[er.j * n + er.j] += w;
            AtA[er.i * n + er.j] -= w;
            AtA[er.j * n + er.i] -= w;
            Atb[er.i] -= w * r;
            Atb[er.j] += w * r;
          }

          if (!hasData) return null;

          // Fix reference (strong prior)
          AtA[refIdx * n + refIdx] += 1000;
          // Regularization toward gain=1  (much stronger when same camera settings)
          const regWeight = sameCam ? 1.0 : 0.01;
          for (let i = 0; i < n; i++) AtA[i * n + i] += regWeight;

          const logGains = solveLinearN(AtA, Atb, n);
          if (!logGains) return solution; // return last good solution
          solution = logGains;

          // Update IRLS weights using Huber loss
          for (let ei = 0; ei < edgeRatios.length; ei++) {
            const er = edgeRatios[ei];
            const r = er[ratioKey];
            if (r === undefined) continue;
            const residual = Math.abs((logGains[er.j] - logGains[er.i]) - r);
            // Huber weight: w = 1 if |r| ≤ δ, else δ/|r|
            irlsWeights[ei] = residual <= HUBER_DELTA ? 1.0 : HUBER_DELTA / residual;
          }
        }

        if (!solution) return null;
        // Clamp gains to reasonable range [0.05, 20] for extreme exposure support
        return solution.map(v => {
          const g = Math.exp(v);
          return (Number.isFinite(g) && g > 0) ? Math.min(20, Math.max(0.05, g)) : 1.0;
        });
      }

      const grayGains = solveGainSystem('ratioGray');
      const rGains = solveGainSystem('ratioR');
      const gGains = solveGainSystem('ratioG');
      const bGains = solveGainSystem('ratioB');

      // Helper: safely extract a gain value with NaN protection
      function safeGain(arr, idx, fallback) {
        if (!arr) return fallback;
        const v = arr[idx];
        return (Number.isFinite(v) && v > 0) ? v : fallback;
      }

      const gains = ids.map((id, i) => {
        const g = safeGain(grayGains, i, 1.0);
        return {
          imageId: id,
          gain: g,
          gainR: safeGain(rGains, i, g),
          gainG: safeGain(gGains, i, g),
          gainB: safeGain(bGains, i, g),
        };
      });

      // When same-camera, check if all gains are negligible (within ±5%)
      // If so, snap them to exactly 1.0 to avoid unnecessary pixel work.
      if (sameCam) {
        const allNear1 = gains.every(g =>
          Math.abs(g.gainR - 1) < 0.05 &&
          Math.abs(g.gainG - 1) < 0.05 &&
          Math.abs(g.gainB - 1) < 0.05
        );
        if (allNear1) {
          for (const g of gains) {
            g.gain = 1.0; g.gainR = 1.0; g.gainG = 1.0; g.gainB = 1.0;
          }
          postMessage({type:'progress', stage:'exposure', percent:100, info:'same-camera: gains negligible, snapped to 1.0'});
        }
      }

      postMessage({ type: 'exposure', gains });

      // Free per-image RGB buffers — no longer needed after exposure computation
      for (const id of ids) {
        if (images[id]) images[id].rgbSmall = null;
      }
      return;
    }

    if (msg.type === 'computeLocalMesh') {
      const { imageId, parentId, meshGrid, sigma, depthSigma, minSupport, faceRects } = msg;
      const sameCam = msg.sameCameraSettings || false;
      const G = meshGrid || 4;
      const faces = faceRects || [];
      const img = images[imageId];
      if (!img) {
        postMessage({type:'error', message:`Image ${imageId} not found`});
        return;
      }

      // Global transform for this image
      const Ti = transforms[imageId];

      if (!Ti) {
        // Fallback: return identity mesh warped by global transform
        const result = buildGlobalMesh(imageId, G, Ti, img);
        postMessage({type:'mesh', imageId, ...result});
        return;
      }

      // ── Multi-edge APAP ──────────────────────────────────────────
      // Collect correspondences from ALL edges involving this image, not
      // just the parent edge. An image may overlap with 4–5 neighbours but
      // the MST only connects it to one parent — using only that edge
      // wastes 60–80% of available data. For each edge, project the
      // partner-side inlier points to global coords via that partner's
      // refined global transform. This gives a much richer set of
      // constraints for the weighted DLT.
      //
      // Reference: Zaragoza et al., "As-Projective-As-Possible Image
      // Stitching", CVPR 2013 — the original paper uses all available
      // correspondences for each mesh vertex, not just pairwise.
      const srcPts = []; // [x, y] in this image's coords
      const dstPts = []; // [X, Y] in global coords

      for (const edge of edges) {
        let isI = false, isJ = false;
        if (edge.i === imageId) isI = true;
        else if (edge.j === imageId) isJ = true;
        else continue; // edge doesn't involve this image

        // Get the partner image's global transform
        const partnerId = isI ? edge.j : edge.i;
        const Tp = transforms[partnerId];
        if (!Tp) continue; // partner not yet transformed

        for (const inl of edge.inliers) {
          // Extract this image's point and partner's point
          let si_x, si_y, sp_x, sp_y;
          if (isI) {
            si_x = inl.xi; si_y = inl.yi;
            sp_x = inl.xj; sp_y = inl.yj;
          } else {
            si_x = inl.xj; si_y = inl.yj;
            sp_x = inl.xi; sp_y = inl.yi;
          }

          // Project partner point to global using Tp
          const denom = Tp[6] * sp_x + Tp[7] * sp_y + Tp[8];
          if (Math.abs(denom) < 1e-10) continue;
          const gx = (Tp[0] * sp_x + Tp[1] * sp_y + Tp[2]) / denom;
          const gy = (Tp[3] * sp_x + Tp[4] * sp_y + Tp[5]) / denom;

          srcPts.push([si_x, si_y]);
          dstPts.push([gx, gy]);
        }
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

      // Compute affine fallback for vertices that can't be projected via the
      // full perspective transform (i.e. vertices near or behind the camera).
      // Use the affine part of T: gx = (T[0]*x + T[1]*y + T[2]) / T[8]
      // This ignores perspective but gives a reasonable position.
      function affineProject(T, px, py) {
        const s = Math.abs(T[8]) > 1e-10 ? T[8] : 1;
        return [(T[0] * px + T[1] * py + T[2]) / s, (T[3] * px + T[4] * py + T[5]) / s];
      }

      let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;

      // Maximum reasonable global coordinate: images shouldn't project
      // far beyond a few image-widths from the origin.
      // Defined once outside the vertex loop for efficiency.
      const maxGlobalExtent = Math.max(w, h) * 5;

      // Maximum acceptable deviation of local APAP warp from global transform
      const maxDeviation = Math.max(w, h) * 0.5;

      // Helper: project vertex through a homography with safety checks.
      // Returns [gx, gy] or null if behind camera or out of range.
      function projectVertex(H, px, py) {
        const denom = H[6] * px + H[7] * py + H[8];
        if (denom < 1e-4) return null; // behind camera or near-singular
        const gx = (H[0] * px + H[1] * py + H[2]) / denom;
        const gy = (H[3] * px + H[4] * py + H[5]) / denom;
        // Reject if projected too far (strong perspective artifact)
        if (Math.abs(gx) > maxGlobalExtent || Math.abs(gy) > maxGlobalExtent) return null;
        return [gx, gy];
      }

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

            // Face-aware weight boost: if this correspondence is inside or near a face,
            // triple the weight so the local warp better preserves faces
            for (const face of faces) {
              const fCx = face.x + face.width * 0.5;
              const fCy = face.y + face.height * 0.5;
              const fRadius = Math.max(face.width, face.height) * 0.75;
              const fdx = srcPts[k][0] - fCx;
              const fdy = srcPts[k][1] - fCy;
              if (fdx * fdx + fdy * fdy < fRadius * fRadius) {
                ws *= 3.0;
                break; // only boost once per correspondence
              }
            }

            weights[k] = ws;
            totalWeight += ws;
          }

          // Check effective support
          const effectiveSupport = totalWeight;
          let gx, gy;

          // Compute global transform prediction as a reference for sanity checking
          const globalPt = projectVertex(Ti, vx, vy);
          // If perspective projection fails (vertex behind camera), use affine approx
          let globalGx, globalGy;
          if (globalPt) {
            globalGx = globalPt[0];
            globalGy = globalPt[1];
          } else {
            const affinePt = affineProject(Ti, vx, vy);
            globalGx = affinePt[0];
            globalGy = affinePt[1];
          }

          if (effectiveSupport < minSupport || srcPts.length < 4) {
            // Fallback to global transform
            gx = globalGx;
            gy = globalGy;
          } else {
            // Tikhonov-regularized weighted DLT: pull toward global homography
            // where data is sparse. Adaptive γ: stronger regularization when
            // effective support is weak (far from correspondences).
            // When sameCameraSettings: shared lens → same distortion → local
            // warps should deviate less from the global model.  Triple the
            // base gamma to enforce this.
            const supportRatio = Math.min(effectiveSupport / (minSupport * 10), 1.0);
            const baseGamma = sameCam ? 0.3 : 0.1;
            const adaptiveGamma = baseGamma * (1.0 - supportRatio);
            const Hv = weightedDLT(srcPts, dstPts, weights, Ti, adaptiveGamma);
            if (Hv) {
              const localPt = projectVertex(Hv, vx, vy);
              if (localPt) {
                // Validate: local result shouldn't deviate too far from global
                const dx = localPt[0] - globalGx;
                const dy = localPt[1] - globalGy;
                if (dx * dx + dy * dy < maxDeviation * maxDeviation) {
                  gx = localPt[0];
                  gy = localPt[1];
                } else {
                  // DLT result too extreme — fallback to global
                  gx = globalGx;
                  gy = globalGy;
                }
              } else {
                gx = globalGx;
                gy = globalGy;
              }
            } else {
              // DLT failed, fallback
              gx = globalGx;
              gy = globalGy;
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

      // ── Edge trimming via Jacobian area analysis ─────────────────
      // For each quad in the mesh, compute the area ratio between the output
      // (warped) quad and the input (source) quad.  Quads where this ratio
      // deviates significantly from the median indicate extreme distortion
      // (perspective blow-up near edges).  Nudge those boundary vertices
      // inward toward the median-quality interior, effectively trimming
      // the most distorted output edges.
      const quadAreas = [];
      function triArea2(ax,ay, bx,by, cx,cy) {
        return Math.abs((bx-ax)*(cy-ay) - (cx-ax)*(by-ay));
      }
      for (let r = 0; r < G; r++) {
        for (let c = 0; c < G; c++) {
          const tl = r * cols + c;
          const tr = tl + 1;
          const bl = tl + cols;
          const br = bl + 1;
          // Source quad area (uniform grid)
          const su = 1 / G;
          const srcArea = su * su * w * h; // constant for uniform grid
          // Output quad area: sum of two triangles (diagonally split)
          const outArea =
            triArea2(vertices[tl*2], vertices[tl*2+1], vertices[bl*2], vertices[bl*2+1], vertices[tr*2], vertices[tr*2+1]) +
            triArea2(vertices[tr*2], vertices[tr*2+1], vertices[bl*2], vertices[bl*2+1], vertices[br*2], vertices[br*2+1]);
          const ratio = outArea / Math.max(srcArea, 1e-10);
          quadAreas.push(ratio);
        }
      }
      // Median area ratio
      const sortedAreas = quadAreas.slice().sort((a, b) => a - b);
      const medianArea = sortedAreas[Math.floor(sortedAreas.length / 2)];
      // Mark extreme quads (ratio > 4× or < 0.25× median)
      const extremeThresh = 4.0;
      const extremeQuads = new Set();
      for (let q = 0; q < quadAreas.length; q++) {
        if (quadAreas[q] > medianArea * extremeThresh || quadAreas[q] < medianArea / extremeThresh) {
          extremeQuads.add(q);
        }
      }
      // For boundary vertices that belong only to extreme quads, pull them
      // toward the nearest non-extreme neighbour vertex.
      if (extremeQuads.size > 0 && extremeQuads.size < quadAreas.length * 0.5) {
        // Build vertex-to-quad adjacency (only boundary vertices)
        const vertQuads = new Map(); // vertexIndex -> [quadIndex, ...]
        for (let q = 0; q < quadAreas.length; q++) {
          const r = Math.floor(q / G);
          const c = q % G;
          const cornerVerts = [r*cols+c, r*cols+c+1, (r+1)*cols+c, (r+1)*cols+c+1];
          for (const vi of cornerVerts) {
            // Only process boundary vertices (first/last row/column)
            const vr = Math.floor(vi / cols);
            const vc = vi % cols;
            if (vr === 0 || vr === rows-1 || vc === 0 || vc === cols-1) {
              if (!vertQuads.has(vi)) vertQuads.set(vi, []);
              vertQuads.get(vi).push(q);
            }
          }
        }
        // Pull extreme boundary vertices inward
        for (const [vi, quads] of vertQuads) {
          const allExtreme = quads.every(q => extremeQuads.has(q));
          if (!allExtreme) continue;
          // Find nearest interior neighbour (1 step inward)
          const vr = Math.floor(vi / cols);
          const vc = vi % cols;
          let nr = vr, nc = vc;
          if (vr === 0) nr = 1;
          else if (vr === rows-1) nr = rows-2;
          if (vc === 0) nc = 1;
          else if (vc === cols-1) nc = cols-2;
          const ni = nr * cols + nc;
          if (ni !== vi) {
            // Blend 70% toward interior neighbour
            vertices[vi*2]   = vertices[vi*2]   * 0.3 + vertices[ni*2]   * 0.7;
            vertices[vi*2+1] = vertices[vi*2+1] * 0.3 + vertices[ni*2+1] * 0.7;
          }
        }
        // Recompute bounds after trimming
        minX = Infinity; minY = Infinity; maxX = -Infinity; maxY = -Infinity;
        for (let i = 0; i < vertices.length; i += 2) {
          minX = Math.min(minX, vertices[i]);
          minY = Math.min(minY, vertices[i+1]);
          maxX = Math.max(maxX, vertices[i]);
          maxY = Math.max(maxY, vertices[i+1]);
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

/**
 * Validate that a homography is physically plausible for panorama stitching.
 * Uses the Jacobian at the image centre (not just the top-left 2×2) so that
 * perspective terms don't cause false-positive reflection detection.
 * Rejects if:
 * - Local affine Jacobian at image centre has negative determinant (reflection)
 * - Rotation angle > 120 degrees (likely spurious match)
 * - Extreme scale change (>4x or <0.25x)
 * - Any source corner projects behind the camera
 */
function isHomographyValid(H, srcW, srcH) {
  // Work on a local copy to avoid mutating the caller's homography array
  const Hn = new Float64Array(9);
  for (let i = 0; i < 9; i++) Hn[i] = H[i];
  // Normalize so Hn[8] = 1 before checking
  if (Math.abs(Hn[8]) > 1e-10) {
    const inv = 1.0 / Hn[8];
    for (let i = 0; i < 9; i++) Hn[i] *= inv;
  }
  H = Hn; // use local copy for rest of function

  // Compute local affine Jacobian at the image centre (more robust than
  // top-left 2×2 for perspective homographies with non-trivial H[6], H[7]).
  const cx = (srcW || 100) * 0.5;
  const cy = (srcH || 100) * 0.5;
  const w = H[6] * cx + H[7] * cy + H[8];
  if (w < 0.1) {
    console.warn('Homography rejected: image centre behind camera');
    return false;
  }
  const w2 = w * w;
  const u = H[0] * cx + H[1] * cy + H[2];
  const v = H[3] * cx + H[4] * cy + H[5];

  // Jacobian of the projective warp at (cx, cy):
  //   dX/dx = (H[0]*w - H[6]*u) / w²     dX/dy = (H[1]*w - H[7]*u) / w²
  //   dY/dx = (H[3]*w - H[6]*v) / w²     dY/dy = (H[4]*w - H[7]*v) / w²
  const a = (H[0] * w - H[6] * u) / w2;
  const b = (H[1] * w - H[7] * u) / w2;
  const c = (H[3] * w - H[6] * v) / w2;
  const d = (H[4] * w - H[7] * v) / w2;

  // Determinant of local Jacobian — negative means reflection
  const det2x2 = a * d - b * c;
  if (det2x2 < 0) {
    console.warn('Homography rejected: reflection detected (det < 0)');
    return false;
  }
  
  // Compute rotation angle from atan2(c, a) — should be small for panoramas
  const rotationRad = Math.atan2(c, a);
  const rotationDeg = Math.abs(rotationRad * 180 / Math.PI);
  if (rotationDeg > 120) {
    console.warn(`Homography rejected: extreme rotation ${rotationDeg.toFixed(1)}°`);
    return false;
  }
  
  // Compute scale from sqrt(det) — should be near 1 for same-camera images
  const scale = Math.sqrt(Math.abs(det2x2));
  if (scale > 4.0 || scale < 0.25) {
    console.warn(`Homography rejected: extreme scale ${scale.toFixed(2)}x`);
    return false;
  }

  // Check that all 4 corners of the source image project with positive denom.
  // This rejects homographies where the perspective vanishing line intersects the image.
  if (srcW && srcH) {
    const corners = [[0,0], [srcW,0], [srcW,srcH], [0,srcH]];
    for (const [x, y] of corners) {
      const denom = H[6] * x + H[7] * y + H[8];
      if (denom < 0.1) {
        console.warn(`Homography rejected: source corner (${x},${y}) has denom=${denom.toFixed(4)} (behind camera)`);
        return false;
      }
    }
  }
  
  return true;
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
        if (denom > 1e-4) {
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

// Sample depth value at image-space (x, y) from depth map.
// Depth map may be at a different resolution (e.g. 256x256) than the image (w x h),
// so coordinates are scaled accordingly.
function sampleDepth(depthData, w, h, x, y) {
  // Infer depth map dimensions from data length (always square from model)
  const depthSize = Math.round(Math.sqrt(depthData.length));
  const sx = depthSize / w;
  const sy = depthSize / h;
  const ix = Math.min(Math.max(Math.round(x * sx), 0), depthSize - 1);
  const iy = Math.min(Math.max(Math.round(y * sy), 0), depthSize - 1);
  return depthData[iy * depthSize + ix] / 65535.0;
}

/**
 * Weighted DLT with Tikhonov regularization toward a global homography.
 *
 * Standard weighted DLT can overfit in mesh cells far from correspondences,
 * producing wild local warps. Tikhonov regularization penalises deviation
 * from the global homography H_global, pulling the solution toward a
 * physically reasonable warp where data is sparse.
 *
 * Solves: min_h  Σ_k w_k |A_k h|²  +  γ |h - h_global|²
 * where γ = regularization strength (adaptive based on effective support).
 *
 * Reference: Zaragoza et al., "As-Projective-As-Possible Image Stitching
 * with Moving DLT", CVPR 2013 (with Tikhonov extension).
 *
 * @param srcPts source points [[x,y], ...]
 * @param dstPts destination points [[x,y], ...]
 * @param weights per-point weights (Gaussian kernel × face boost × depth)
 * @param globalH optional 9-element global homography to regularize toward
 * @param gamma regularization strength (0 = off, higher = more regularization)
 * @returns 9-element Float64Array (row-major 3×3) or null if degenerate
 */
function weightedDLT(srcPts, dstPts, weights, globalH, gamma) {
  const n = srcPts.length;
  if (n < 4) return null;
  const regGamma = gamma || 0;

  // Build weighted A matrix for Ah = 0
  // Each correspondence gives 2 rows in A (9 columns)
  // Row 1: [0, 0, 0, -w*x, -w*y, -w, w*y'*x, w*y'*y, w*y']
  // Row 2: [w*x, w*y, w, 0, 0, 0, -w*x'*x, -w*x'*y, -w*x']
  const cols = 9;

  // Use AtA (9x9) directly instead of forming the full A matrix
  const AtA = new Float64Array(81); // 9x9
  // Pre-allocate row buffers outside the loop to avoid per-point allocation
  const r1 = new Float64Array(9);
  const r2 = new Float64Array(9);

  for (let k = 0; k < n; k++) {
    const w = weights[k];
    if (w < 1e-12) continue;
    const sx = srcPts[k][0];
    const sy = srcPts[k][1];
    const dx = dstPts[k][0];
    const dy = dstPts[k][1];

    // Row 1 of A: [0, 0, 0, -w*sx, -w*sy, -w, w*dy*sx, w*dy*sy, w*dy]
    r1[0] = 0; r1[1] = 0; r1[2] = 0;
    r1[3] = -w * sx; r1[4] = -w * sy; r1[5] = -w;
    r1[6] = w * dy * sx; r1[7] = w * dy * sy; r1[8] = w * dy;
    // Row 2 of A: [w*sx, w*sy, w, 0, 0, 0, -w*dx*sx, -w*dx*sy, -w*dx]
    r2[0] = w * sx; r2[1] = w * sy; r2[2] = w;
    r2[3] = 0; r2[4] = 0; r2[5] = 0;
    r2[6] = -w * dx * sx; r2[7] = -w * dx * sy; r2[8] = -w * dx;

    // Accumulate AtA += r1^T * r1 + r2^T * r2
    for (let i = 0; i < 9; i++) {
      for (let j = i; j < 9; j++) {
        const val = r1[i] * r1[j] + r2[i] * r2[j];
        AtA[i * 9 + j] += val;
        if (i !== j) AtA[j * 9 + i] += val; // symmetric
      }
    }
  }

  // Solve AtA * h = 0 by fixing h[8] = 1 → 8×8 system
  // With Tikhonov: (AtA_8 + γI) * h_8 = -AtA_col8 + γ * h_global_8
  const A8 = new Float64Array(64); // 8x8
  const b8 = new Float64Array(8);

  for (let i = 0; i < 8; i++) {
    for (let j = 0; j < 8; j++) {
      A8[i * 8 + j] = AtA[i * 9 + j];
    }
    b8[i] = -AtA[i * 9 + 8];
  }

  // Add Tikhonov regularization toward globalH
  if (regGamma > 0 && globalH) {
    for (let i = 0; i < 8; i++) {
      A8[i * 8 + i] += regGamma;
      b8[i] += regGamma * globalH[i]; // pull toward h_global[i]
    }
  }

  // Solve 8x8 system using Gaussian elimination
  const h8 = solveLinear8(A8, b8);
  if (!h8) return null;

  const H = new Float64Array(9);
  for (let i = 0; i < 8; i++) H[i] = h8[i];
  H[8] = 1.0;

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
 * Sample RGB pixel at (x, y) from an interleaved RGB buffer (Uint8Array, 3 bytes/pixel).
 * Uses bilinear interpolation for consistency with sampleGray.
 * Writes result into `out` array [R, G, B] and returns it.
 */
function sampleRGB(rgb, w, h, x, y, out) {
  const ix = Math.floor(x);
  const iy = Math.floor(y);
  const fx = x - ix;
  const fy = y - iy;

  const x0 = Math.min(Math.max(ix, 0), w - 1);
  const x1 = Math.min(x0 + 1, w - 1);
  const y0 = Math.min(Math.max(iy, 0), h - 1);
  const y1 = Math.min(y0 + 1, h - 1);

  for (let c = 0; c < 3; c++) {
    const v00 = rgb[(y0 * w + x0) * 3 + c];
    const v10 = rgb[(y0 * w + x1) * 3 + c];
    const v01 = rgb[(y1 * w + x0) * 3 + c];
    const v11 = rgb[(y1 * w + x1) * 3 + c];
    out[c] = (1 - fx) * (1 - fy) * v00 + fx * (1 - fy) * v10 +
              (1 - fx) * fy * v01 + fx * fy * v11;
  }
  return out;
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
