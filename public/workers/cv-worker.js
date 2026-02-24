// cv-worker.js - classic worker using OpenCV.js
// Loads opencv.js via importScripts and exposes message handlers per spec.
// TODO: Provide full implementations of feature extraction, matching, homography estimation, refinement and local mesh computation.
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
let cvModule = null;
let images = {}; // imageId -> {width, height, gray:Uint8Array, depth?:Uint16Array, keypoints, descriptors}

self.addEventListener('message', async (ev) => {
  const msg = ev.data;
  try {
    if (msg.type === 'init') {
      const opencvPath = msg.opencvPath || 'opencv/opencv.js';
      importScripts(new URL(opencvPath, msg.baseUrl).toString());
      // Wait for cv to be ready
      const checkReady = () => (typeof cv !== 'undefined' && cv.Mat) ? true : false;
      const start = Date.now();
      while (!checkReady()) {
        await new Promise(r => setTimeout(r, 50));
        if (Date.now() - start > 15000) throw new Error('OpenCV failed to load in time');
      }
      cvModule = cv;
      cvReady = true;
      postMessage({type:'progress', stage:'cv-init', percent:100, info:'OpenCV loaded'});
      return;
    }

    if (!cvReady) throw new Error('cv-worker not initialized');

    if (msg.type === 'addImage') {
      const {imageId, grayBuffer, width, height, depth} = msg;
      const gray = new Uint8ClampedArray(grayBuffer);
      images[imageId] = {width, height, gray, depth};
      postMessage({type:'progress', stage:'addImage', percent:100, info:`added ${imageId}`});
      return;
    }

    if (msg.type === 'computeFeatures') {
      const {orbParams} = msg;
      // Simple ORB extraction per image
      for (const [id, img] of Object.entries(images)) {
        const mat = cv.matFromArray(img.height, img.width, cv.CV_8UC1, img.gray);
        const orb = new cv.ORB(orbParams && orbParams.nFeatures ? orbParams.nFeatures : 2000);
        const keypoints = new cv.KeyPointVector();
        const descriptors = new cv.Mat();
        orb.detectAndCompute(mat, new cv.Mat(), keypoints, descriptors);
        // Serialize keypoints to Float32Array [x,y,...]
        const kps = new Float32Array(keypoints.size() * 2);
        for (let i=0;i<keypoints.size();i++) {
          const kp = keypoints.get(i);
          kps[i*2] = kp.pt.x;
          kps[i*2+1] = kp.pt.y;
        }
        const descCols = descriptors.cols;
        const descBuf = new Uint8Array(descriptors.data);
        img.keypoints = kps.buffer;
        img.descriptors = descBuf.buffer;
        orb.delete(); keypoints.delete(); descriptors.delete(); mat.delete();
        postMessage({type:'features', imageId:id, keypointsBuffer:img.keypoints, descriptorsBuffer:img.descriptors, descCols});
      }
      return;
    }

    if (msg.type === 'matchGraph') {
      const params = msg;
      // TODO: implement pair selection, knn matching, ratio test, RANSAC homography estimation
      // For now, return an empty edges list to let caller handle fallback.
      postMessage({type:'edges', edges: []});
      return;
    }

    if (msg.type === 'buildMST') {
      // TODO: build MST from edges and return order + parent mapping
      postMessage({type:'mst', refId:null, order:[], parent:{}});
      return;
    }

    if (msg.type === 'computeLocalMesh') {
      // TODO: compute per-vertex local homographies and return mesh buffers
      postMessage({type:'mesh', imageId: msg.imageId, verticesBuffer: new Float32Array([]).buffer, uvsBuffer: new Float32Array([]).buffer, indicesBuffer: new Uint32Array([]).buffer, bounds:{minX:0,minY:0,maxX:0,maxY:0}});
      return;
    }

    if (msg.type === 'refine') {
      // TODO: implement LM refinement loop over homography params
      postMessage({type:'progress', stage:'refine', percent:100, info:'refinement placeholder completed'});
      return;
    }

    postMessage({type:'error', message:`unknown message: ${JSON.stringify(msg).slice(0,200)}`});
  } catch (err) {
    postMessage({type:'error', message: err.message || err.toString()});
  }
}, false);
