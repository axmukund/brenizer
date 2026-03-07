(function registerBrenizerMaxflowThreadsLoader() {
  self.__brenizerMaxflowAssetLoaders = self.__brenizerMaxflowAssetLoaders || {};

  function deriveModuleUrl(scriptUrl) {
    return scriptUrl.replace(/\.js(\?.*)?$/, '.module.js$1');
  }

  function deriveWorkerUrl(scriptUrl) {
    return scriptUrl.replace(/\.js(\?.*)?$/, '.worker.js$1');
  }

  function resolveFactory(globalName) {
    if (typeof self[globalName] === 'function') return self[globalName];
    try {
      const candidate = Function(`return typeof ${globalName} === "function" ? ${globalName} : null;`)();
      if (typeof candidate === 'function') {
        self[globalName] = candidate;
        return candidate;
      }
    } catch {
      // Ignore lookup failures and let the caller raise a clearer error.
    }
    return null;
  }

  function readCString(Module, ptr) {
    if (!ptr) return 'unknown error';
    if (typeof Module.UTF8ToString === 'function') return Module.UTF8ToString(ptr);
    let out = '';
    for (let i = ptr; Module.HEAPU8[i] !== 0; i++) {
      out += String.fromCharCode(Module.HEAPU8[i]);
    }
    return out || 'unknown error';
  }

  function allocFloat32(Module, values) {
    if (!values || values.length === 0) return 0;
    const ptr = Module._malloc(values.byteLength);
    Module.HEAPF32.set(values, ptr >> 2);
    return ptr;
  }

  function allocUint8(Module, values) {
    if (!values || values.length === 0) return 0;
    const ptr = Module._malloc(values.byteLength);
    Module.HEAPU8.set(values, ptr);
    return ptr;
  }

  function createSolverBridge(Module, backendId, description) {
    return {
      backendId,
      description,
      async solve(args) {
        const nodeCount = args.gridW * args.gridH;
        const allocations = [];
        const remember = (ptr) => {
          if (ptr) allocations.push(ptr);
          return ptr;
        };

        try {
          const dataCostsPtr = remember(allocFloat32(Module, args.dataCosts));
          const edgeWeightsHPtr = remember(allocFloat32(Module, args.edgeWeightsH));
          const edgeWeightsVPtr = remember(allocFloat32(Module, args.edgeWeightsV));
          const hardConstraintsPtr = remember(allocUint8(Module, args.hardConstraints));
          const labelsPtr = remember(Module._malloc(nodeCount));
          const statsPtr = remember(Module._malloc(Int32Array.BYTES_PER_ELEMENT * 4));
          Module.HEAP32.fill(0, statsPtr >> 2, (statsPtr >> 2) + 4);

          const startedAt = performance.now();
          const rc = Module._solve_grid(
            args.gridW,
            args.gridH,
            dataCostsPtr,
            edgeWeightsHPtr,
            edgeWeightsVPtr,
            hardConstraintsPtr,
            labelsPtr,
            statsPtr,
          );
          const solverMs = Math.round(performance.now() - startedAt);
          if (rc !== 0) {
            const message = readCString(Module, Module._last_error_message());
            throw new Error(`Threaded WASM maxflow solve failed (${rc}): ${message}`);
          }

          const labels = new Uint8Array(Module.HEAPU8.slice(labelsPtr, labelsPtr + nodeCount));
          const stats = new Int32Array(Module.HEAP32.slice(statsPtr >> 2, (statsPtr >> 2) + 4));
          return {
            labels,
            backendId,
            solverMs,
            pushes: stats[0] || 0,
            relabels: stats[1] || 0,
            globalRelabels: stats[2] || 0,
          };
        } finally {
          for (const ptr of allocations.reverse()) {
            Module._free(ptr);
          }
        }
      },
    };
  }

  self.__brenizerMaxflowAssetLoaders.threads = async function loadBrenizerMaxflowThreads(options) {
    const moduleUrl = deriveModuleUrl(options.scriptUrl);
    let factory = resolveFactory('createBrenizerMaxflowThreadsModule');
    if (!factory) {
      importScripts(moduleUrl);
    }
    factory = resolveFactory('createBrenizerMaxflowThreadsModule');
    if (!factory) {
      throw new Error('Threaded module factory did not register');
    }

    const Module = await factory({
      locateFile(path) {
        if (/\.wasm($|\?)/.test(path)) return options.wasmUrl;
        if (/\.worker\.js($|\?)/.test(path)) return options.workerUrl || deriveWorkerUrl(options.scriptUrl);
        return new URL(path, moduleUrl).toString();
      },
      mainScriptUrlOrBlob: moduleUrl,
    });

    return createSolverBridge(Module, 'wasm-threads', 'compiled threaded maxflow');
  };
})();
