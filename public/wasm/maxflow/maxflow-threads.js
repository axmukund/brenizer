(function registerBrenizerMaxflowThreadsLoader() {
  self.__brenizerMaxflowAssetLoaders = self.__brenizerMaxflowAssetLoaders || {};
  self.__brenizerMaxflowAssetLoaders.threads = async function loadBrenizerMaxflowThreads(options) {
    if (options && options.workerUrl) {
      const workerResponse = await fetch(options.workerUrl);
      if (!workerResponse.ok) {
        throw new Error(`Thread helper fetch failed: HTTP ${workerResponse.status}`);
      }
      await workerResponse.text();
    }
    if (options && options.wasmUrl) {
      const wasmResponse = await fetch(options.wasmUrl);
      if (!wasmResponse.ok) {
        throw new Error(`Threaded artifact fetch failed: HTTP ${wasmResponse.status}`);
      }
      await wasmResponse.arrayBuffer();
    }
    throw new Error('Threaded maxflow artifact placeholder present, but no compiled WASM solver is bundled');
  };
})();
