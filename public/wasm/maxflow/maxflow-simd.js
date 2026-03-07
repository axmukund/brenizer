(function registerBrenizerMaxflowSimdLoader() {
  self.__brenizerMaxflowAssetLoaders = self.__brenizerMaxflowAssetLoaders || {};
  self.__brenizerMaxflowAssetLoaders.simd = async function loadBrenizerMaxflowSimd(options) {
    if (options && options.wasmUrl) {
      const response = await fetch(options.wasmUrl);
      if (!response.ok) {
        throw new Error(`SIMD artifact fetch failed: HTTP ${response.status}`);
      }
      await response.arrayBuffer();
    }
    throw new Error('SIMD maxflow artifact placeholder present, but no compiled WASM solver is bundled');
  };
})();
