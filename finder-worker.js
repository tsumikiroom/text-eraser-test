// Runs the PaddleOCR text detector off the page's main thread, so drawing never waits for it.
// In: { type: 'load', ortUrl, modelUrl, ep } / { type: 'find', id, w, h, px (RGBA bytes) }
// Out: { type: 'ready', backend } / { type: 'boxes', id, boxes } / { type: 'error', id, message }
'use strict';
let session = null;

self.onmessage = async (ev) => {
  const m = ev.data;
  try {
    if (m.type === 'load') {
      importScripts(m.ortUrl + 'ort.webgpu.min.js');
      ort.env.wasm.wasmPaths = m.ortUrl;
      ort.env.wasm.numThreads = 1; // threads need cross-origin isolation, which GitHub Pages does not give
      const tries = m.ep === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];
      let lastErr = null;
      for (const ep of tries) {
        try {
          session = await ort.InferenceSession.create(m.modelUrl, { executionProviders: [ep], graphOptimizationLevel: 'all' });
          self.postMessage({ type: 'ready', backend: ep });
          return;
        } catch (e) { lastErr = e; }
      }
      throw lastErr || new Error('no backend');
    }
    if (m.type === 'find') {
      const { w, h } = m, px = new Uint8ClampedArray(m.px), n = w * h;
      const x = new Float32Array(3 * n);
      // PaddleOCR normalisation, channels in BGR order like its OpenCV-based training
      const mean = [0.406, 0.456, 0.485], std = [0.225, 0.224, 0.229];
      for (let ch = 0; ch < 3; ch++) {
        const off = ch * n, src = 2 - ch, mu = mean[ch], k = 1 / (255 * std[ch]), b = mu / std[ch];
        for (let i = 0; i < n; i++) x[off + i] = px[i * 4 + src] * k - b;
      }
      const out = await session.run({ [session.inputNames[0]]: new ort.Tensor('float32', x, [1, 3, h, w]) });
      const prob = out[session.outputNames[0]];
      const data = prob.data instanceof Float32Array ? prob.data : await prob.getData();
      self.postMessage({ type: 'boxes', id: m.id, boxes: dbBoxes(data, prob.dims[3], prob.dims[2]) });
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: m.id, message: String((e && e.message) || e) });
  }
};

// DBNet post-process: threshold the text probability map, take connected regions, grow them back out
// (the model predicts a shrunken core of each text line). Boxes come back in 0..1 of the frame.
function dbBoxes(prob, w, h) {
  const TH = 0.3, BOX_TH = 0.5, UNCLIP = 1.5;
  const lab = new Uint8Array(w * h), stack = new Int32Array(w * h), out = [];
  for (let s = 0; s < w * h; s++) {
    if (lab[s] || prob[s] < TH) continue;
    let sp = 0, x0 = w, y0 = h, x1 = 0, y1 = 0, cnt = 0, sum = 0;
    stack[sp++] = s; lab[s] = 1;
    while (sp) {
      const i = stack[--sp], x = i % w, y = (i / w) | 0;
      cnt++; sum += prob[i];
      if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      if (x > 0 && !lab[i - 1] && prob[i - 1] >= TH) { lab[i - 1] = 1; stack[sp++] = i - 1; }
      if (x < w - 1 && !lab[i + 1] && prob[i + 1] >= TH) { lab[i + 1] = 1; stack[sp++] = i + 1; }
      if (y > 0 && !lab[i - w] && prob[i - w] >= TH) { lab[i - w] = 1; stack[sp++] = i - w; }
      if (y < h - 1 && !lab[i + w] && prob[i + w] >= TH) { lab[i + w] = 1; stack[sp++] = i + w; }
    }
    const bw = x1 - x0 + 1, bh = y1 - y0 + 1;
    if (cnt < 6 || Math.min(bw, bh) < 2 || sum / cnt < BOX_TH) continue;
    const d = (bw * bh * UNCLIP) / (2 * (bw + bh));
    out.push({ x: (x0 - d) / w, y: (y0 - d) / h, w: (bw + 2 * d) / w, h: (bh + 2 * d) / h });
  }
  return out;
}
