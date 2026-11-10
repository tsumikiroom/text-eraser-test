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
      self.postMessage({ type: 'boxes', id: m.id, boxes: dbBoxes(data, prob.dims[3], prob.dims[2], m.opt || {}) });
    }
  } catch (e) {
    self.postMessage({ type: 'error', id: m.id, message: String((e && e.message) || e) });
  }
};

// DBNet post-process: threshold the text probability map, take connected regions, grow them back out
// (the model predicts a shrunken core of each text line). Each region gets a box turned to its own direction
// (principal axis of its pixels), so slanted text gets a slanted box instead of a big upright one that also
// covers the wall around it. Boxes come back in input pixels: centre cx, cy, size w x h, angle a (radians,
// within +-45 degrees; a vertical column is a tall box with a near 0).
function dbBoxes(prob, w, h, opt) {
  const TH = opt.th || 0.3, BOX_TH = opt.bth || 0.5, UNCLIP = opt.unclip || 1.5;
  const lab = new Uint8Array(w * h), stack = new Int32Array(w * h), pix = new Int32Array(w * h), out = [];
  for (let s = 0; s < w * h; s++) {
    if (lab[s] || prob[s] < TH) continue;
    let sp = 0, cnt = 0, sum = 0, sx = 0, sy = 0;
    stack[sp++] = s; lab[s] = 1;
    while (sp) {
      const i = stack[--sp], x = i % w, y = (i / w) | 0;
      pix[cnt++] = i; sum += prob[i]; sx += x; sy += y;
      if (x > 0 && !lab[i - 1] && prob[i - 1] >= TH) { lab[i - 1] = 1; stack[sp++] = i - 1; }
      if (x < w - 1 && !lab[i + 1] && prob[i + 1] >= TH) { lab[i + 1] = 1; stack[sp++] = i + 1; }
      if (y > 0 && !lab[i - w] && prob[i - w] >= TH) { lab[i - w] = 1; stack[sp++] = i - w; }
      if (y < h - 1 && !lab[i + w] && prob[i + w] >= TH) { lab[i + w] = 1; stack[sp++] = i + w; }
    }
    if (cnt < 6 || sum / cnt < BOX_TH) continue;
    const mx = sx / cnt, my = sy / cnt;
    let cxx = 0, cyy = 0, cxy = 0;
    for (let k = 0; k < cnt; k++) { const x = pix[k] % w - mx, y = ((pix[k] / w) | 0) - my; cxx += x * x; cyy += y * y; cxy += x * y; }
    let ang = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
    if (Math.abs(ang) < 0.03) ang = 0; // straight text stays exactly straight
    const c = Math.cos(ang), sn = Math.sin(ang);
    let u0 = Infinity, u1 = -Infinity, v0 = Infinity, v1 = -Infinity;
    for (let k = 0; k < cnt; k++) {
      const x = pix[k] % w - mx, y = ((pix[k] / w) | 0) - my, u = c * x + sn * y, v = -sn * x + c * y;
      if (u < u0) u0 = u; if (u > u1) u1 = u; if (v < v0) v0 = v; if (v > v1) v1 = v;
    }
    let bw = u1 - u0 + 1, bh = v1 - v0 + 1;
    if (Math.min(bw, bh) < 2) continue;
    const d = (bw * bh * UNCLIP) / (2 * (bw + bh));
    const ou = (u0 + u1) / 2, ov = (v0 + v1) / 2;
    let a = ang;
    bw += 2 * d; bh += 2 * d;
    if (a > Math.PI / 4) { a -= Math.PI / 2; [bw, bh] = [bh, bw]; } else if (a < -Math.PI / 4) { a += Math.PI / 2; [bw, bh] = [bh, bw]; }
    out.push({ cx: mx + c * ou - sn * ov + 0.5, cy: my + sn * ou + c * ov + 0.5, w: bw, h: bh, a });
  }
  return out;
}
