// WebGPU port of the painterly morph tool.
//
// Milestone 1: scaffold + smooth organic crossfade.
// The JS shell (state, Tweakpane, slots, recorder, presets) mirrors the WebGL2
// version but the rendering path is rebuilt on WebGPU. Modes will be ported in
// subsequent milestones; right now only the default smooth dissolve runs.

import { Pane } from 'tweakpane';

const canvas = document.getElementById('canvas');

if (!navigator.gpu) {
  document.getElementById('gpu-error').classList.add('show');
  throw new Error('WebGPU not available');
}

const adapter = await navigator.gpu.requestAdapter();
if (!adapter) {
  document.getElementById('gpu-error').classList.add('show');
  throw new Error('No GPU adapter');
}
const device = await adapter.requestDevice();
device.addEventListener('uncapturederror', e => {
  console.error('[WebGPU uncaptured]', e.error?.message || e.error);
});
const ctx = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format: presentationFormat, alphaMode: 'opaque' });

const GPU_MAX_TEX = device.limits.maxTextureDimension2D;
console.log('[transition-tool-v2] device limits.maxTextureDimension2D =', GPU_MAX_TEX);

// ============================================================================
// Shader (WGSL)
// ============================================================================
const SHADER = /* wgsl */`
struct VSOut {
  @builtin(position) pos: vec4f,
  @location(0) uv: vec2f,
};

struct Params {
  t: f32,
  spread: f32,
  organic: f32,
  edges: f32,
  maskScale: f32,
  seed: f32,
  validA: u32,
  validB: u32,
  scaleA: vec2f,
  offsetA: vec2f,
  scaleB: vec2f,
  offsetB: vec2f,
  bg: vec3f,
  curve: u32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
  // 6-vertex fullscreen triangle pair, with UV in [0,1] (y up to match WebGL).
  let positions = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f( 1.0, -1.0), vec2f(-1.0,  1.0),
    vec2f(-1.0,  1.0), vec2f( 1.0, -1.0), vec2f( 1.0,  1.0),
  );
  let uvs = array<vec2f, 6>(
    vec2f(0.0, 1.0), vec2f(1.0, 1.0), vec2f(0.0, 0.0),
    vec2f(0.0, 0.0), vec2f(1.0, 1.0), vec2f(1.0, 0.0),
  );
  var out: VSOut;
  out.pos = vec4f(positions[idx], 0.0, 1.0);
  out.uv = uvs[idx];
  return out;
}

fn applyCurve(x: f32, mode: u32) -> f32 {
  let c = clamp(x, 0.0, 1.0);
  if (mode == 1u) { return c * c * (3.0 - 2.0 * c); }     // ease-in-out
  if (mode == 2u) { return c * c; }                        // ease-in
  if (mode == 3u) { return 1.0 - (1.0 - c) * (1.0 - c); }  // ease-out
  return c;
}

fn hash21(q: vec2f) -> f32 {
  var x = fract(q * vec2f(123.34, 456.21));
  x += dot(x, x + 45.32);
  return fract(x.x * x.y);
}
fn vnoise(q: vec2f) -> f32 {
  let i = floor(q);
  let f = fract(q);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(q: vec2f) -> f32 {
  var v = 0.0;
  var amp = 0.5;
  var pp = q;
  for (var i = 0; i < 5; i = i + 1) {
    v += amp * vnoise(pp);
    pp *= 2.03;
    amp *= 0.5;
  }
  return v;
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }

fn sampleFit(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32) -> vec4f {
  if (valid == 0u) { return vec4f(p.bg, 1.0); }
  let q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec4f(p.bg, 1.0); }
  // textureSampleLevel avoids the uniform-control-flow requirement of
  // textureSample (no implicit derivatives → safe inside conditionals).
  return textureSampleLevel(tex, samp, q, 0.0);
}

fn edgeMag(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32) -> f32 {
  if (valid == 0u) { return 0.0; }
  let e = 0.0025;
  let cx1 = luma(sampleFit(tex, uv + vec2f( e, 0.0), scale, offset, valid).rgb);
  let cx2 = luma(sampleFit(tex, uv - vec2f( e, 0.0), scale, offset, valid).rgb);
  let cy1 = luma(sampleFit(tex, uv + vec2f(0.0, e), scale, offset, valid).rgb);
  let cy2 = luma(sampleFit(tex, uv - vec2f(0.0, e), scale, offset, valid).rgb);
  return clamp(length(vec2f(cx1 - cx2, cy1 - cy2)) * 4.0, 0.0, 1.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  let t = applyCurve(p.t, p.curve);

  let cA = sampleFit(texA, uv, p.scaleA, p.offsetA, p.validA);
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB);
  let lA = luma(cA.rgb);
  let lB = luma(cB.rgb);

  // organic mask: noise + lum delta - edges
  let n1 = fbm(uv * p.maskScale + p.seed * 0.13);
  let n2 = fbm(uv * p.maskScale * 2.3 + 17.0 + p.seed * 0.09);
  let noiseMask = mix(n1, n2, 0.35);
  let lumMask = 0.5 + 0.5 * (lB - lA);
  var mask = mix(noiseMask, lumMask, p.organic);
  let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA);
  let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB);
  mask = clamp(mask - p.edges * max(eA, eB) * 0.45, 0.0, 1.0);

  let sp = mix(0.05, 0.7, p.spread);
  let mixT = clamp(smoothstep(mask - sp, mask + sp, t), 0.0, 1.0);
  return vec4f(mix(cA.rgb, cB.rgb, mixT), 1.0);
}
`;

const module = device.createShaderModule({ code: SHADER });
const compInfo = await module.getCompilationInfo();
if (compInfo.messages.length) {
  for (const m of compInfo.messages) {
    console[m.type === 'error' ? 'error' : 'warn']('[WGSL]', m.type, m.lineNum + ':' + m.linePos, m.message);
  }
}

// Bind group layout
const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer: { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
  ],
});
device.pushErrorScope('validation');
const pipeline = device.createRenderPipeline({
  label: 'main-pipeline',
  layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
  vertex: { module, entryPoint: 'vs' },
  fragment: { module, entryPoint: 'fs', targets: [{ format: presentationFormat }] },
  primitive: { topology: 'triangle-list' },
});
device.popErrorScope().then(err => { if (err) console.error('[pipeline error]', err.message); });

// Uniform buffer — std140-ish layout. Build with care:
//   t f32 | spread f32 | organic f32 | edges f32   (16)
//   maskScale f32 | seed f32 | validA u32 | validB u32  (16)
//   scaleA vec2f | offsetA vec2f                  (16)
//   scaleB vec2f | offsetB vec2f                  (16)
//   bg vec3f | curve u32                          (16)
// Total: 80 bytes.
const UBO_SIZE = 80;
const uniformBuffer = device.createBuffer({
  size: UBO_SIZE,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const uboHost = new ArrayBuffer(UBO_SIZE);
const uboF32 = new Float32Array(uboHost);
const uboU32 = new Uint32Array(uboHost);

const sampler = device.createSampler({
  magFilter: 'linear', minFilter: 'linear',
  addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
});

// Placeholder 1×1 textures so the bind group is valid before images load.
function makePlaceholderTexture() {
  const tex = device.createTexture({
    size: [1, 1, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, new Uint8Array([0, 0, 0, 255]), { bytesPerRow: 4 }, [1, 1, 1]);
  return tex;
}
let texA = makePlaceholderTexture();
let texB = makePlaceholderTexture();

function makeBindGroup() {
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texA.createView() },
      { binding: 2, resource: texB.createView() },
      { binding: 3, resource: sampler },
    ],
  });
}
let bindGroup = makeBindGroup();

async function uploadImageToSlot(img, slot) {
  const bitmap = await createImageBitmap(img);
  const w = bitmap.width, h = bitmap.height;
  console.log(`[upload ${slot}] bitmap ${w}x${h}`);
  const tex = device.createTexture({
    label: `tex${slot}`,
    size: [w, h, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.pushErrorScope('validation');
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [w, h, 1]);
  device.popErrorScope().then(err => { if (err) console.error(`[upload ${slot}] error`, err.message); });
  if (slot === 'A') { texA.destroy(); texA = tex; }
  else              { texB.destroy(); texB = tex; }
  bitmap.close();
  bindGroup = makeBindGroup();
  console.log(`[upload ${slot}] new bind group ready`);
}

// ============================================================================
// State
// ============================================================================
const state = {
  imgA: null, imgB: null,
  t: 0,
  playing: false,
  loop: true,
  startTime: 0,
  reverse: false,
  duration: 10.0,
  organic: 0.65,
  edges: 0.25,
  spread: 0.55,
  maskScale: 0.9,
  curve: 0,        // 0 linear
  seed: 42,
  fit: 'cover',
  bg: '#000000',
  zoomA: 1.0, panAx: 0.0, panAy: 0.0,
  zoomB: 1.0, panBx: 0.0, panBy: 0.0,
};

// ============================================================================
// Sizing
// ============================================================================
function fitInfo(img, cw, ch, mode) {
  if (!img) return { sx: 1, sy: 1, ox: 0, oy: 0 };
  const ia = img.naturalWidth / img.naturalHeight;
  const ca = cw / ch;
  if (mode === 'stretch') return { sx: 1, sy: 1, ox: 0, oy: 0 };
  if (mode === 'cover') {
    if (ia > ca) { const sx = ca / ia; return { sx, sy: 1, ox: (1 - sx) * 0.5, oy: 0 }; }
    const sy = ia / ca; return { sx: 1, sy, ox: 0, oy: (1 - sy) * 0.5 };
  }
  // contain
  if (ia > ca) { const sy = ca / ia; return { sx: 1, sy, ox: 0, oy: (1 - sy) * 0.5 }; }
  const sx = ia / ca; return { sx, sy: 1, ox: (1 - sx) * 0.5, oy: 0 };
}
function composedFit(slot, cw, ch) {
  const img = slot === 'A' ? state.imgA : state.imgB;
  const z = slot === 'A' ? state.zoomA : state.zoomB;
  const px = slot === 'A' ? state.panAx : state.panBx;
  const py = slot === 'A' ? state.panAy : state.panBy;
  const f = fitInfo(img, cw, ch, state.fit);
  return {
    sx: f.sx * z, sy: f.sy * z,
    ox: z * f.ox + 0.5 * (1 - z) - z * px,
    oy: z * f.oy + 0.5 * (1 - z) - z * py,
  };
}

function resizeCanvas() {
  if (!state.imgA && !state.imgB) return;
  const ref = state.imgA || state.imgB;
  const minimized = document.body.classList.contains('minimized');
  const sidePanel = minimized ? 0 : 360;
  const padding = minimized ? 0 : 32;
  const maxW = window.innerWidth - sidePanel - padding;
  const maxH = window.innerHeight - padding;
  const ia = ref.naturalWidth / ref.naturalHeight;
  let w = ref.naturalWidth, h = ref.naturalHeight;
  const longer = Math.max(w, h);
  if (longer > GPU_MAX_TEX) {
    const scale = GPU_MAX_TEX / longer;
    w = Math.round(w * scale); h = Math.round(h * scale);
  }
  canvas.width = w; canvas.height = h;
  const fit = Math.min(maxW / w, maxH / h, 1);
  canvas.style.width = (w * fit) + 'px';
  canvas.style.height = (h * fit) + 'px';
}

function hexToRgb(hex) {
  const m = hex.match(/^#([0-9a-f]{6})$/i);
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

function writeUniforms() {
  const cw = canvas.width, ch = canvas.height;
  const fA = composedFit('A', cw, ch);
  const fB = composedFit('B', cw, ch);
  const bg = hexToRgb(state.bg);

  // Offset layout, in bytes / 4 (u32 indices):
  // 0  t            (f32)
  // 1  spread       (f32)
  // 2  organic      (f32)
  // 3  edges        (f32)
  // 4  maskScale    (f32)
  // 5  seed         (f32)
  // 6  validA       (u32)
  // 7  validB       (u32)
  // 8  scaleA.x     (f32)   <- vec2 aligned at 8 floats (32 bytes) ✓
  // 9  scaleA.y     (f32)
  // 10 offsetA.x    (f32)
  // 11 offsetA.y    (f32)
  // 12 scaleB.x     (f32)
  // 13 scaleB.y     (f32)
  // 14 offsetB.x    (f32)
  // 15 offsetB.y    (f32)
  // 16 bg.r         (f32)   <- vec3 aligned at 16 floats (64 bytes) ✓
  // 17 bg.g         (f32)
  // 18 bg.b         (f32)
  // 19 curve        (u32)
  uboF32[0] = state.t;
  uboF32[1] = state.spread;
  uboF32[2] = state.organic;
  uboF32[3] = state.edges;
  uboF32[4] = state.maskScale;
  uboF32[5] = state.seed;
  uboU32[6] = state.imgA ? 1 : 0;
  uboU32[7] = state.imgB ? 1 : 0;
  uboF32[8] = fA.sx; uboF32[9] = fA.sy;
  uboF32[10] = fA.ox; uboF32[11] = fA.oy;
  uboF32[12] = fB.sx; uboF32[13] = fB.sy;
  uboF32[14] = fB.ox; uboF32[15] = fB.oy;
  uboF32[16] = bg[0]; uboF32[17] = bg[1]; uboF32[18] = bg[2];
  uboU32[19] = state.curve;
  device.queue.writeBuffer(uniformBuffer, 0, uboHost);
}

// ============================================================================
// Render loop
// ============================================================================
let _frameCount = 0;
window.__frameCount = () => _frameCount;
function render() {
  _frameCount++;
  if (state.playing) {
    const now = performance.now();
    const elapsed = (now - state.startTime) / 1000;
    let pT = elapsed / state.duration;
    if (pT >= 1) {
      if (state.loop) {
        state.startTime = now; pT = 0;
        state.reverse = !state.reverse;
      } else {
        pT = 1; state.playing = false;
        if (typeof btnPlay !== 'undefined') btnPlay.title = 'Play';
      }
    }
    state.t = state.reverse ? (1 - pT) : pT;
    if (typeof bT !== 'undefined') bT.refresh();
  }

  if (state.imgA || state.imgB) {
    writeUniforms();
    const enc = device.createCommandEncoder();
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bindGroup);
    pass.draw(6);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  requestAnimationFrame(render);
}
requestAnimationFrame(render);

// ============================================================================
// UI — image slots
// ============================================================================
const filepicker = document.getElementById('filepicker');
let pickingSlot = null;
document.querySelectorAll('.slot').forEach(s => {
  s.addEventListener('click', () => { pickingSlot = s.dataset.slot; filepicker.click(); });
});
filepicker.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) loadFile(f, pickingSlot);
  e.target.value = '';
});

function loadFile(file, slot) {
  if (!file.type.startsWith('image/')) return;
  loadFromUrl(URL.createObjectURL(file), slot);
}
function loadFromUrl(url, slot) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = async () => {
    if (slot === 'A') state.imgA = img; else state.imgB = img;
    await uploadImageToSlot(img, slot);
    updateSlotPreview(slot, url);
    canvas.classList.remove('empty');
    resizeCanvas();
    maybeAutoplay();
  };
  img.src = url;
}
function updateSlotPreview(slot, url) {
  const el = document.querySelector(`.slot[data-slot="${slot}"]`);
  el.querySelector('.placeholder')?.remove();
  let im = el.querySelector('img');
  if (!im) { im = document.createElement('img'); el.appendChild(im); }
  im.src = url;
}

let _autoplayStarted = false;
function maybeAutoplay() {
  if (_autoplayStarted) return;
  if (!state.imgA || !state.imgB) return;
  _autoplayStarted = true;
  state.playing = true;
  state.t = 0;
  state.startTime = performance.now();
  if (typeof btnPlay !== 'undefined') btnPlay.title = 'Pause';
}
loadFromUrl('./defaults/lofoten_A.jpg', 'A');
loadFromUrl('./defaults/lofoten_B.jpg', 'B');

// per-slot drag-and-drop
document.querySelectorAll('.slot').forEach(s => {
  s.addEventListener('dragenter', e => { e.preventDefault(); e.stopPropagation(); s.classList.add('drop-target'); });
  s.addEventListener('dragover',  e => { e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = 'copy'; });
  s.addEventListener('dragleave', e => { e.preventDefault(); e.stopPropagation(); s.classList.remove('drop-target'); });
  s.addEventListener('drop', e => {
    e.preventDefault(); e.stopPropagation();
    s.classList.remove('drop-target');
    const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
    if (files.length) loadFile(files[0], s.dataset.slot);
  });
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop',     e => e.preventDefault());

document.getElementById('swap').addEventListener('click', async () => {
  if (!state.imgA && !state.imgB) return;
  const tmpImg = state.imgA; state.imgA = state.imgB; state.imgB = tmpImg;
  // re-upload both
  if (state.imgA) await uploadImageToSlot(state.imgA, 'A');
  if (state.imgB) await uploadImageToSlot(state.imgB, 'B');
  const sA = document.querySelector('.slot[data-slot="A"] img');
  const sB = document.querySelector('.slot[data-slot="B"] img');
  if (sA && sB) { const u = sA.src; sA.src = sB.src; sB.src = u; }
});
document.getElementById('clear').addEventListener('click', () => {
  state.imgA = null; state.imgB = null;
  texA.destroy(); texB.destroy();
  texA = makePlaceholderTexture();
  texB = makePlaceholderTexture();
  bindGroup = makeBindGroup();
  document.querySelectorAll('.slot').forEach(s => {
    s.querySelector('img')?.remove();
    if (!s.querySelector('.placeholder')) {
      const ph = document.createElement('span');
      ph.className = 'placeholder'; ph.textContent = 'click / drop';
      s.appendChild(ph);
    }
  });
  canvas.classList.add('empty');
});

window.addEventListener('resize', resizeCanvas);

// Panel minimize
const togglePanelBtn = document.getElementById('toggle-panel');
function setMinimized(min) {
  document.body.classList.toggle('minimized', min);
  togglePanelBtn.textContent = min ? '›' : '‹';
  togglePanelBtn.title = min ? 'Show controls (Tab)' : 'Hide controls (Tab)';
  requestAnimationFrame(() => requestAnimationFrame(resizeCanvas));
  setTimeout(resizeCanvas, 250);
}
togglePanelBtn.addEventListener('click', () => setMinimized(!document.body.classList.contains('minimized')));

// ============================================================================
// Tweakpane (minimal for milestone 1)
// ============================================================================
const pane = new Pane({ container: document.getElementById('tp-host') });

const fPlay = pane.addFolder({ title: 'Playback', expanded: true });
const bT = fPlay.addBinding(state, 't', { min: 0, max: 1, step: 0.001, label: 'progress' });
fPlay.addBinding(state, 'duration', { min: 0.5, max: 30, step: 0.1 });
const btnPlay = fPlay.addButton({ title: 'Pause' });
const btnRestart = fPlay.addButton({ title: 'Restart from start' });
const btnLoop = fPlay.addButton({ title: 'Loop: on' });
btnPlay.on('click', () => {
  if (state.playing) { state.playing = false; btnPlay.title = 'Play'; }
  else {
    state.playing = true;
    const consumed = state.reverse ? (1 - state.t) : state.t;
    state.startTime = performance.now() - consumed * state.duration * 1000;
    btnPlay.title = 'Pause';
  }
});
btnRestart.on('click', () => {
  state.t = 0; state.reverse = false; state.playing = true;
  state.startTime = performance.now(); btnPlay.title = 'Pause';
});
btnLoop.on('click', () => {
  state.loop = !state.loop;
  btnLoop.title = 'Loop: ' + (state.loop ? 'on' : 'off');
});

const fDis = pane.addFolder({ title: 'Dissolve', expanded: true });
fDis.addBinding(state, 'organic',   { min: 0, max: 1, step: 0.01 });
fDis.addBinding(state, 'edges',     { min: -1, max: 1, step: 0.01 });
fDis.addBinding(state, 'spread',    { min: 0, max: 1, step: 0.01 });
fDis.addBinding(state, 'maskScale', { min: 0.3, max: 4, step: 0.05, label: 'mask scale' });
fDis.addBinding(state, 'curve', { options: { 'linear': 0, 'ease-in-out': 1, 'ease-in': 2, 'ease-out': 3 } });
fDis.addBinding(state, 'seed', { min: 0, max: 999, step: 1 });

const fImg = pane.addFolder({ title: 'Framing', expanded: false });
fImg.addBinding(state, 'zoomA', { min: 0.5, max: 4, step: 0.01, label: 'A zoom' });
fImg.addBinding(state, 'panAx', { min: -1, max: 1, step: 0.005, label: 'A pan x' });
fImg.addBinding(state, 'panAy', { min: -1, max: 1, step: 0.005, label: 'A pan y' });
fImg.addBinding(state, 'zoomB', { min: 0.5, max: 4, step: 0.01, label: 'B zoom' });
fImg.addBinding(state, 'panBx', { min: -1, max: 1, step: 0.005, label: 'B pan x' });
fImg.addBinding(state, 'panBy', { min: -1, max: 1, step: 0.005, label: 'B pan y' });

const fStyle = pane.addFolder({ title: 'Style', expanded: false });
fStyle.addBinding(state, 'fit', {
  options: { 'cover (crop)': 'cover', 'contain': 'contain', 'stretch': 'stretch' },
});
fStyle.addBinding(state, 'bg', { view: 'color' });

// Keyboard
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === ' ')          { e.preventDefault(); btnPlay.element.querySelector('button').click(); }
  if (e.key === 'Tab')        { e.preventDefault(); setMinimized(!document.body.classList.contains('minimized')); }
  if (e.key === 'ArrowLeft')  { state.t = Math.max(0, state.t - 0.02); pane.refresh(); }
  if (e.key === 'ArrowRight') { state.t = Math.min(1, state.t + 0.02); pane.refresh(); }
});

// Expose for headless / automation experiments
window.__tool = { state, pane, device, adapter };
console.log('[transition-tool-v2] WebGPU ready, format:', presentationFormat);
