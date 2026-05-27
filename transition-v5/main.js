// transition-v5 — focused AI dissolve.
//
// One image, picked-by-click regions via SAM, sequential reveal that fades
// each region in turn against an inpainted background. Phase 1 (this file):
// scaffold + SAM + sequential reveal + a placeholder blurred inpaint so the
// dissolve mechanic can be validated end-to-end. Phase 2: replace the
// placeholder with cumulative LaMa inpaints (N+1 texture-array layers, each
// pixel reaching its target value via its own layer).

import { Pane } from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';

const TRANSFORMERS_URL = 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.5.2';
// SAM model registry. slimsam is fast & small but struggles with thin shapes;
// sam-vit-base is full Meta SAM (~360 MB) with much better thin-structure
// segmentation. Pickable in the Segment pane.
const SAM_MODELS = {
  'slimsam-77 (~25 MB, fast)':    'Xenova/slimsam-77-uniform',
  'sam-vit-base (~360 MB, best)': 'Xenova/sam-vit-base',
};
const SAM_DEFAULT_MODEL = 'Xenova/slimsam-77-uniform';

// ============================================================================
// WebGPU init
// ============================================================================
const canvas = document.getElementById('canvas');
if (!navigator.gpu) { document.getElementById('gpu-error').classList.add('show'); throw new Error('WebGPU not supported'); }
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) { document.getElementById('gpu-error').classList.add('show'); throw new Error('No WebGPU adapter'); }
const device = await adapter.requestDevice();
const ctx = canvas.getContext('webgpu');
const presentationFormat = navigator.gpu.getPreferredCanvasFormat();
ctx.configure({ device, format: presentationFormat, alphaMode: 'opaque' });
const GPU_MAX_TEX = device.limits?.maxTextureDimension2D ?? 8192;

// ============================================================================
// Shader — single mode: sequential region reveal
//
// Each pixel reads its target "fade time" from texRegions.r. The global t
// drives a smoothstep around that per-pixel midpoint, mixing A → Inpaint.
// When `inpaintValid == 0` (user hasn't generated an inpaint yet), we fall
// back to A so the dissolve has nothing to reveal — image stays visible.
// ============================================================================
const SHADER = /* wgsl */`
struct Params {
  t: f32,
  softness: f32,
  valid: u32,
  inpaintValid: u32,
  videoStrength: f32, // 0 = no video influence; 1 = luma can shift pixelT by ±0.5
  videoInvert: u32,
  videoValid: u32,
  _pad: u32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texInpaint: texture_2d<f32>;
@group(0) @binding(3) var texRegions: texture_2d<f32>;
@group(0) @binding(4) var samp: sampler;
@group(0) @binding(5) var texT: texture_2d<f32>;

struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

@vertex
fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
  let qx = array<f32, 6>(-1.0,  1.0, -1.0, -1.0,  1.0,  1.0);
  let qy = array<f32, 6>(-1.0, -1.0,  1.0,  1.0, -1.0,  1.0);
  var o: VSOut;
  o.pos = vec4f(qx[idx], qy[idx], 0.0, 1.0);
  o.uv  = vec2f((qx[idx] + 1.0) * 0.5, 1.0 - (qy[idx] + 1.0) * 0.5);
  return o;
}

@fragment
fn fs(in: VSOut) -> @location(0) vec4f {
  if (p.valid == 0u) {
    return vec4f(0.04, 0.04, 0.04, 1.0);
  }
  let cA = textureSampleLevel(texA, samp, in.uv, 0.0).rgb;
  let cI = select(cA, textureSampleLevel(texInpaint, samp, in.uv, 0.0).rgb, p.inpaintValid == 1u);
  let reg = textureSampleLevel(texRegions, samp, in.uv, 0.0);
  var pixelT     = clamp(reg.r, 0.0, 1.0);
  let fadeWeight = clamp(reg.g, 0.0, 1.0); // 0 for background pixels: never fade
  // Video-driven offset: video luma at this pixel shifts its reveal time, so
  // the bright/dark pattern of the mask video acts as a dissolve front within
  // each region's time slot. Skipped entirely when no video is loaded.
  if (p.videoValid == 1u && p.videoStrength > 0.001) {
    let cT = textureSampleLevel(texT, samp, in.uv, 0.0).rgb;
    let luma = 0.299 * cT.r + 0.587 * cT.g + 0.114 * cT.b;
    let signed = (luma - 0.5) * 2.0; // -1..1
    let shift = select(signed, -signed, p.videoInvert == 1u) * p.videoStrength * 0.5;
    pixelT = clamp(pixelT + shift, 0.0, 1.0);
  }
  let sp = max(0.001, p.softness);
  let mixT = smoothstep(pixelT - sp, pixelT + sp, p.t) * fadeWeight;
  return vec4f(mix(cA, cI, mixT), 1.0);
}
`;

const shaderModule = device.createShaderModule({ code: SHADER });

const bindGroupLayout = device.createBindGroupLayout({
  entries: [
    { binding: 0, visibility: GPUShaderStage.FRAGMENT, buffer:  { type: 'uniform' } },
    { binding: 1, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 2, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 3, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, sampler: { type: 'filtering' } },
    { binding: 5, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
  ],
});

const pipeline = device.createRenderPipeline({
  layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
  vertex:   { module: shaderModule, entryPoint: 'vs' },
  fragment: { module: shaderModule, entryPoint: 'fs', targets: [{ format: presentationFormat }] },
  primitive: { topology: 'triangle-list' },
});

// ============================================================================
// Uniform buffer (16 bytes: t, softness, valid, inpaintValid)
// ============================================================================
const UBO_SIZE = 32;
const uniformBuffer = device.createBuffer({
  size: UBO_SIZE,
  usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
});
const uboHost = new ArrayBuffer(UBO_SIZE);
const uboF32 = new Float32Array(uboHost);
const uboU32 = new Uint32Array(uboHost);

// ============================================================================
// Textures + sampler
// ============================================================================
const sampler = device.createSampler({
  magFilter: 'linear', minFilter: 'linear',
  addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge',
});

function makePlaceholderTexture(rgba = [0, 0, 0, 255]) {
  const tex = device.createTexture({
    size: [1, 1, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, new Uint8Array(rgba), { bytesPerRow: 4 }, [1, 1, 1]);
  return tex;
}

let texA       = makePlaceholderTexture();
let texInpaint = makePlaceholderTexture();
// r=255 (=pixelT 1.0) — placeholder regions texture means "nothing dissolves
// until t reaches 1", which leaves the image visible while no regions exist.
const placeholderRegions = makePlaceholderTexture([255, 0, 0, 255]);
let texRegions = placeholderRegions;
// Video mask texture — gray placeholder until user drops a video into the T
// slot. videoValid flag in the UBO gates whether the shader actually samples.
let texT = makePlaceholderTexture([128, 128, 128, 255]);

function makeBindGroup() {
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texA.createView() },
      { binding: 2, resource: texInpaint.createView() },
      { binding: 3, resource: texRegions.createView() },
      { binding: 4, resource: sampler },
      { binding: 5, resource: texT.createView() },
    ],
  });
}
let bindGroup = makeBindGroup();

// ============================================================================
// App state
// ============================================================================
const state = {
  t: 0,
  duration: 10,
  loop: true,
  playing: false,
  softness: 0.3,
  stagger: 1.0,       // 0 = all regions fade simultaneously; 1 = fully sequential
  videoStrength: 0,   // 0 = no video influence; 0.5 = strong; 1 = max shift ±0.5
  videoInvert: false,
  videoT: null,       // HTMLVideoElement when a transition video is loaded
  img: null,            // HTMLImageElement currently loaded
  inpaintValid: false,  // has the user generated an inpaint for the current image?
  inpaintMethod: 'lama',// 'lama' | 'background' | 'api' | 'sd'
  bgImg: null,          // HTMLImageElement for the "manual background" method
  lamaPatchPad: 96,     // px of source context around each mask before LaMa sees it
  lamaMaskDilation: 6,  // px to grow the SAM mask so LaMa has a halo to blend into
  lamaPasses: 1,        // 1 = single pass; >1 = jittered crops averaged together
  lamaTextureBlend: 0.5,// 0 = pure LaMa (smooth); 1 = full paper-grain transfer
  samModelId: SAM_DEFAULT_MODEL,
};

// Recorder state — hoisted here so Tweakpane bindings can target it (the
// Recorder section that uses these fields is defined further down).
const recState = {
  recording: false,
  fps: 30,
  sizeMode: 'src',
  muxer: null,
};
try {
  const sm = localStorage.getItem('v5-sam-model');
  if (sm && Object.values(SAM_MODELS).includes(sm)) state.samModelId = sm;
} catch {}

// ============================================================================
// DOM refs
// ============================================================================
const slotEl       = document.getElementById('slot-A');
const filepicker   = document.getElementById('filepicker');
const overlay      = document.getElementById('overlay');
const overlayCtx   = overlay.getContext('2d');

const slotBgEl   = document.getElementById('slot-bg');
const bgSlotWrap = document.getElementById('bg-slot-wrap');

const regionsListEl  = document.getElementById('regions-list');
const regionCountEl  = document.getElementById('region-count');
const regionsClearBtn = document.getElementById('regions-clear');

// Tweakpane state proxies — Tweakpane bindings live on these objects so the
// pane can read/write them cleanly. State changes propagate into real `state`
// via .on('change') handlers wired during pane construction.
const samUI     = { status: 'load model to begin', mode: false };
const inpaintUI = { status: 'save regions first', method: 'lama' };
const apiUI     = { key: '', model: 'fal-ai/flux-pro/v1/fill', prompt: '', negative: '' };
const recUI     = { fps: 30, sizeMode: 'src' };
// Tweakpane element refs set during pane construction — used to update labels,
// expand/collapse method-specific folders, and trigger refresh() on monitors.
const tp = {
  pane: null,
  panePlay: null, paneSeg: null, paneInpaint: null, paneRec: null,
  playBtn: null, loopToggle: null,
  samStatusBinding: null, inpaintStatusBinding: null,
  inpaintMethodBinding: null,
  fLama: null, fApi: null,
  inpaintBtn: null, recBtn: null,
};

// ============================================================================
// Status helpers + UI updaters
// ============================================================================
function setStatus(msg) {
  samUI.status = msg;
  if (tp.samStatusBinding) tp.samStatusBinding.refresh();
  refreshWorkflow();
  console.log('[SAM]', msg);
}
function setInpaintStatus(msg) {
  inpaintUI.status = msg;
  if (tp.inpaintStatusBinding) tp.inpaintStatusBinding.refresh();
  refreshWorkflow();
}

// Workflow steps — visualizes where the user is in the pipeline. States are
// derived from real state; this is purely a display function.
const workflowSteps = document.querySelectorAll('#workflow .step');
function refreshWorkflow() {
  const stepDone = {
    1: !!state.img,
    2: regions.length > 0,
    3: !!state.inpaintValid,
    4: false, // "Play" is always available, never marked done
  };
  // Active = the first step that isn't done yet (or step 4 if everything's done)
  let activeIdx = 1;
  for (let i = 1; i <= 4; i++) {
    if (!stepDone[i]) { activeIdx = i; break; }
    if (i === 4) activeIdx = 4;
  }
  workflowSteps.forEach(el => {
    const n = parseInt(el.dataset.step, 10);
    el.classList.toggle('done',   stepDone[n]);
    el.classList.toggle('active', n === activeIdx && !stepDone[n]);
  });
}

// Progress bar helpers. Each bar is identified by its element id; we toggle
// the `active` class to show it and write the inner .bar-fill width.
function setProgress(barId, pct, opts = {}) {
  const bar = document.getElementById(barId);
  if (!bar) return;
  const fill = bar.querySelector('.bar-fill');
  if (pct === null) {
    bar.classList.remove('active', 'indeterminate');
    fill.style.width = '0%';
    return;
  }
  bar.classList.add('active');
  if (opts.indeterminate) {
    bar.classList.add('indeterminate');
  } else {
    bar.classList.remove('indeterminate');
    fill.style.width = `${Math.max(0, Math.min(100, pct))}%`;
  }
}

// ============================================================================
// IndexedDB — persists the current image, the background image, and a small
// library of every image ever loaded (so refreshing the page restores state
// and previously-used images can be reloaded by clicking a thumbnail).
// ============================================================================
const IDB_NAME = 'transition-v5';
const IDB_IMG_STORE = 'images';
const IDB_LIB_STORE = 'library';

function idbOpen() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains(IDB_IMG_STORE)) db.createObjectStore(IDB_IMG_STORE);
      if (!db.objectStoreNames.contains(IDB_LIB_STORE)) {
        const store = db.createObjectStore(IDB_LIB_STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('addedAt', 'addedAt');
      }
    };
    r.onsuccess = () => resolve(r.result);
    r.onerror = () => reject(r.error);
  });
}
async function idbGet(key) {
  try {
    const db = await idbOpen();
    const result = await new Promise((resolve, reject) => {
      const req = db.transaction(IDB_IMG_STORE, 'readonly').objectStore(IDB_IMG_STORE).get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
    console.log(`[IDB] get ${key} →`, result ? `${result.size || '?'} bytes, type=${result.type || '?'}` : 'EMPTY');
    return result;
  } catch (err) {
    console.error(`[IDB] get ${key} failed`, err);
    return null;
  }
}
async function idbPut(key, value) {
  try {
    const db = await idbOpen();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_IMG_STORE, 'readwrite');
      tx.objectStore(IDB_IMG_STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    console.log(`[IDB] put ${key} (${value?.size || '?'} bytes, type=${value?.type || '?'})`);
  } catch (err) {
    console.error(`[IDB] put ${key} failed`, err);
  }
}
async function idbDelete(key) {
  try {
    const db = await idbOpen();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_IMG_STORE, 'readwrite');
      tx.objectStore(IDB_IMG_STORE).delete(key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
    console.log(`[IDB] delete ${key}`);
  } catch (err) {
    console.error(`[IDB] delete ${key} failed`, err);
  }
}
async function libList() {
  try {
    const db = await idbOpen();
    return await new Promise((resolve, reject) => {
      const tx  = db.transaction(IDB_LIB_STORE, 'readonly');
      const req = tx.objectStore(IDB_LIB_STORE).getAll();
      req.onsuccess = () => resolve((req.result || []).sort((a, b) => b.addedAt - a.addedAt));
      req.onerror = () => reject(req.error);
    });
  } catch { return []; }
}
async function libAdd(entry) {
  try {
    const db = await idbOpen();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_LIB_STORE, 'readwrite');
      const req = tx.objectStore(IDB_LIB_STORE).add(entry);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } catch { return null; }
}
async function libDelete(id) {
  try {
    const db = await idbOpen();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_LIB_STORE, 'readwrite');
      tx.objectStore(IDB_LIB_STORE).delete(id);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {}
}
async function makeThumb(blob, maxW = 256) {
  try {
    const bmp = await createImageBitmap(blob);
    const w = Math.min(maxW, bmp.width);
    const h = Math.round(w * bmp.height / bmp.width);
    const c = (typeof OffscreenCanvas !== 'undefined')
      ? new OffscreenCanvas(w, h)
      : Object.assign(document.createElement('canvas'), { width: w, height: h });
    c.getContext('2d').drawImage(bmp, 0, 0, w, h);
    bmp.close?.();
    if (c.convertToBlob) return await c.convertToBlob({ type: 'image/jpeg', quality: 0.78 });
    return await new Promise(r => c.toBlob(r, 'image/jpeg', 0.78));
  } catch { return null; }
}
async function blobToImage(blob) {
  const url = URL.createObjectURL(blob);
  const img = new Image();
  img.src = url;
  await img.decode();
  return img;
}

// ============================================================================
// Image upload
// ============================================================================
async function uploadImage(img) {
  state.img = img;
  state.inpaintValid = false;
  let bitmap = await createImageBitmap(img);
  let w = bitmap.width, h = bitmap.height;
  if (Math.max(w, h) > GPU_MAX_TEX) {
    const scale = GPU_MAX_TEX / Math.max(w, h);
    const nw = Math.round(w * scale), nh = Math.round(h * scale);
    const big = bitmap;
    bitmap = await createImageBitmap(big, { resizeWidth: nw, resizeHeight: nh, resizeQuality: 'high' });
    big.close();
    w = nw; h = nh;
    console.log(`[upload] downscaled to ${w}×${h} (max ${GPU_MAX_TEX})`);
  }
  const tex = device.createTexture({
    size: [w, h, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bitmap }, { texture: tex }, [w, h, 1]);
  bitmap.close();
  texA = tex;
  texInpaint = makePlaceholderTexture();
  regions.length = 0;
  sam.encodedImgRef = null;
  rebuildRegionsTexture();
  bindGroup = makeBindGroup();
  canvas.width = w; canvas.height = h;
  canvas.classList.remove('empty');
  // Reset zoom/pan so a fresh image fills the stage cleanly
  zoomState.zoom = 1; zoomState.panX = 0; zoomState.panY = 0;
  applyCanvasTransform();
  setSlotPreview(img);
  refreshRegionsUI();
  refreshRecBtn();
  setStatus('image loaded — Load SAM next');
  setInpaintStatus('save regions, then generate');
}

function setSlotPreview(img) {
  slotEl.innerHTML = '<span class="lab">image</span>';
  const im = document.createElement('img');
  im.src = img.src;
  slotEl.appendChild(im);
}

async function loadFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  await uploadImage(img);
  // Persist as "last loaded" and add to the library. Both writes are awaited
  // so a fast reload after upload doesn't lose them.
  await idbPut('imageA', file);
  // Switching to a new image: stale regions from the previous image no
  // longer apply, so clear the saved list.
  await idbDelete('regions');
  const thumb = await makeThumb(file, 256);
  if (thumb) {
    const id = await libAdd({ blob: file, thumb, addedAt: Date.now(), name: file.name || 'untitled' });
    state._libIdCurrent = id;
  }
  renderLibrary();
}

slotEl.addEventListener('click', () => filepicker.click());
filepicker.addEventListener('change', (e) => {
  const f = e.target.files?.[0];
  if (f) loadFile(f);
});

document.addEventListener('dragover', (e) => {
  e.preventDefault();
  if (e.dataTransfer?.types?.includes('Files')) slotEl.classList.add('drop-target');
});
document.addEventListener('dragleave', (e) => {
  if (e.target === document.documentElement) slotEl.classList.remove('drop-target');
});
document.addEventListener('drop', (e) => {
  e.preventDefault();
  slotEl.classList.remove('drop-target');
  slotTEl.classList.remove('drop-target');
  const f = e.dataTransfer?.files?.[0];
  if (!f) return;
  // Route by mime: video → T slot, image → A slot
  if (f.type.startsWith('video/')) loadVideoToT(f);
  else if (f.type.startsWith('image/')) loadFile(f);
});

// T-slot (transition video) — click opens a video file picker, drop accepts
// video files. Video is held in state.videoT (HTMLVideoElement) and each
// rAF frame the latest frame is copied to texT for the shader to sample.
const slotTEl = document.getElementById('slot-T');
slotTEl.addEventListener('click', () => {
  const p = document.createElement('input');
  p.type = 'file'; p.accept = 'video/*';
  p.addEventListener('change', () => {
    const f = p.files?.[0];
    if (f) loadVideoToT(f);
  });
  p.click();
});
slotTEl.addEventListener('dragover', (e) => { e.preventDefault(); slotTEl.classList.add('drop-target'); });
slotTEl.addEventListener('dragleave', () => slotTEl.classList.remove('drop-target'));
slotTEl.addEventListener('drop', (e) => {
  e.preventDefault();
  slotTEl.classList.remove('drop-target');
  const f = e.dataTransfer?.files?.[0];
  if (f?.type.startsWith('video/')) loadVideoToT(f);
});

async function loadVideoToT(file) {
  const url = URL.createObjectURL(file);
  slotTEl.innerHTML = '<span class="lab">video</span>';
  let v = document.createElement('video');
  v.muted = true; v.loop = true; v.playsInline = true; v.autoplay = true;
  v.style.cssText = 'width:100%;height:100%;object-fit:cover;display:block;';
  v.src = url;
  slotTEl.appendChild(v);
  v.addEventListener('loadeddata', () => {
    state.videoT = v;
    v.play().catch(() => {/* autoplay can be blocked until first user gesture */});
  }, { once: true });
  try { await idbPut('videoT', file); } catch {}
}

// Per-frame: copy the current video frame into texT. Sized lazily on first
// frame, then re-allocated only if the source dimensions change. videoT may
// briefly be in readyState<2 at startup; we skip those frames safely.
async function uploadVideoFrameToT() {
  const v = state.videoT;
  if (!v || v.readyState < 2) return;
  const w = v.videoWidth, h = v.videoHeight;
  if (!w || !h) return;
  if (!texT._videoSized || texT.width !== w || texT.height !== h) {
    try { texT.destroy(); } catch {}
    texT = device.createTexture({
      label: 'texT-video',
      size: [w, h, 1], format: 'rgba8unorm',
      usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
    });
    texT._videoSized = true;
    bindGroup = makeBindGroup();
  }
  try {
    device.queue.copyExternalImageToTexture({ source: v }, { texture: texT }, [w, h, 1]);
  } catch (err) { /* transient — happens during seek; next frame will succeed */ }
}

document.getElementById('clear').addEventListener('click', () => {
  state.img = null;
  state.inpaintValid = false;
  texA = makePlaceholderTexture();
  texInpaint = makePlaceholderTexture();
  regions.length = 0;
  sam.encodedImgRef = null;
  sam.maskCanvas = null;
  sam.points = [];
  rebuildRegionsTexture();
  bindGroup = makeBindGroup();
  canvas.classList.add('empty');
  slotEl.innerHTML = '<span class="lab">image</span><span class="placeholder">click / drop</span>';
  refreshRegionsUI();
  refreshRecBtn();
  drawOverlay();
  setStatus('cleared');
  setInpaintStatus('save regions first');
});

// ============================================================================
// Playback
// ============================================================================
function setT(t) {
  state.t = Math.max(0, Math.min(1, t));
  tp.panePlay?.refresh();
}
function setPlaying(p) {
  state.playing = p;
  if (tp.playBtn) tp.playBtn.title = p ? '⏸ Pause' : '▶ Play';
}

document.addEventListener('keydown', (e) => {
  if (e.target.matches('input, textarea')) return;
  if (e.code === 'Space') { e.preventDefault(); setPlaying(!state.playing); }
  if (e.key === 'Tab')    { e.preventDefault(); document.body.classList.toggle('minimized'); }
  if (e.key === 'r' || e.key === 'R') setT(0);
});

// ============================================================================
// Render loop
// ============================================================================
let lastTime = performance.now();
function frame(now) {
  const dt = (now - lastTime) / 1000;
  lastTime = now;
  if (state.playing && state.img) {
    setT(state.t + dt / Math.max(0.01, state.duration));
    if (state.t >= 1) {
      if (state.loop) setT(0);
      else setPlaying(false);
    }
  }
  render();
  requestAnimationFrame(frame);
}
requestAnimationFrame((t) => { lastTime = t; frame(t); });

function render() {
  uboF32[0] = state.t;
  uboF32[1] = state.softness;
  uboU32[2] = state.img ? 1 : 0;
  uboU32[3] = state.inpaintValid ? 1 : 0;
  uboF32[4] = state.videoStrength;
  uboU32[5] = state.videoInvert ? 1 : 0;
  uboU32[6] = state.videoT ? 1 : 0;
  uboU32[7] = 0;
  uploadVideoFrameToT();
  device.queue.writeBuffer(uniformBuffer, 0, uboHost);
  const encoder = device.createCommandEncoder();
  const view = ctx.getCurrentTexture().createView();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view, loadOp: 'clear', storeOp: 'store',
      clearValue: { r: 0.04, g: 0.04, b: 0.04, a: 1 },
    }],
  });
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  pass.draw(6);
  pass.end();
  device.queue.submit([encoder.finish()]);
}

// ============================================================================
// SAM — point-prompt segmentation. Same architecture as v4 but trimmed.
// ============================================================================
const sam = {
  lib: null, model: null, processor: null,
  rawImage: null, imageInputs: null, imageEmbeddings: null,
  encodedImgRef: null,
  points: [],          // [{x, y, label}] in source-image pixels
  maskCanvas: null,    // OffscreenCanvas at source resolution; alpha>0 = on
  segmentMode: false,
};

// Brush + mask-edit UI state. brushUI.mode toggles paint-mode (mousedown on
// the overlay paints into sam.maskCanvas instead of dropping a SAM point).
const brushUI = { mode: false, size: 30, action: 'include' };
const growUI  = { amount: 10 };
let brushDrag = null;  // { lastX, lastY } during an active stroke

// Ensure sam.maskCanvas exists at the SAM-encoded resolution (or the source
// image's natural resolution if no SAM encode has happened). Created lazily
// so the user can brush even before pressing "Encode image".
function ensureMaskCanvas() {
  if (sam.maskCanvas) return sam.maskCanvas;
  const ref = sam.encodedImgRef || state.img;
  if (!ref) return null;
  const w = ref.naturalWidth  || ref.width;
  const h = ref.naturalHeight || ref.height;
  sam.maskCanvas = new OffscreenCanvas(w, h);
  return sam.maskCanvas;
}

function paintBrushDot(x, y) {
  const c = ensureMaskCanvas();
  if (!c) return;
  const ctx = c.getContext('2d');
  if (brushUI.action === 'include') {
    ctx.globalCompositeOperation = 'source-over';
    ctx.fillStyle = 'rgba(255, 255, 255, 1)';
  } else {
    ctx.globalCompositeOperation = 'destination-out';
    ctx.fillStyle = 'rgba(0, 0, 0, 1)';
  }
  ctx.beginPath();
  ctx.arc(x, y, brushUI.size, 0, Math.PI * 2);
  ctx.fill();
  ctx.globalCompositeOperation = 'source-over';
}
function paintBrushLine(x1, y1, x2, y2) {
  const dist = Math.hypot(x2 - x1, y2 - y1);
  const steps = Math.max(1, Math.ceil(dist / Math.max(2, brushUI.size * 0.4)));
  for (let i = 0; i <= steps; i++) {
    const t = i / steps;
    paintBrushDot(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t);
  }
}

// Grow / shrink the current SAM mask in place. Uses the same dilateMask /
// erodeMask helpers (added below) as LaMa's halo step but applied to the
// active mask before saving as a region. So the user can grab a tight SAM
// outline and expand it 10–40 px to cover thin antennas, halos, etc.
function applyMaskMorphology(pixels, direction) {
  if (!sam.maskCanvas) { setStatus('no mask to grow/shrink — click first'); return; }
  const W = sam.maskCanvas.width, H = sam.maskCanvas.height;
  const ctx = sam.maskCanvas.getContext('2d');
  const img = ctx.getImageData(0, 0, W, H);
  const binMask = new Uint8Array(W * H);
  for (let i = 0, j = 3; i < binMask.length; i++, j += 4) binMask[i] = img.data[j] > 0 ? 1 : 0;
  const out = direction === 'grow'
    ? dilateMask(binMask, W, H, pixels)
    : erodeMask(binMask, W, H, pixels);
  for (let i = 0, j = 0; i < out.length; i++, j += 4) {
    img.data[j + 0] = 255;
    img.data[j + 1] = 255;
    img.data[j + 2] = 255;
    img.data[j + 3] = out[i] ? 180 : 0;
  }
  ctx.putImageData(img, 0, 0);
  drawOverlay();
  setStatus(`${direction === 'grow' ? 'grew' : 'shrunk'} mask by ${pixels}px`);
}

// Erosion via canvas-blur + high threshold (mirror of dilateMask).
function erodeMask(mask, W, H, radius) {
  if (radius <= 0) return mask;
  const a = new OffscreenCanvas(W, H);
  const aImg = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const v = mask[i] ? 255 : 0;
    aImg.data[i * 4 + 0] = v;
    aImg.data[i * 4 + 3] = 255;
  }
  a.getContext('2d').putImageData(aImg, 0, 0);
  const b = new OffscreenCanvas(W, H);
  const bctx = b.getContext('2d');
  bctx.filter = `blur(${radius}px)`;
  bctx.drawImage(a, 0, 0);
  bctx.filter = 'none';
  const blurred = bctx.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = blurred[i * 4] > 200 ? 1 : 0;
  return out;
}

// SAM button handlers are wired through Tweakpane below. Helper for the
// "clear current mask" action (referenced by Tweakpane and overlay alt-click).
function samClearMask() {
  sam.points = []; sam.maskCanvas = null; drawOverlay();
  setStatus('cleared — click to start');
}

async function samLoadModel() {
  if (sam.model && sam.loadedModelId === state.samModelId) {
    setStatus('model already loaded');
    return;
  }
  const modelId = state.samModelId;
  const sizeNote = modelId.includes('vit-base') ? '~360 MB' : '~25 MB';
  setStatus(`loading SAM (${sizeNote}) …`);
  try {
    sam.lib = sam.lib || await import(TRANSFORMERS_URL);
    // Clear stale embeddings if switching models
    sam.model = null;
    sam.processor = null;
    sam.imageEmbeddings = null;
    sam.imageInputs = null;
    sam.encodedImgRef = null;
    sam.model = await sam.lib.SamModel.from_pretrained(modelId, { device: 'webgpu', dtype: 'fp16' })
      .catch(async (e) => {
        console.warn('[SAM] webgpu/fp16 failed, falling back', e);
        return sam.lib.SamModel.from_pretrained(modelId);
      });
    sam.processor = await sam.lib.AutoProcessor.from_pretrained(modelId);
    sam.loadedModelId = modelId;
    setStatus('model ready — encode image next');
  } catch (err) {
    console.error('[SAM] load failed', err);
    setStatus(`load failed: ${err.message || err}`);
  }
}

async function samEncodeImage() {
  if (!sam.model) { setStatus('load model first'); return; }
  if (!state.img) { setStatus('drop an image first'); return; }
  setStatus('encoding image …');
  try {
    const bm = await createImageBitmap(state.img);
    const c = new OffscreenCanvas(bm.width, bm.height);
    const cctx = c.getContext('2d');
    cctx.drawImage(bm, 0, 0);
    bm.close();
    const d = cctx.getImageData(0, 0, c.width, c.height);
    sam.rawImage = new sam.lib.RawImage(d.data, c.width, c.height, 4);
    sam.imageInputs = await sam.processor(sam.rawImage);
    sam.imageEmbeddings = await sam.model.get_image_embeddings(sam.imageInputs);
    sam.encodedImgRef = state.img;
    sam.points = []; sam.maskCanvas = null; drawOverlay();
    setStatus(`encoded ${c.width}×${c.height} — turn on segment mode and click`);
  } catch (err) {
    console.error('[SAM] encode failed', err);
    setStatus(`encode failed: ${err.message || err}`);
  }
}

async function segmentAt(imgX, imgY, label = 1) {
  if (!sam.imageEmbeddings) { setStatus('encode image first'); return; }
  if (sam.encodedImgRef !== state.img) { setStatus('image changed — re-encode'); return; }
  sam.points.push({ x: imgX, y: imgY, label });
  await runSamInference();
}

// Run SAM with whatever's currently in sam.points (no new point added). Used
// by point-removal and undo-last-point so they don't have to duplicate the
// model invocation logic.
async function runSamInference() {
  if (!sam.imageEmbeddings) { setStatus('encode image first'); return; }
  if (!sam.points.length) { sam.maskCanvas = null; drawOverlay(); setStatus('no points — click to start'); return; }
  setStatus(`segmenting (${sam.points.length} pt${sam.points.length === 1 ? '' : 's'}) …`);
  try {
    const points = [sam.points.map(p => [p.x, p.y])];
    const labels = [sam.points.map(p => p.label)];
    const di = await sam.processor(sam.rawImage, { input_points: points, input_labels: labels });
    const outs = await sam.model({
      ...sam.imageEmbeddings,
      input_points: di.input_points,
      input_labels: di.input_labels,
    });
    const masks = await sam.processor.post_process_masks(
      outs.pred_masks, sam.imageInputs.original_sizes, sam.imageInputs.reshaped_input_sizes
    );
    const scores = outs.iou_scores.data;
    let bi = 0;
    for (let i = 1; i < scores.length; i++) if (scores[i] > scores[bi]) bi = i;
    sam.maskCanvas = buildMaskCanvas(masks[0], bi);
    drawOverlay();
    setStatus(`mask iou=${scores[bi].toFixed(2)} — refine, or save as region`);
  } catch (err) {
    console.error('[SAM] segment failed', err);
    setStatus(`segment failed: ${err.message || err}`);
  }
}

function buildMaskCanvas(tensor, idx) {
  const dims = tensor.dims;
  const H = dims[dims.length - 2];
  const W = dims[dims.length - 1];
  const stride = H * W;
  const start = idx * stride;
  const c = new OffscreenCanvas(W, H);
  const cx = c.getContext('2d');
  const img = cx.createImageData(W, H);
  for (let i = 0; i < stride; i++) {
    const on = Boolean(tensor.data[start + i]);
    const o = i * 4;
    img.data[o + 0] = 255;
    img.data[o + 1] = 255;
    img.data[o + 2] = 255;
    img.data[o + 3] = on ? 180 : 0;
  }
  cx.putImageData(img, 0, 0);
  return c;
}

// ============================================================================
// Overlay — tints saved regions in distinct colors; current candidate in white
// ============================================================================
function syncOverlaySize() {
  const r = canvas.getBoundingClientRect();
  overlay.style.left   = r.left   + 'px';
  overlay.style.top    = r.top    + 'px';
  overlay.style.width  = r.width  + 'px';
  overlay.style.height = r.height + 'px';
  const dpr = window.devicePixelRatio || 1;
  const bw = Math.max(1, Math.round(r.width  * dpr));
  const bh = Math.max(1, Math.round(r.height * dpr));
  if (overlay.width  !== bw) overlay.width  = bw;
  if (overlay.height !== bh) overlay.height = bh;
}

const REGION_COLORS = [
  '#4af', '#fb6', '#6fa', '#f6c',
  '#fc4', '#6cf', '#f74', '#a6f',
];

function tintMask(maskCanvas, colorHex, alpha = 0.5) {
  const c = new OffscreenCanvas(maskCanvas.width, maskCanvas.height);
  const cx = c.getContext('2d');
  cx.fillStyle = colorHex;
  cx.fillRect(0, 0, c.width, c.height);
  cx.globalCompositeOperation = 'destination-in';
  cx.drawImage(maskCanvas, 0, 0);
  cx.globalCompositeOperation = 'source-over';
  // Apply alpha by drawing onto a fresh canvas at lower opacity
  const out = new OffscreenCanvas(maskCanvas.width, maskCanvas.height);
  const ox = out.getContext('2d');
  ox.globalAlpha = alpha;
  ox.drawImage(c, 0, 0);
  return out;
}

function drawOverlay() {
  if (!sam.segmentMode) return;
  syncOverlaySize();
  overlayCtx.clearRect(0, 0, overlay.width, overlay.height);
  if (!sam.encodedImgRef) return;
  const iw = sam.encodedImgRef.naturalWidth  || sam.encodedImgRef.width;
  const ih = sam.encodedImgRef.naturalHeight || sam.encodedImgRef.height;
  overlayCtx.imageSmoothingEnabled = false;
  for (let i = 0; i < regions.length; i++) {
    const tinted = tintMask(regions[i].maskCanvas, REGION_COLORS[i % REGION_COLORS.length], 0.5);
    overlayCtx.drawImage(tinted, 0, 0, overlay.width, overlay.height);
  }
  if (sam.maskCanvas) {
    const tinted = tintMask(sam.maskCanvas, '#fff', 0.6);
    overlayCtx.drawImage(tinted, 0, 0, overlay.width, overlay.height);
  }
  const dpr = window.devicePixelRatio || 1;
  for (const pt of sam.points) {
    const cx = (pt.x / iw) * overlay.width;
    const cy = (pt.y / ih) * overlay.height;
    overlayCtx.beginPath();
    overlayCtx.arc(cx, cy, 6 * dpr, 0, Math.PI * 2);
    overlayCtx.fillStyle = pt.label === 1 ? '#4af' : '#f55';
    overlayCtx.fill();
    overlayCtx.lineWidth = 1.5 * dpr;
    overlayCtx.strokeStyle = '#000';
    overlayCtx.stroke();
  }
}

function setSegmentMode(on) {
  sam.segmentMode = on;
  overlay.classList.toggle('visible', on);
  overlay.classList.toggle('interactive', on);
  // Keep the Tweakpane checkbox in sync with the actual state
  if (samUI.mode !== on) { samUI.mode = on; tp.paneSeg?.refresh(); }
  if (on) drawOverlay();
}

overlay.addEventListener('click', (e) => {
  if (!sam.segmentMode) return;
  if (brushUI.mode) return; // brush owns mouse events when active
  if (!sam.encodedImgRef) { setStatus('encode image first'); return; }
  const r = overlay.getBoundingClientRect();
  const iw = sam.encodedImgRef.naturalWidth  || sam.encodedImgRef.width;
  const ih = sam.encodedImgRef.naturalHeight || sam.encodedImgRef.height;
  if (e.altKey) {
    sam.points = []; sam.maskCanvas = null; drawOverlay();
    setStatus('reset — click to start');
    return;
  }
  const px = ((e.clientX - r.left) / r.width)  * iw;
  const py = ((e.clientY - r.top)  / r.height) * ih;
  // If the click is very close to an existing point, REMOVE that point instead
  // of adding a new one. Tolerance is ~14 display px scaled into image coords.
  const hitImgRadius = 14 * (iw / Math.max(1, r.width));
  let nearestIdx = -1, nearestDist = Infinity;
  for (let i = 0; i < sam.points.length; i++) {
    const dx = sam.points[i].x - px, dy = sam.points[i].y - py;
    const d = Math.hypot(dx, dy);
    if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
  }
  if (nearestIdx >= 0 && nearestDist < hitImgRadius) {
    sam.points.splice(nearestIdx, 1);
    drawOverlay();
    if (sam.points.length === 0) {
      sam.maskCanvas = null;
      drawOverlay();
      setStatus('all points removed — click to start');
    } else {
      runSamInference();
    }
    return;
  }
  const label = e.shiftKey ? 0 : 1;
  segmentAt(px, py, label);
});

// Right-click on a point also removes it (alternative to clicking the dot).
overlay.addEventListener('contextmenu', (e) => {
  if (!sam.segmentMode) return;
  e.preventDefault();
  if (!sam.encodedImgRef || !sam.points.length) return;
  const r = overlay.getBoundingClientRect();
  const iw = sam.encodedImgRef.naturalWidth  || sam.encodedImgRef.width;
  const ih = sam.encodedImgRef.naturalHeight || sam.encodedImgRef.height;
  const px = ((e.clientX - r.left) / r.width)  * iw;
  const py = ((e.clientY - r.top)  / r.height) * ih;
  let nearestIdx = -1, nearestDist = Infinity;
  for (let i = 0; i < sam.points.length; i++) {
    const dx = sam.points[i].x - px, dy = sam.points[i].y - py;
    const d = Math.hypot(dx, dy);
    if (d < nearestDist) { nearestDist = d; nearestIdx = i; }
  }
  const hitImgRadius = 22 * (iw / Math.max(1, r.width));
  if (nearestIdx >= 0 && nearestDist < hitImgRadius) {
    sam.points.splice(nearestIdx, 1);
    drawOverlay();
    if (sam.points.length === 0) {
      sam.maskCanvas = null; drawOverlay();
      setStatus('all points removed — click to start');
    } else {
      runSamInference();
    }
  }
});

// Brush painting (active when sam.segmentMode AND brushUI.mode are both on).
function imgPxAtMouse(e) {
  const r = overlay.getBoundingClientRect();
  const ref = sam.encodedImgRef || state.img;
  const iw = ref?.naturalWidth  || ref?.width  || 1;
  const ih = ref?.naturalHeight || ref?.height || 1;
  return {
    x: ((e.clientX - r.left) / r.width)  * iw,
    y: ((e.clientY - r.top)  / r.height) * ih,
  };
}
overlay.addEventListener('mousedown', (e) => {
  if (!sam.segmentMode || !brushUI.mode || e.button !== 0) return;
  e.preventDefault();
  e.stopPropagation();
  const p = imgPxAtMouse(e);
  brushDrag = { x: p.x, y: p.y };
  paintBrushDot(p.x, p.y);
  drawOverlay();
});
overlay.addEventListener('mousemove', (e) => {
  if (!brushDrag) return;
  const p = imgPxAtMouse(e);
  paintBrushLine(brushDrag.x, brushDrag.y, p.x, p.y);
  brushDrag.x = p.x; brushDrag.y = p.y;
  drawOverlay();
});
document.addEventListener('mouseup', () => { brushDrag = null; });

window.addEventListener('resize', drawOverlay);
new ResizeObserver(drawOverlay).observe(canvas);

// ============================================================================
// Regions — saved per-pixel masks; build a per-pixel "fade time" GPU texture
// ============================================================================
const regions = []; // [{ id, name, w, h, data: Uint8Array(w*h), maskCanvas }]

// Persist regions to IDB so reload restores them. Stored as a plain array
// with the binary Uint8Array data — maskCanvas is regenerated on restore.
async function saveRegionsToIDB() {
  try {
    const payload = regions.map(r => ({
      id: r.id, name: r.name, w: r.w, h: r.h,
      data: r.data,
    }));
    await idbPut('regions', payload);
  } catch (err) { console.warn('[regions] save failed', err); }
}

function saveCurrentMaskAsRegion() {
  if (!sam.maskCanvas) { setStatus('no mask to save — click first'); return; }
  const W = sam.maskCanvas.width, H = sam.maskCanvas.height;
  const ctx2d = sam.maskCanvas.getContext('2d');
  const img = ctx2d.getImageData(0, 0, W, H);
  const data = new Uint8Array(W * H);
  for (let i = 0, j = 3; i < data.length; i++, j += 4) data[i] = img.data[j] > 0 ? 1 : 0;
  if (regions.length && (regions[0].w !== W || regions[0].h !== H)) {
    setStatus(`size mismatch ${W}×${H} ≠ ${regions[0].w}×${regions[0].h} — clear regions`);
    return;
  }
  // Copy the mask canvas so it survives the active-mask clear below
  const mc = new OffscreenCanvas(W, H);
  mc.getContext('2d').drawImage(sam.maskCanvas, 0, 0);
  const id = Date.now() + Math.random();
  const name = `region ${regions.length + 1}`;
  regions.push({ id, name, w: W, h: H, data, maskCanvas: mc });
  sam.points = []; sam.maskCanvas = null;
  state.inpaintValid = false;
  rebuildRegionsTexture();
  refreshRegionsUI();
  drawOverlay();
  saveRegionsToIDB();
  setStatus(`saved "${name}" — ${regions.length} region${regions.length === 1 ? '' : 's'} — click to add another`);
  setInpaintStatus('regions saved — generate inpaint when ready');
}

function removeRegion(id) {
  const i = regions.findIndex(r => r.id === id);
  if (i < 0) return;
  regions.splice(i, 1);
  state.inpaintValid = false;
  rebuildRegionsTexture();
  refreshRegionsUI();
  drawOverlay();
  saveRegionsToIDB();
}

function moveRegion(id, delta) {
  const i = regions.findIndex(r => r.id === id);
  if (i < 0) return;
  const ni = i + delta;
  if (ni < 0 || ni >= regions.length) return;
  [regions[i], regions[ni]] = [regions[ni], regions[i]];
  state.inpaintValid = false;
  rebuildRegionsTexture();
  refreshRegionsUI();
  drawOverlay();
  saveRegionsToIDB();
}

function renameRegion(id, newName) {
  const r = regions.find(r => r.id === id);
  if (r) { r.name = newName; saveRegionsToIDB(); }
}

regionsClearBtn.addEventListener('click', () => {
  if (!regions.length) return;
  regions.length = 0;
  state.inpaintValid = false;
  rebuildRegionsTexture();
  refreshRegionsUI();
  drawOverlay();
  saveRegionsToIDB();
  setStatus('all regions cleared');
  setInpaintStatus('save regions first');
});

function rebuildRegionsTexture() {
  if (!regions.length) {
    texRegions = placeholderRegions;
    bindGroup = makeBindGroup();
    return;
  }
  const W = regions[0].w, H = regions[0].h;
  const N = regions.length;
  // Spread the N regions evenly across the full [0,1] progress range. Each
  // region's midpoint = (i - 0.5) / N, so N=1 → 0.5 (center), N=2 → 0.25 and
  // 0.75, etc. Background pixels use fadeWeight=0 and never visibly fade.
  const slots = N;
  const data = new Uint8Array(W * H * 4);
  // Init all pixels as background — fadeWeight=0 means they stay at A forever.
  for (let i = 0; i < W * H; i++) {
    data[i * 4 + 0] = 255; // pixelT (unused for bg)
    data[i * 4 + 1] = 0;   // fadeWeight: 0 = no fade
    data[i * 4 + 3] = 255;
  }
  // Paint each region (later regions overwrite earlier — overlaps win by latest).
  // pixelT distributes regions across [0,1] but blends toward t=0.5 (the
  // shared midpoint) as state.stagger → 0, so all regions fade together at
  // stagger=0 and stay fully sequential at stagger=1.
  const sg = Math.max(0, Math.min(1, state.stagger));
  for (let r = 0; r < N; r++) {
    const region = regions[r];
    const pure  = ((r + 1) - 0.5) / slots; // fully-sequential midpoint
    const mixed = 0.5 + (pure - 0.5) * sg;  // stagger=0 → 0.5, stagger=1 → pure
    const ptByte = Math.round(mixed * 255);
    const m = region.data;
    for (let i = 0; i < W * H; i++) {
      if (m[i]) {
        data[i * 4 + 0] = ptByte;
        data[i * 4 + 1] = 255; // fadeWeight: 1 = fade fully during this region's time slot
      }
    }
  }
  const tex = device.createTexture({
    label: 'tex-regions',
    size: [W, H, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  device.queue.writeTexture({ texture: tex }, data, { bytesPerRow: W * 4 }, [W, H, 1]);
  texRegions = tex;
  bindGroup = makeBindGroup();
}

function refreshRegionsUI() {
  regionsListEl.innerHTML = '';
  regionCountEl.textContent = `(${regions.length})`;
  regionsClearBtn.disabled = !regions.length;
  refreshInpaintButton();
  if (!regions.length) {
    const row = document.createElement('div');
    row.className = 'region-row empty';
    row.textContent = '(none yet — segment, then + Save above)';
    regionsListEl.appendChild(row);
    return;
  }
  for (let i = 0; i < regions.length; i++) {
    const region = regions[i];
    const color = REGION_COLORS[i % REGION_COLORS.length];
    const row = document.createElement('div');
    row.className = 'region-row';
    row.style.borderLeftColor = color;
    row.innerHTML = `
      <span class="num">${i + 1}</span>
      <input class="name" type="text" />
      <button class="up" title="Move up">↑</button>
      <button class="down" title="Move down">↓</button>
      <button class="del" title="Delete">×</button>
    `;
    const nameInput = row.querySelector('.name');
    nameInput.value = region.name;
    nameInput.addEventListener('change', () => renameRegion(region.id, nameInput.value));
    row.querySelector('.up').addEventListener('click', () => moveRegion(region.id, -1));
    row.querySelector('.down').addEventListener('click', () => moveRegion(region.id, +1));
    row.querySelector('.del').addEventListener('click', () => removeRegion(region.id));
    regionsListEl.appendChild(row);
  }
}
refreshRegionsUI();

// ============================================================================
// Inpaint — real LaMa via onnxruntime-web (Carve/LaMa-ONNX, ~208 MB).
//
// Architecture (cumulative N-pass):
//   1. For each saved region i in order, run LaMa once with the cumulative
//      mask (union of regions 1..i). This gives N inpainted views of A,
//      each at full source resolution.
//   2. Bake a single target texture: for each pixel that belongs to region i,
//      copy from inpaint_i; background pixels stay at A (they won't visibly
//      fade because rebuildRegionsTexture sets their fadeWeight to 0).
//   3. Upload that one composite texture as texInpaint. The shader does the
//      rest exactly like the placeholder path — only the underlying texture
//      content is now real LaMa instead of a blur.
//
// Model contract (from Carve/LaMa-ONNX + the next-lama port):
//   input  "image": float32 [1, 3, 512, 512], values in [0, 1] (rgb/255)
//   input  "mask":  float32 [1, 1, 512, 512], values in {0, 1}
//   output         : float32 [1, 3, 512, 512], values already in [0, 255]
// ============================================================================

const LAMA_MODEL_URL = 'https://huggingface.co/Carve/LaMa-ONNX/resolve/main/lama_fp32.onnx';
const ORT_VERSION = '1.20.1';
const ORT_BASE = `https://cdn.jsdelivr.net/npm/onnxruntime-web@${ORT_VERSION}`;
const LAMA_INPUT_SIZE = 512;

const lama = {
  ort: null,
  session: null,
  buffer: null,   // kept so we can recreate the session on EP fallback
  ep: null,       // 'webgpu' | 'wasm' — actual execution provider in use
  loading: false,
};

// Restore persisted LaMa slider values + API fields before Tweakpane reads them
try {
  const lp  = localStorage.getItem('v5-lama-pad');     if (lp)  state.lamaPatchPad     = parseInt(lp, 10);
  const ld  = localStorage.getItem('v5-lama-dil');     if (ld)  state.lamaMaskDilation = parseInt(ld, 10);
  const lpa = localStorage.getItem('v5-lama-passes');  if (lpa) state.lamaPasses       = parseInt(lpa, 10);
  const ltx = localStorage.getItem('v5-lama-tex');     if (ltx) state.lamaTextureBlend = parseFloat(ltx);
  apiUI.key      = localStorage.getItem('v5-falai-key')     || '';
  apiUI.prompt   = localStorage.getItem('v5-api-prompt')    || '';
  apiUI.negative = localStorage.getItem('v5-api-negative')  || '';
} catch {}

// Background image slot — accepts drop + click, just like the main A slot
slotBgEl.addEventListener('click', () => {
  const picker = document.createElement('input');
  picker.type = 'file'; picker.accept = 'image/*';
  picker.addEventListener('change', () => {
    const f = picker.files?.[0];
    if (f) loadBackgroundFile(f);
  });
  picker.click();
});
slotBgEl.addEventListener('dragover', (e) => { e.preventDefault(); slotBgEl.classList.add('drop-target'); });
slotBgEl.addEventListener('dragleave', () => slotBgEl.classList.remove('drop-target'));
slotBgEl.addEventListener('drop', (e) => {
  e.preventDefault();
  slotBgEl.classList.remove('drop-target');
  const f = e.dataTransfer?.files?.[0];
  if (f) loadBackgroundFile(f);
});

async function loadBackgroundFile(file) {
  if (!file || !file.type.startsWith('image/')) return;
  const url = URL.createObjectURL(file);
  const img = new Image();
  img.src = url;
  await img.decode();
  state.bgImg = img;
  slotBgEl.innerHTML = '<span class="lab">background</span>';
  const im = document.createElement('img');
  im.src = url; slotBgEl.appendChild(im);
  refreshInpaintButton();
  await idbPut('bgImg', file);
}

function refreshInpaintButton() {
  if (!tp.inpaintBtn) return;
  if (!state.img || !regions.length) { tp.inpaintBtn.disabled = true; return; }
  switch (state.inpaintMethod) {
    case 'background': tp.inpaintBtn.disabled = !state.bgImg; break;
    case 'sd':         tp.inpaintBtn.disabled = true; break;
    default:           tp.inpaintBtn.disabled = false;
  }
}

// Dispatcher: picks the backend based on the current method selection.
async function runInpaint() {
  if (!state.img || !regions.length) return;
  if (lama.loading) { setInpaintStatus('already running'); return; }
  lama.loading = true;
  if (tp.inpaintBtn) tp.inpaintBtn.disabled = true;
  setProgress('inpaint-progress', null, { indeterminate: true });
  try {
    switch (state.inpaintMethod) {
      case 'lama':       await runLamaInpaint();       break;
      case 'background': await runBackgroundInpaint(); break;
      case 'api':        await runApiInpaint();        break;
      case 'sd':         setInpaintStatus('browser SD not yet wired — pick another method'); break;
      default:           setInpaintStatus(`unknown method: ${state.inpaintMethod}`);
    }
  } catch (err) {
    console.error('[inpaint] failed', err);
    setInpaintStatus(`failed: ${err.message || err}`);
  } finally {
    lama.loading = false;
    setProgress('inpaint-progress', null);
    refreshInpaintButton();
  }
}

async function ensureLamaSession() {
  if (lama.session) return;
  if (!lama.ort) {
    // Pure-WASM ORT build — no JSEP/WebGPU code at all. LaMa's Fourier
    // convolutions are flaky on the WebGPU EP (FFC Add ops fail dynamically),
    // so we use the WASM-only bundle to guarantee CPU execution regardless of
    // what `executionProviders` says. SAM keeps using its own webgpu-capable
    // ORT (bundled inside @huggingface/transformers).
    setInpaintStatus('loading ONNX runtime (WASM) …');
    lama.ort = await import(`${ORT_BASE}/dist/ort.wasm.min.mjs`);
    lama.ort.env.wasm.wasmPaths = `${ORT_BASE}/dist/`;
    // Run inference in a dedicated worker so the long LaMa session.run() calls
    // never freeze the main thread (Chrome flags "page unresponsive" otherwise
    // during multi-pass runs). This is ORT's built-in proxy mode.
    lama.ort.env.wasm.proxy = true;
  }
  if (!lama.buffer) {
    setInpaintStatus('fetching LaMa (~208 MB, first time only) …');
    lama.buffer = await fetchLamaModel();
  }
  setInpaintStatus('compiling LaMa (this can take ~10 s) …');
  await new Promise(r => setTimeout(r, 16));
  await createLamaSession(['wasm']);
  console.log('[LaMa] ready — EP:', lama.ep, 'inputs:', lama.session.inputNames, 'outputs:', lama.session.outputNames);
}

async function createLamaSession(eps) {
  try {
    lama.session = await lama.ort.InferenceSession.create(lama.buffer, {
      executionProviders: eps,
      graphOptimizationLevel: 'all',
    });
    lama.ep = eps[0];
  } catch (err) {
    if (eps[0] !== 'wasm') {
      console.warn('[LaMa] session create failed on', eps[0], '— falling back to WASM', err);
      await createLamaSession(['wasm']);
    } else {
      throw err;
    }
  }
}

// LaMa's Fourier convolutions occasionally fail on the WebGPU EP at run() time
// (e.g. "Can't perform binary op" inside FFC layers). When that happens we
// permanently switch this session to the WASM EP and retry.
async function lamaFallbackToWasm() {
  if (lama.ep === 'wasm') return;
  setInpaintStatus('WebGPU EP unstable for LaMa — switching to WASM (slower but reliable) …');
  await new Promise(r => setTimeout(r, 16));
  lama.session = null;
  await createLamaSession(['wasm']);
}

async function fetchLamaModel() {
  // Cache API so a page reload doesn't re-download. Falls through to the
  // browser HTTP cache if the Cache API write fails (private mode, etc).
  const cacheName = 'lama-onnx-v1';
  try {
    const cache = await caches.open(cacheName);
    const hit = await cache.match(LAMA_MODEL_URL);
    if (hit) {
      const buf = await hit.arrayBuffer();
      setInpaintStatus(`LaMa loaded from cache (${(buf.byteLength / 1e6).toFixed(0)} MB)`);
      return buf;
    }
  } catch {}
  const resp = await fetch(LAMA_MODEL_URL);
  if (!resp.ok) throw new Error(`LaMa download HTTP ${resp.status}`);
  const total = +(resp.headers.get('content-length') || 0);
  const reader = resp.body.getReader();
  const chunks = [];
  let received = 0;
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    const pctNum = total ? (received / total) * 100 : null;
    if (pctNum !== null) setProgress('inpaint-progress', pctNum);
    else setProgress('inpaint-progress', null, { indeterminate: true });
    setInpaintStatus(`downloading LaMa ${pctNum !== null ? pctNum.toFixed(1) + '%' : ''} (${(received / 1e6).toFixed(0)} MB)`);
  }
  const buf = new Uint8Array(received);
  let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; }
  try {
    const cache = await caches.open(cacheName);
    await cache.put(LAMA_MODEL_URL, new Response(buf.slice(), {
      headers: { 'content-type': 'application/octet-stream', 'content-length': String(received) },
    }));
  } catch (e) { console.warn('[LaMa] cache write failed', e); }
  return buf.buffer;
}

async function runLamaInpaint() {
  await ensureLamaSession();
  const W = regions[0].w, H = regions[0].h;
  const srcCanvas = new OffscreenCanvas(W, H);
  srcCanvas.getContext('2d').drawImage(state.img, 0, 0, W, H);
  const srcData = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const target = new Uint8ClampedArray(srcData);
  const cumMask = new Uint8Array(W * H);
  const t0 = performance.now();
  for (let r = 0; r < regions.length; r++) {
    const m = regions[r].data;
    for (let i = 0; i < W * H; i++) if (m[i]) cumMask[i] = 1;
    setProgress('inpaint-progress', (r / regions.length) * 100);
    setInpaintStatus(`LaMa region ${r + 1}/${regions.length} — running …`);
    await new Promise(r => setTimeout(r, 16));
    const inpaint = await lamaInfer(srcCanvas, cumMask, W, H);
    for (let i = 0; i < W * H; i++) {
      if (regions[r].data[i]) {
        const o = i * 4;
        target[o + 0] = inpaint[o + 0];
        target[o + 1] = inpaint[o + 1];
        target[o + 2] = inpaint[o + 2];
        target[o + 3] = 255;
      }
    }
    console.log(`[LaMa] region ${r + 1} done @ ${((performance.now() - t0)/1000).toFixed(1)}s`);
  }
  if (state.lamaTextureBlend > 0.001) {
    setInpaintStatus(`transferring paper texture …`);
    await new Promise(r => setTimeout(r, 16));
    applyTextureTransfer(target, srcCanvas, cumMask, W, H, state.lamaTextureBlend);
  }
  await uploadInpaintTarget(target, W, H);
  setInpaintStatus(`LaMa inpaint ready (${regions.length} regions, ${W}×${H}, ${((performance.now() - t0)/1000).toFixed(1)}s)`);
}

// Paper-grain transfer: LaMa output is smooth where the original had brush
// texture and paper noise. We compute the source's high-frequency detail
// (source − blurred source), and for each masked pixel we OVERLAY detail
// sampled from a non-masked patch of the source. Pseudo-random per-pixel
// offsets break the obvious tile pattern. Result: the inpainted area picks
// up the same paper grain magnitude/style as the rest of the painting.
function applyTextureTransfer(target, srcCanvas, mask, W, H, strength) {
  // 1. Blurred copy of A (low-pass)
  const blurred = new OffscreenCanvas(W, H);
  const bctx = blurred.getContext('2d');
  bctx.filter = 'blur(3px)';
  bctx.drawImage(srcCanvas, 0, 0);
  bctx.filter = 'none';
  const srcData  = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const blurData = bctx.getImageData(0, 0, W, H).data;

  // 2. Find a 96×96 patch that's entirely non-masked. Scan the borders +
  // mid-edges. If nothing clean is found, fall back to (0, 0) — its detail
  // may include object content but the grain transfer is still better than
  // pure LaMa smoothness.
  const patch = 96;
  const candidates = [];
  for (let cy = 0; cy <= H - patch; cy += 48) {
    for (let cx = 0; cx <= W - patch; cx += 48) candidates.push([cx, cy]);
  }
  let cornerX = 0, cornerY = 0;
  for (const [cx, cy] of candidates) {
    let blocked = false;
    for (let y = 0; y < patch && !blocked; y += 12) {
      for (let x = 0; x < patch && !blocked; x += 12) {
        if (mask[(cy + y) * W + cx + x]) blocked = true;
      }
    }
    if (!blocked) { cornerX = cx; cornerY = cy; break; }
  }

  // 3. For each masked pixel, sample detail at a pseudo-random offset into
  // the chosen patch and add it to the LaMa result.
  const hash2d = (x, y) => {
    let h = (x | 0) * 374761393 + (y | 0) * 668265263;
    h = (h ^ (h >>> 13)) * 1274126177;
    return ((h ^ (h >>> 16)) >>> 0);
  };
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const i = y * W + x;
      if (!mask[i]) continue;
      const rng = hash2d(x, y);
      const tx = cornerX + (rng % patch);
      const ty = cornerY + ((rng >>> 8) % patch);
      const sIdx = (ty * W + tx) * 4;
      const tIdx = i * 4;
      const dr = srcData[sIdx + 0] - blurData[sIdx + 0];
      const dg = srcData[sIdx + 1] - blurData[sIdx + 1];
      const db = srcData[sIdx + 2] - blurData[sIdx + 2];
      target[tIdx + 0] = Math.max(0, Math.min(255, target[tIdx + 0] + dr * strength));
      target[tIdx + 1] = Math.max(0, Math.min(255, target[tIdx + 1] + dg * strength));
      target[tIdx + 2] = Math.max(0, Math.min(255, target[tIdx + 2] + db * strength));
    }
  }
}

// Manual: user supplies a pre-inpainted image (e.g. from Photoshop generative
// fill). We resize it to the SAM-encoded resolution and use it directly as
// the dissolve target. No per-region work needed.
async function runBackgroundInpaint() {
  if (!state.bgImg) { setInpaintStatus('upload a background image first'); return; }
  const W = regions[0].w, H = regions[0].h;
  setInpaintStatus(`pasting background image at ${W}×${H} …`);
  await new Promise(r => setTimeout(r, 16));
  const c = new OffscreenCanvas(W, H);
  const cctx = c.getContext('2d');
  cctx.imageSmoothingQuality = 'high';
  cctx.drawImage(state.bgImg, 0, 0, W, H);
  const bm = await createImageBitmap(c);
  await uploadInpaintBitmap(bm, W, H);
  setInpaintStatus(`background image applied (${W}×${H})`);
}

// fal.ai: one HTTP call per region with cumulative mask. Standard schema is
// { image_url, mask_url, prompt } with the result at result.images[0].url.
// fal.ai endpoints commonly have a hard input cap around 1024px; we resize
// payloads to fit and composite results back at full source resolution.
async function runApiInpaint() {
  const key = (apiUI.key || '').trim();
  if (!key) { setInpaintStatus('paste a fal.ai key first'); return; }
  const model = (apiUI.model || 'fal-ai/flux-pro/v1/fill').trim();
  const prompt = (apiUI.prompt || '').trim() || 'inpaint, seamless background';
  const negativePrompt = (apiUI.negative || '').trim();
  const W = regions[0].w, H = regions[0].h;
  const API_MAX = 1024;
  const srcCanvas = new OffscreenCanvas(W, H);
  srcCanvas.getContext('2d').drawImage(state.img, 0, 0, W, H);
  const srcData = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const srcDataUrl = await canvasToDataUrl(srcCanvas, API_MAX);
  const target = new Uint8ClampedArray(srcData);
  const cumMask = new Uint8Array(W * H);
  const t0 = performance.now();
  for (let r = 0; r < regions.length; r++) {
    const m = regions[r].data;
    for (let i = 0; i < W * H; i++) if (m[i]) cumMask[i] = 1;
    setInpaintStatus(`fal.ai region ${r + 1}/${regions.length} — calling …`);
    await new Promise(r => setTimeout(r, 16));
    // Render the mask as a black/white image, then encode as data URL
    const maskCanvas = new OffscreenCanvas(W, H);
    const mImg = new ImageData(W, H);
    for (let i = 0; i < W * H; i++) {
      const v = cumMask[i] ? 255 : 0;
      mImg.data[i * 4 + 0] = v;
      mImg.data[i * 4 + 1] = v;
      mImg.data[i * 4 + 2] = v;
      mImg.data[i * 4 + 3] = 255;
    }
    maskCanvas.getContext('2d').putImageData(mImg, 0, 0);
    const maskDataUrl = await canvasToDataUrl(maskCanvas, API_MAX);
    // Call fal.ai
    const resp = await fetch(`https://fal.run/${model}`, {
      method: 'POST',
      headers: {
        'Authorization': `Key ${key}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        image_url: srcDataUrl,
        mask_url: maskDataUrl,
        prompt: prompt,
        ...(negativePrompt ? { negative_prompt: negativePrompt } : {}),
        // The legacy fal-ai/inpaint endpoint requires a model_name field that
        // the FLUX endpoints don't. Auto-inject a sensible SDXL checkpoint so
        // users who manually pick that endpoint don't hit a 422.
        ...(model === 'fal-ai/inpaint'
          ? { model_name: 'stabilityai/stable-diffusion-xl-1.0-inpainting-0.1' }
          : {}),
      }),
    });
    if (!resp.ok) {
      const txt = await resp.text().catch(() => '');
      throw new Error(`fal.ai ${resp.status}: ${txt.slice(0, 200)}`);
    }
    const result = await resp.json();
    const outUrl = result.images?.[0]?.url || result.image?.url || result.url;
    if (!outUrl) throw new Error('no image url in fal.ai response');
    setInpaintStatus(`fal.ai region ${r + 1}/${regions.length} — fetching result …`);
    const imgResp = await fetch(outUrl);
    if (!imgResp.ok) throw new Error(`result fetch ${imgResp.status}`);
    const blob = await imgResp.blob();
    const bm = await createImageBitmap(blob);
    // Composite the API result at full source resolution
    const apiCanvas = new OffscreenCanvas(W, H);
    const actx = apiCanvas.getContext('2d');
    actx.imageSmoothingQuality = 'high';
    actx.drawImage(bm, 0, 0, W, H);
    bm.close();
    const apiData = actx.getImageData(0, 0, W, H).data;
    for (let i = 0; i < W * H; i++) {
      if (regions[r].data[i]) {
        const o = i * 4;
        target[o + 0] = apiData[o + 0];
        target[o + 1] = apiData[o + 1];
        target[o + 2] = apiData[o + 2];
        target[o + 3] = 255;
      }
    }
    console.log(`[fal.ai] region ${r + 1} done @ ${((performance.now() - t0)/1000).toFixed(1)}s`);
  }
  await uploadInpaintTarget(target, W, H);
  setInpaintStatus(`fal.ai inpaint ready (${regions.length} regions, ${((performance.now() - t0)/1000).toFixed(1)}s)`);
}

// Common: upload a Uint8ClampedArray target as texInpaint
async function uploadInpaintTarget(rgbaData, W, H) {
  const c = new OffscreenCanvas(W, H);
  c.getContext('2d').putImageData(new ImageData(rgbaData, W, H), 0, 0);
  const bm = await createImageBitmap(c);
  await uploadInpaintBitmap(bm, W, H);
}

async function uploadInpaintBitmap(bm, W, H) {
  const tex = device.createTexture({
    label: 'tex-inpaint',
    size: [W, H, 1], format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  device.queue.copyExternalImageToTexture({ source: bm }, { texture: tex }, [W, H, 1]);
  bm.close();
  texInpaint = tex;
  state.inpaintValid = true;
  bindGroup = makeBindGroup();
}

async function canvasToDataUrl(canvas, maxSize) {
  let c = canvas;
  if (maxSize && (canvas.width > maxSize || canvas.height > maxSize)) {
    const scale = maxSize / Math.max(canvas.width, canvas.height);
    const nw = Math.round(canvas.width * scale);
    const nh = Math.round(canvas.height * scale);
    c = new OffscreenCanvas(nw, nh);
    const cx = c.getContext('2d');
    cx.imageSmoothingQuality = 'high';
    cx.drawImage(canvas, 0, 0, nw, nh);
  }
  const blob = await c.convertToBlob({ type: 'image/png' });
  return await new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(r.result);
    r.onerror = reject;
    r.readAsDataURL(blob);
  });
}

// LaMa inference defaults — runtime values come from state.* and are
// adjustable via the LaMa method panel sliders. Constants kept here only as
// fallback hints; the slider values are the source of truth at run time.

// Compute a square bbox around the on pixels of `mask`, padded and clamped
// to the image bounds. Returns null for an empty mask.
function computePatchBbox(mask, W, H, padding, jx = 0, jy = 0) {
  let xmin = W, ymin = H, xmax = -1, ymax = -1;
  for (let y = 0; y < H; y++) {
    const row = y * W;
    for (let x = 0; x < W; x++) {
      if (mask[row + x]) {
        if (x < xmin) xmin = x;
        if (x > xmax) xmax = x;
        if (y < ymin) ymin = y;
        if (y > ymax) ymax = y;
      }
    }
  }
  if (xmax < 0) return null;
  xmin = Math.max(0, xmin - padding);
  ymin = Math.max(0, ymin - padding);
  xmax = Math.min(W - 1, xmax + padding);
  ymax = Math.min(H - 1, ymax + padding);
  const size = Math.max(xmax - xmin + 1, ymax - ymin + 1);
  const cx = (xmin + xmax) / 2 + jx, cy = (ymin + ymax) / 2 + jy;
  let x0 = Math.round(cx - size / 2);
  let y0 = Math.round(cy - size / 2);
  let x1 = x0 + size;
  let y1 = y0 + size;
  // Shift to fit inside image
  if (x0 < 0) { x1 -= x0; x0 = 0; }
  if (y0 < 0) { y1 -= y0; y0 = 0; }
  if (x1 > W) { x0 -= (x1 - W); x1 = W; }
  if (y1 > H) { y0 -= (y1 - H); y1 = H; }
  x0 = Math.max(0, x0); y0 = Math.max(0, y0);
  return { x0, y0, w: x1 - x0, h: y1 - y0 };
}

// Cheap morphological dilation via canvas blur + threshold. Way faster than
// a JS nested loop for radius > 2, and quality is plenty for SAM-edge fudge.
function dilateMask(mask, W, H, radius) {
  if (radius <= 0) return mask;
  const a = new OffscreenCanvas(W, H);
  const aImg = new ImageData(W, H);
  for (let i = 0; i < W * H; i++) {
    const v = mask[i] ? 255 : 0;
    aImg.data[i * 4 + 0] = v;
    aImg.data[i * 4 + 3] = 255;
  }
  a.getContext('2d').putImageData(aImg, 0, 0);
  const b = new OffscreenCanvas(W, H);
  const bctx = b.getContext('2d');
  bctx.filter = `blur(${radius}px)`;
  bctx.drawImage(a, 0, 0);
  bctx.filter = 'none';
  const blurred = bctx.getImageData(0, 0, W, H).data;
  const out = new Uint8Array(W * H);
  for (let i = 0; i < W * H; i++) out[i] = blurred[i * 4] > 24 ? 1 : 0;
  return out;
}

// LaMa inference at the full source resolution by cropping a focused square
// patch around the mask's bbox. The patch (cropped → resized to 512² → LaMa
// → upscaled back) is the only thing that round-trips through 512px, so the
// rest of the image stays pixel-perfect.
async function lamaInfer(srcCanvas, maskBinaryFull, W, H) {
  const srcPx = srcCanvas.getContext('2d').getImageData(0, 0, W, H).data;
  const baseline = new Uint8ClampedArray(srcPx); // start from A, only overwrite mask area
  const maskDil = dilateMask(maskBinaryFull, W, H, state.lamaMaskDilation);
  // For each pass we recompute the bbox with a jittered center so LaMa sees
  // a slightly different slice of surrounding pixels each time. Average the
  // outputs at the masked pixels to reduce single-pass texture repetition.
  const passes = Math.max(1, state.lamaPasses | 0);
  // Per-pixel accumulator (3 channels + count) for in-mask averaging
  const acc = new Float32Array(W * H * 4);
  for (let pass = 0; pass < passes; pass++) {
    if (pass > 0) await new Promise(r => setTimeout(r, 16));
    const jitter = passes > 1 ? Math.round(state.lamaPatchPad * 0.4) : 0;
    const jx = passes > 1 ? Math.round((Math.random() - 0.5) * jitter * 2) : 0;
    const jy = passes > 1 ? Math.round((Math.random() - 0.5) * jitter * 2) : 0;
    const bbox = computePatchBbox(maskDil, W, H, state.lamaPatchPad, jx, jy);
    if (!bbox) return baseline;
    const patchImg = new OffscreenCanvas(bbox.w, bbox.h);
    patchImg.getContext('2d').drawImage(
      srcCanvas, bbox.x0, bbox.y0, bbox.w, bbox.h, 0, 0, bbox.w, bbox.h
    );
    const patchMask = new Uint8Array(bbox.w * bbox.h);
    for (let y = 0; y < bbox.h; y++) {
      for (let x = 0; x < bbox.w; x++) {
        patchMask[y * bbox.w + x] = maskDil[(bbox.y0 + y) * W + (bbox.x0 + x)];
      }
    }
    const patchOut = await lamaInferPatch(patchImg, patchMask, bbox.w, bbox.h);
    // Accumulate this pass's pixels at the masked locations
    for (let y = 0; y < bbox.h; y++) {
      const globalRow = (bbox.y0 + y) * W;
      const localRow  = y * bbox.w;
      for (let x = 0; x < bbox.w; x++) {
        if (maskDil[globalRow + (bbox.x0 + x)]) {
          const go = (globalRow + (bbox.x0 + x)) * 4;
          const lo = (localRow + x) * 4;
          acc[go + 0] += patchOut[lo + 0];
          acc[go + 1] += patchOut[lo + 1];
          acc[go + 2] += patchOut[lo + 2];
          acc[go + 3] += 1; // pass count for this pixel
        }
      }
    }
  }
  // Write averaged pixels into the baseline
  for (let i = 0; i < W * H; i++) {
    if (maskDil[i]) {
      const o = i * 4;
      const n = acc[o + 3] || 1;
      baseline[o + 0] = Math.max(0, Math.min(255, acc[o + 0] / n));
      baseline[o + 1] = Math.max(0, Math.min(255, acc[o + 1] / n));
      baseline[o + 2] = Math.max(0, Math.min(255, acc[o + 2] / n));
      baseline[o + 3] = 255;
    }
  }
  return baseline;
}

// Run LaMa on a single patch of arbitrary size, returning RGBA at the patch's
// native size. Internal 512×512 trip is unavoidable (model is fixed-shape).
async function lamaInferPatch(patchCanvas, patchMask, pw, ph) {
  const S = LAMA_INPUT_SIZE;
  // Resize image to 512×512
  const smallImg = new OffscreenCanvas(S, S);
  const sictx = smallImg.getContext('2d');
  sictx.imageSmoothingQuality = 'high';
  sictx.drawImage(patchCanvas, 0, 0, S, S);
  const imgPx = sictx.getImageData(0, 0, S, S).data;
  // Resize mask to 512×512
  const maskFullC = new OffscreenCanvas(pw, ph);
  const mFullImg = new ImageData(pw, ph);
  for (let i = 0; i < pw * ph; i++) {
    const v = patchMask[i] ? 255 : 0;
    mFullImg.data[i * 4 + 0] = v;
    mFullImg.data[i * 4 + 1] = v;
    mFullImg.data[i * 4 + 2] = v;
    mFullImg.data[i * 4 + 3] = 255;
  }
  maskFullC.getContext('2d').putImageData(mFullImg, 0, 0);
  const smallMask = new OffscreenCanvas(S, S);
  smallMask.getContext('2d').drawImage(maskFullC, 0, 0, S, S);
  const maskPx = smallMask.getContext('2d').getImageData(0, 0, S, S).data;
  // Pack tensors
  const SP = S * S;
  const imgArr = new Float32Array(3 * SP);
  for (let i = 0; i < SP; i++) {
    imgArr[0 * SP + i] = imgPx[i * 4 + 0] / 255;
    imgArr[1 * SP + i] = imgPx[i * 4 + 1] / 255;
    imgArr[2 * SP + i] = imgPx[i * 4 + 2] / 255;
  }
  const maskArr = new Float32Array(SP);
  for (let i = 0; i < SP; i++) maskArr[i] = maskPx[i * 4] > 127 ? 1.0 : 0.0;
  // Bind by name (with positional fallback)
  const inNames = lama.session.inputNames;
  const imgT  = new lama.ort.Tensor('float32', imgArr,  [1, 3, S, S]);
  const maskT = new lama.ort.Tensor('float32', maskArr, [1, 1, S, S]);
  const feeds = (inNames.includes('image') && inNames.includes('mask'))
    ? { image: imgT, mask: maskT }
    : { [inNames[0]]: imgT, [inNames[1]]: maskT };
  let results;
  try {
    results = await lama.session.run(feeds);
  } catch (err) {
    // Should only fire if someone wires a WebGPU-capable EP back in; harmless.
    if (lama.ep !== 'wasm') {
      console.warn('[LaMa] run failed on', lama.ep, '— switching to WASM and retrying', err);
      await lamaFallbackToWasm();
      const imgT2  = new lama.ort.Tensor('float32', imgArr,  [1, 3, S, S]);
      const maskT2 = new lama.ort.Tensor('float32', maskArr, [1, 1, S, S]);
      const inN2 = lama.session.inputNames;
      const feeds2 = (inN2.includes('image') && inN2.includes('mask'))
        ? { image: imgT2, mask: maskT2 }
        : { [inN2[0]]: imgT2, [inN2[1]]: maskT2 };
      results = await lama.session.run(feeds2);
    } else { throw err; }
  }
  const outArr = results[lama.session.outputNames[0]].data;
  // Build 512×512 output image
  const outImg = new ImageData(S, S);
  for (let i = 0; i < SP; i++) {
    outImg.data[i * 4 + 0] = Math.max(0, Math.min(255, outArr[0 * SP + i]));
    outImg.data[i * 4 + 1] = Math.max(0, Math.min(255, outArr[1 * SP + i]));
    outImg.data[i * 4 + 2] = Math.max(0, Math.min(255, outArr[2 * SP + i]));
    outImg.data[i * 4 + 3] = 255;
  }
  // Upscale back to patch size
  const outSmall = new OffscreenCanvas(S, S);
  outSmall.getContext('2d').putImageData(outImg, 0, 0);
  const outFull = new OffscreenCanvas(pw, ph);
  const ofctx = outFull.getContext('2d');
  ofctx.imageSmoothingQuality = 'high';
  ofctx.drawImage(outSmall, 0, 0, pw, ph);
  return ofctx.getImageData(0, 0, pw, ph).data;
}

// ============================================================================
// Panel toggle
// ============================================================================
document.getElementById('toggle-panel').addEventListener('click', () => {
  document.body.classList.toggle('minimized');
});

// ============================================================================
// Library UI — thumbnails of every image ever loaded. Click = reload as A;
// right-click = delete from library (and IDB).
// ============================================================================
const libGridEl = document.getElementById('library-grid');
const _libThumbUrls = new Map();
let _libCache = [];

function libRevokeAll() {
  for (const url of _libThumbUrls.values()) URL.revokeObjectURL(url);
  _libThumbUrls.clear();
}

async function renderLibrary() {
  _libCache = await libList();
  libRevokeAll();
  libGridEl.innerHTML = '';
  if (_libCache.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'library-empty';
    empty.textContent = 'drop or pick an image — it shows here';
    libGridEl.appendChild(empty);
    return;
  }
  for (const entry of _libCache) {
    const url = URL.createObjectURL(entry.thumb);
    _libThumbUrls.set(entry.id, url);
    const tile = document.createElement('div');
    tile.className = 'library-thumb' + (entry.id === state._libIdCurrent ? ' active' : '');
    tile.title = `${entry.name}\nclick to load · right-click to delete`;
    tile.dataset.libId = entry.id;
    const img = document.createElement('img');
    img.src = url;
    tile.appendChild(img);
    libGridEl.appendChild(tile);
  }
}

libGridEl.addEventListener('click', async (e) => {
  const tile = e.target.closest('.library-thumb');
  if (!tile) return;
  const id = parseInt(tile.dataset.libId, 10);
  const entry = _libCache.find(x => x.id === id);
  if (!entry) return;
  state._libIdCurrent = id;
  await idbPut('imageA', entry.blob);
  await idbDelete('regions');
  const img = await blobToImage(entry.blob);
  await uploadImage(img);
  renderLibrary();
});

libGridEl.addEventListener('contextmenu', async (e) => {
  const tile = e.target.closest('.library-thumb');
  if (!tile) return;
  e.preventDefault();
  const id = parseInt(tile.dataset.libId, 10);
  if (!confirm('Delete this image from the library?')) return;
  await libDelete(id);
  if (state._libIdCurrent === id) state._libIdCurrent = null;
  renderLibrary();
});

// ============================================================================
// Startup restore — bring back the last-loaded image and background, render
// the library. Runs after all UI handlers are wired so the restored state
// flows through the same paths a fresh upload would.
// ============================================================================
async function restoreSession() {
  console.log('[session] restoring …');
  try {
    const lastBlob = await idbGet('imageA');
    if (lastBlob) {
      console.log('[session] restoring imageA');
      const img = await blobToImage(lastBlob);
      await uploadImage(img);
      // uploadImage clears regions — restore saved ones now if the dimensions match
      const W = sam.encodedImgRef ? (sam.encodedImgRef.naturalWidth || sam.encodedImgRef.width) : img.naturalWidth;
      const H = sam.encodedImgRef ? (sam.encodedImgRef.naturalHeight || sam.encodedImgRef.height) : img.naturalHeight;
      const saved = await idbGet('regions');
      if (saved && Array.isArray(saved) && saved.length) {
        let restored = 0;
        for (const r of saved) {
          const data = r.data instanceof Uint8Array ? r.data : new Uint8Array(r.data);
          if (data.length !== r.w * r.h) { console.warn('[regions] data length mismatch, skipping', r); continue; }
          const mc = new OffscreenCanvas(r.w, r.h);
          const mctx = mc.getContext('2d');
          const mim = new ImageData(r.w, r.h);
          for (let i = 0, j = 0; i < data.length; i++, j += 4) {
            mim.data[j + 0] = 255; mim.data[j + 1] = 255; mim.data[j + 2] = 255;
            mim.data[j + 3] = data[i] ? 180 : 0;
          }
          mctx.putImageData(mim, 0, 0);
          regions.push({ id: r.id, name: r.name, w: r.w, h: r.h, data, maskCanvas: mc });
          restored++;
        }
        if (restored) {
          rebuildRegionsTexture();
          refreshRegionsUI();
          console.log(`[session] restored ${restored} region(s)`);
        }
      }
      console.log('[session] imageA restored OK');
    } else {
      console.log('[session] no imageA in IDB');
    }
    const bgBlob = await idbGet('bgImg');
    if (bgBlob) {
      console.log('[session] restoring bgImg');
      const img = await blobToImage(bgBlob);
      state.bgImg = img;
      slotBgEl.innerHTML = '<span class="lab">background</span>';
      const im = document.createElement('img');
      im.src = img.src; slotBgEl.appendChild(im);
      refreshInpaintButton();
    }
    const vBlob = await idbGet('videoT');
    if (vBlob) {
      console.log('[session] restoring videoT');
      await loadVideoToT(vBlob);
    }
  } catch (err) { console.error('[session] restore failed', err); }
  renderLibrary();
}
// ============================================================================
// Canvas zoom / pan — wheel scrolls zoom (anchored to cursor), drag (when not
// in segment mode) pans, double-click resets. The overlay tracks automatically
// because syncOverlaySize() uses canvas.getBoundingClientRect() which already
// reflects the CSS transform.
// ============================================================================
const stage = document.getElementById('stage');
const zoomState = { zoom: 1, panX: 0, panY: 0 };

function applyCanvasTransform() {
  canvas.style.transformOrigin = '0 0';
  canvas.style.transform = `translate(${zoomState.panX}px, ${zoomState.panY}px) scale(${zoomState.zoom})`;
  drawOverlay();
}
function resetZoom() {
  zoomState.zoom = 1; zoomState.panX = 0; zoomState.panY = 0;
  applyCanvasTransform();
}

stage.addEventListener('wheel', (e) => {
  if (canvas.classList.contains('empty')) return;
  e.preventDefault();
  const rect = canvas.getBoundingClientRect();
  const dx = e.clientX - rect.left;
  const dy = e.clientY - rect.top;
  const oldZ = zoomState.zoom;
  const factor = e.deltaY < 0 ? 1.15 : 1 / 1.15;
  const newZ = Math.max(0.25, Math.min(12, oldZ * factor));
  if (newZ === oldZ) return;
  // Keep the canvas pixel under the cursor anchored
  zoomState.panX += dx * (1 - newZ / oldZ);
  zoomState.panY += dy * (1 - newZ / oldZ);
  zoomState.zoom = newZ;
  applyCanvasTransform();
}, { passive: false });

// Pan: left-drag when NOT in segment mode (so SAM clicks aren't hijacked), or
// middle-button drag any time. Space+drag also pans.
let isPanning = false;
let spaceHeld = false;
const panStart = { x: 0, y: 0, panX: 0, panY: 0 };
document.addEventListener('keydown', (e) => {
  if (e.code === 'Space' && !e.target.matches('input, textarea')) spaceHeld = true;
});
document.addEventListener('keyup', (e) => { if (e.code === 'Space') spaceHeld = false; });

stage.addEventListener('mousedown', (e) => {
  if (canvas.classList.contains('empty')) return;
  const isMiddle = e.button === 1;
  const isLeftWithModifier = e.button === 0 && (spaceHeld || (!sam.segmentMode && e.target !== overlay));
  if (!(isMiddle || isLeftWithModifier)) return;
  isPanning = true;
  panStart.x = e.clientX; panStart.y = e.clientY;
  panStart.panX = zoomState.panX; panStart.panY = zoomState.panY;
  stage.style.cursor = 'grabbing';
  e.preventDefault();
});
document.addEventListener('mousemove', (e) => {
  if (!isPanning) return;
  zoomState.panX = panStart.panX + (e.clientX - panStart.x);
  zoomState.panY = panStart.panY + (e.clientY - panStart.y);
  applyCanvasTransform();
});
document.addEventListener('mouseup', () => {
  if (isPanning) { isPanning = false; stage.style.cursor = ''; }
});
stage.addEventListener('dblclick', (e) => {
  if (sam.segmentMode && e.target === overlay) return;
  resetZoom();
});

// ============================================================================
// Tweakpane setup — 4 small panes anchored to their containers in index.html
// so custom DOM blocks (regions list, bg slot, progress bars) can sit in the
// right workflow order between them.
// ============================================================================

// Playback pane
tp.panePlay = new Pane({ container: document.getElementById('tp-playback'), title: 'Playback' });
tp.panePlay.registerPlugin(EssentialsPlugin);
tp.panePlay.addBinding(state, 't',        { min: 0, max: 1,  step: 0.001, label: 'progress' });
tp.panePlay.addBinding(state, 'duration', { min: 1, max: 60, step: 0.5,   label: 'duration (s)' });
tp.panePlay.addBinding(state, 'softness', { min: 0, max: 0.6, step: 0.005, label: 'softness' });
tp.panePlay.addBinding(state, 'stagger',  { min: 0, max: 1,   step: 0.01,  label: 'stagger' })
  .on('change', () => rebuildRegionsTexture());
const fVideo = tp.panePlay.addFolder({ title: 'Video mask (T slot)', expanded: false });
fVideo.addBinding(state, 'videoStrength', { min: 0, max: 1, step: 0.01, label: 'strength' });
fVideo.addBinding(state, 'videoInvert',   { label: 'invert (dark first)' });
// Play / Restart / Loop as a 3-cell button grid for compactness
const playGrid = tp.panePlay.addBlade({
  view: 'buttongrid', size: [3, 1],
  cells: (x) => ({ title: ['▶ Play', '⟲ Restart', 'Loop'][x] }),
});
playGrid.on('click', (e) => {
  if (e.index[0] === 0) setPlaying(!state.playing);
  if (e.index[0] === 1) setT(0);
  if (e.index[0] === 2) { state.loop = !state.loop; }
});
tp.panePlay.addBinding(state, 'loop', { label: 'loop on end' });

// Segment pane (SAM)
tp.paneSeg = new Pane({ container: document.getElementById('tp-segment'), title: 'Segment (SAM)' });
tp.samStatusBinding = tp.paneSeg.addBinding(samUI, 'status', { readonly: true, label: 'status' });
tp.paneSeg.addBinding(state, 'samModelId', {
  label: 'model',
  options: SAM_MODELS,
}).on('change', () => {
  try { localStorage.setItem('v5-sam-model', state.samModelId); } catch {}
  // Switching models invalidates the loaded model — user has to hit Load again
  if (sam.loadedModelId && sam.loadedModelId !== state.samModelId) {
    setStatus(`model changed — click 'Load SAM' to switch`);
  }
});
tp.paneSeg.addButton({ title: '1. Load SAM model' }).on('click', () => samLoadModel());
tp.paneSeg.addButton({ title: '2. Encode image' }).on('click', () => samEncodeImage());
tp.paneSeg.addBinding(samUI, 'mode', { label: '3. segment mode' }).on('change', (e) => setSegmentMode(e.value));
tp.paneSeg.addButton({ title: '↶ Undo last point' }).on('click', () => {
  if (!sam.points.length) { setStatus('no points to undo'); return; }
  sam.points.pop();
  if (sam.points.length === 0) {
    sam.maskCanvas = null; drawOverlay();
    setStatus('all points removed — click to start');
  } else {
    drawOverlay();
    runSamInference();
  }
});
tp.paneSeg.addButton({ title: '+ Save mask as region' }).on('click', () => saveCurrentMaskAsRegion());
tp.paneSeg.addButton({ title: 'reset current points' }).on('click', () => samClearMask());

// Mask edit — grow/shrink the active mask, or paint into it manually
const fMaskEdit = tp.paneSeg.addFolder({ title: 'Mask edit', expanded: false });
fMaskEdit.addBinding(growUI, 'amount', { min: 1, max: 64, step: 1, label: 'grow amount (px)' });
fMaskEdit.addButton({ title: '↗ Grow current mask' })
  .on('click', () => applyMaskMorphology(growUI.amount, 'grow'));
fMaskEdit.addButton({ title: '↙ Shrink current mask' })
  .on('click', () => applyMaskMorphology(growUI.amount, 'shrink'));
fMaskEdit.addBlade({ view: 'separator' });
fMaskEdit.addBinding(brushUI, 'mode', { label: 'brush mode (drag to paint)' });
fMaskEdit.addBinding(brushUI, 'size', { min: 5, max: 200, step: 1, label: 'brush size (px)' });
fMaskEdit.addBinding(brushUI, 'action', {
  label: 'brush action',
  options: { 'include (add)': 'include', 'exclude (cut)': 'exclude' },
});

// Inpaint pane
tp.paneInpaint = new Pane({ container: document.getElementById('tp-inpaint'), title: 'Inpaint' });
tp.paneInpaint.registerPlugin(EssentialsPlugin);
inpaintUI.method = state.inpaintMethod;
tp.inpaintMethodBinding = tp.paneInpaint.addBinding(inpaintUI, 'method', {
  label: 'method',
  options: {
    'LaMa (offline)':          'lama',
    'Background image (manual)':'background',
    'fal.ai API (hosted)':     'api',
    'SD (browser, WIP)':       'sd',
  },
});
tp.inpaintMethodBinding.on('change', (e) => {
  state.inpaintMethod = e.value;
  syncMethodVisibility();
  refreshInpaintButton();
});
tp.inpaintStatusBinding = tp.paneInpaint.addBinding(inpaintUI, 'status', { readonly: true, label: 'status' });

// LaMa parameters subfolder
tp.fLama = tp.paneInpaint.addFolder({ title: 'LaMa parameters', expanded: true });
tp.fLama.addBinding(state, 'lamaPatchPad',     { min: 32, max: 512, step: 8, label: 'context (px)' })
  .on('change', () => { try { localStorage.setItem('v5-lama-pad', String(state.lamaPatchPad)); } catch {} });
tp.fLama.addBinding(state, 'lamaMaskDilation', { min: 0,  max: 48,  step: 1, label: 'mask grow (px)' })
  .on('change', () => { try { localStorage.setItem('v5-lama-dil', String(state.lamaMaskDilation)); } catch {} });
tp.fLama.addBinding(state, 'lamaPasses',       { min: 1,  max: 4,   step: 1, label: 'passes' })
  .on('change', () => { try { localStorage.setItem('v5-lama-passes', String(state.lamaPasses)); } catch {} });
tp.fLama.addBinding(state, 'lamaTextureBlend', { min: 0,  max: 1.5, step: 0.05, label: 'texture transfer' })
  .on('change', () => { try { localStorage.setItem('v5-lama-tex', String(state.lamaTextureBlend)); } catch {} });

// fal.ai parameters subfolder
tp.fApi = tp.paneInpaint.addFolder({ title: 'fal.ai parameters', expanded: false });
tp.fApi.addBinding(apiUI, 'key',      { label: 'API key' })
  .on('change', () => { try { localStorage.setItem('v5-falai-key', apiUI.key); } catch {} });
tp.fApi.addBinding(apiUI, 'model',    { label: 'model' });
tp.fApi.addBinding(apiUI, 'prompt',   { label: 'prompt' })
  .on('change', () => { try { localStorage.setItem('v5-api-prompt', apiUI.prompt); } catch {} });
tp.fApi.addBinding(apiUI, 'negative', { label: 'negative' })
  .on('change', () => { try { localStorage.setItem('v5-api-negative', apiUI.negative); } catch {} });

tp.paneInpaint.addBlade({ view: 'separator' });
tp.inpaintBtn = tp.paneInpaint.addButton({ title: 'Generate inpaint' });
tp.inpaintBtn.on('click', () => runInpaint());

function syncMethodVisibility() {
  tp.fLama.hidden = state.inpaintMethod !== 'lama';
  tp.fApi.hidden  = state.inpaintMethod !== 'api';
  bgSlotWrap.classList.toggle('hidden', state.inpaintMethod !== 'background');
}
syncMethodVisibility();

// Record pane
tp.paneRec = new Pane({ container: document.getElementById('tp-record'), title: 'Record' });
tp.paneRec.addBinding(recState, 'fps', { min: 10, max: 60, step: 1, label: 'fps' });
tp.paneRec.addBinding(recState, 'sizeMode', {
  label: 'size',
  options: { 'native': 'src', '1920 wide': '1920', '1280 wide': '1280', '1024 wide': '1024' },
});
tp.recBtn = tp.paneRec.addButton({ title: 'Record video' });
tp.recBtn.on('click', () => startRecording().catch(err => {
  console.error('[record] failed', err);
  setRecBtnLabel(`failed: ${(err.message || err).toString().slice(0, 60)}`);
  setTimeout(() => { setRecBtnLabel('Record video'); refreshRecBtn(); }, 4000);
  recState.recording = false;
}));

// Persistent expansion state per pane (Tweakpane saves to its own internal
// state but doesn't persist across reloads by default — wire localStorage).
[tp.panePlay, tp.paneSeg, tp.paneInpaint, tp.paneRec].forEach((pane, i) => {
  const key = `v5-pane${i}`;
  try { const s = localStorage.getItem(key); if (s) pane.importState(JSON.parse(s)); } catch {}
  pane.on('change', () => { try { localStorage.setItem(key, JSON.stringify(pane.exportState())); } catch {} });
});

// Now restore session (image + bg image) after Tweakpane exists, so any UI
// updates triggered by uploadImage land on real bindings.
restoreSession();

// Hook the Clear button to also drop the persisted "last loaded" key so it
// doesn't come back on refresh (but keep library entries — those are
// explicitly the history).
document.getElementById('clear').addEventListener('click', () => {
  idbDelete('imageA');
  idbDelete('regions');
  state._libIdCurrent = null;
  renderLibrary();
});

refreshWorkflow();
refreshInpaintButton();
refreshRecBtn();

// ============================================================================
// Recorder — MP4 via WebCodecs VideoEncoder + mp4-muxer. Same encoder-picking
// strategy as v4: try HEVC then H.264, descending profile/level. Scales down
// if the chosen codec's max dim is exceeded.
// ============================================================================
// recState is declared earlier (near other state objects) so the Tweakpane
// setup can bind to it before this section executes.

// Record-button label / disabled state are owned by Tweakpane; helpers below.
function refreshRecBtn() {
  if (tp.recBtn) tp.recBtn.disabled = !state.img || recState.recording;
}
function setRecBtnLabel(text) { if (tp.recBtn) tp.recBtn.title = text; }

async function ensureMuxer() {
  if (recState.muxer) return recState.muxer;
  recState.muxer = await import('https://esm.sh/mp4-muxer@5.1.5');
  return recState.muxer;
}

async function pickEncoderConfig(width, height, framerate, bitrate) {
  if (typeof VideoEncoder === 'undefined') return null;
  const candidates = [
    { codec: 'hev1.1.6.L153.B0', muxer: 'hevc' }, // HEVC Main L5.1 — 8K
    { codec: 'hev1.1.6.L120.B0', muxer: 'hevc' }, // HEVC Main L4   — 4K
    { codec: 'avc1.640033',      muxer: 'avc'  }, // H.264 High L5.1
    { codec: 'avc1.640028',      muxer: 'avc'  }, // H.264 High L4
    { codec: 'avc1.42E01E',      muxer: 'avc'  }, // H.264 Baseline L3
  ];
  for (const c of candidates) {
    try {
      const cfg = { codec: c.codec, width, height, framerate, bitrate, hardwareAcceleration: 'prefer-hardware' };
      const r = await VideoEncoder.isConfigSupported(cfg);
      if (r && r.supported) return { config: cfg, muxerCodec: c.muxer };
    } catch {}
  }
  return null;
}

function codecMaxDim(codec) {
  if (codec.includes('L153'))   return 8192;
  if (codec.includes('L120'))   return 4096;
  if (codec.includes('640033')) return 8192;
  if (codec.includes('640028')) return 4096;
  return 3840;
}

async function startRecording() {
  if (recState.recording || !state.img) return;
  const { Muxer, ArrayBufferTarget } = await ensureMuxer();

  let recW = canvas.width, recH = canvas.height;
  if (recState.sizeMode !== 'src') {
    const w = parseInt(recState.sizeMode, 10);
    recH = Math.round(w * canvas.height / canvas.width);
    recW = w;
  }

  recState.recording = true;
  const origLabel = 'Record video';
  setRecBtnLabel('Preparing …');
  if (tp.recBtn) tp.recBtn.disabled = true;

  let scale = 1, pick = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    const tryW = recW + (recW % 2);
    const tryH = recH + (recH % 2);
    pick = await pickEncoderConfig(tryW, tryH, recState.fps, 12_000_000);
    if (pick) {
      const cap = codecMaxDim(pick.config.codec);
      if (Math.max(tryW, tryH) <= cap) break;
    }
    scale *= 0.75;
    recW = Math.round(recW * 0.75);
    recH = Math.round(recH * 0.75);
  }
  if (!pick) {
    setRecBtnLabel('FAILED — no usable encoder');
    setTimeout(() => { setRecBtnLabel(origLabel); recState.recording = false; refreshRecBtn(); }, 4000);
    return;
  }

  const offW   = recW + (recW % 2);
  const totalH = recH + (recH % 2);
  const off = document.createElement('canvas');
  off.width = offW; off.height = totalH;
  const offCtx = off.getContext('2d');
  offCtx.fillStyle = '#000';
  offCtx.fillRect(0, 0, offW, totalH);

  console.log(`[record] codec ${pick.config.codec} ${offW}×${totalH} ${recState.fps}fps${scale < 1 ? `  (×${scale.toFixed(2)} from canvas ${canvas.width}×${canvas.height})` : ''}`);

  const muxer = new Muxer({
    target: new ArrayBufferTarget(),
    video: { codec: pick.muxerCodec, width: offW, height: totalH, frameRate: recState.fps },
    fastStart: 'in-memory',
  });
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: (e) => console.error('[encoder]', e),
  });
  encoder.configure({ ...pick.config, width: offW, height: totalH });

  const totalFrames = Math.max(2, Math.round(state.duration * recState.fps));
  const frameDur = 1_000_000 / recState.fps;
  const wasPlaying = state.playing;
  const prevT = state.t;
  state.playing = false;

  setRecBtnLabel(scale < 1 ? `Recording (scaled to ${offW}×${totalH}) …` : 'Recording …');

  for (let i = 0; i < totalFrames; i++) {
    setT(i / (totalFrames - 1));
    render();
    // wait one rAF so WebGPU commits the just-submitted frame before we read it
    await new Promise(r => requestAnimationFrame(r));
    offCtx.drawImage(canvas, 0, 0, recW, recH);

    const vf = new VideoFrame(off, {
      timestamp: Math.round(i * frameDur),
      duration:  Math.round(frameDur),
    });
    encoder.encode(vf);
    vf.close();

    setRecBtnLabel(`frame ${i + 1} / ${totalFrames}`);
    setProgress('rec-progress', ((i + 1) / totalFrames) * 100);
    if (encoder.encodeQueueSize > 16) await new Promise(r => setTimeout(r, 0));
  }
  setProgress('rec-progress', null);
  await encoder.flush();
  muxer.finalize();

  recState.recording = false;
  setT(prevT);
  setPlaying(wasPlaying);

  const blob = new Blob([muxer.target.buffer], { type: 'video/mp4' });
  if (blob.size < 1024) {
    setRecBtnLabel('FAILED — empty output');
    setTimeout(() => { setRecBtnLabel(origLabel); refreshRecBtn(); }, 4000);
    return;
  }

  const stamp = new Date().toISOString().replace(/[T:]/g, '-').slice(0, 19);
  const tag   = `${Math.round(state.duration)}s_${recState.fps}fps_${offW}x${totalH}`;
  const filename = `transition-v5_${stamp}_${tag}.mp4`;
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);

  setRecBtnLabel(`saved (${(blob.size / 1024 / 1024).toFixed(1)} MB)`);
  setTimeout(() => { setRecBtnLabel(origLabel); refreshRecBtn(); }, 2500);
}

console.log('[transition-v5] WebGPU ready, format:', presentationFormat);
window.__v5 = { state, sam, regions };
