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

// 208-byte uniform layout (carefully aligned for std140-ish WGSL rules).
// Offsets are documented in JS-side writeUniforms() at the matching index.
struct Params {
  // -- 0..31 -- scalars & ints
  t: f32, spread: f32, organic: f32, edges: f32,
  maskScale: f32, seed: f32, validA: u32, validB: u32,
  // -- 32..63 -- image fit transforms (vec2 align 8)
  scaleA: vec2f, offsetA: vec2f, scaleB: vec2f, offsetB: vec2f,
  // -- 64..79 -- bg (vec3 align 16) + mode tightly packed
  bg: vec3f, mode: u32,
  // -- 80..95 -- enum-style u32s
  curve: u32, sedDirection: u32, sedSource: u32, saltSource: u32,
  // -- 96..127 -- rim, paper, blooms scalar params
  rimWidth: f32, rimDark: f32,
  paperAngle: f32, paperAniso: f32, paperGranulation: f32,
  bloomCount: u32, bloomRim: f32, bloomRate: f32,
  // -- 128..159 -- diffusion, sediment, salt scalar params
  diffStrength: f32, diffRadius: f32,
  sedBands: f32, sedSoftness: f32,
  saltDensity: f32, saltContrast: f32, saltBias: f32, saltImage: u32,
  // -- 160..175 -- iris (vec2 align 8) + jitter
  irisFocus: vec2f, irisJitter: f32, _p0: f32,
  // -- 176..191 -- bleed, run scalars
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  // -- 192..207 -- run drip + advection-family params (start)
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  // -- 208..223 -- gravity params
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  // -- 224..239 -- gravity lateral + curl + brush
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  // -- 240..255 -- seed
  advSeedCount: u32, advSeedRadius: f32, _p4: f32, _p5: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var advState: texture_2d<f32>;

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

// ---- mode-specific mask functions ------------------------------------------

fn paperMask(uv: vec2f) -> f32 {
  let ang = p.paperAngle * 3.14159265;
  let ca = cos(ang); let sa = sin(ang);
  let g  = vec2f(ca * (uv.x - 0.5) + sa * (uv.y - 0.5),
                -sa * (uv.x - 0.5) + ca * (uv.y - 0.5));
  let stretched = vec2f(g.x * p.maskScale, g.y * p.maskScale * p.paperAniso);
  let base = fbm(stretched + p.seed * 0.13);
  let tooth = fbm(uv * (p.maskScale * 14.0) + p.seed * 1.7) - 0.5;
  return clamp(base + tooth * p.paperGranulation * 0.35, 0.0, 1.0);
}

fn bloomsMask(uv: vec2f) -> f32 {
  var minReveal = 1.0;
  for (var i = 0u; i < 24u; i = i + 1u) {
    if (i >= p.bloomCount) { break; }
    let fi = f32(i) + p.seed * 0.07 + 1.0;
    let sp = vec2f(hash21(vec2f(fi * 1.3, 13.0)), hash21(vec2f(fi * 2.7, 47.0)));
    let startT = hash21(vec2f(fi, 91.0)) * 0.4;
    let jitter = 0.85 + 0.3 * hash21(vec2f(fi, 11.0));
    let d = distance(uv, sp);
    let wob = (fbm(uv * 4.0 + fi * 3.0) - 0.5) * 0.08;
    let reveal = startT + (d + wob) * (1.0 / max(p.bloomRate, 0.05)) * jitter;
    minReveal = min(minReveal, reveal);
  }
  return clamp(minReveal, 0.0, 1.0);
}

fn sedimentMask(uv: vec2f, cA: vec3f, cB: vec3f) -> f32 {
  let src = (cA + cB) * 0.5;
  var v: f32;
  if (p.sedSource == 0u) {                    // luminance
    v = luma(src);
  } else if (p.sedSource == 1u) {             // saturation
    let mx = max(max(src.r, src.g), src.b);
    let mn = min(min(src.r, src.g), src.b);
    v = select(0.0, (mx - mn) / mx, mx > 1e-4);
  } else if (p.sedSource == 2u) {             // hue
    let mx = max(max(src.r, src.g), src.b);
    let mn = min(min(src.r, src.g), src.b);
    let c  = mx - mn;
    var h = 0.0;
    if (c > 1e-4) {
      if (mx == src.r) {
        h = (src.g - src.b) / c;
        h = h - floor(h / 6.0) * 6.0;
      } else if (mx == src.g) {
        h = ((src.b - src.r) / c) + 2.0;
      } else {
        h = ((src.r - src.g) / c) + 4.0;
      }
      h = h / 6.0;
    }
    v = h;
  } else if (p.sedSource == 3u) {             // edge detail
    let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA);
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB);
    v = max(eA, eB);
  } else {                                     // temperature
    v = clamp(0.5 + (src.r - src.b) * 0.7, 0.0, 1.0);
  }
  if (p.sedDirection == 1u) { v = 1.0 - v; }
  let bands = max(1.0, p.sedBands);
  let quantized = floor(v * bands) / max(1.0, bands - 1.0);
  return clamp(mix(quantized, v, p.sedSoftness), 0.0, 1.0);
}

fn saltMask(uv: vec2f, cA: vec3f, cB: vec3f) -> f32 {
  let density = 6.0 + p.saltDensity * 90.0;
  let n1 = vnoise(uv * density + p.seed * 1.7);
  let n2 = vnoise(uv * density * 0.35 + p.seed * 0.3);
  let n = mix(n2, n1, 0.75);
  let k = 0.5 + p.saltContrast * 5.0;
  let m = clamp(0.5 + (n - 0.5) * k, 0.0, 1.0);
  var salt = 1.0 - m;

  if (p.saltSource != 0u && p.saltBias > 0.001) {
    var src: vec3f;
    if (p.saltImage == 0u)      { src = cA; }
    else if (p.saltImage == 1u) { src = cB; }
    else                         { src = (cA + cB) * 0.5; }
    var prop = 0.0;
    if (p.saltSource == 1u) { prop = luma(src); }
    else if (p.saltSource == 2u) { prop = 1.0 - luma(src); }
    else if (p.saltSource == 3u) {
      let mx = max(max(src.r, src.g), src.b);
      let mn = min(min(src.r, src.g), src.b);
      prop = select(0.0, (mx - mn) / mx, mx > 1e-4);
    } else if (p.saltSource == 4u) {
      let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA);
      let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB);
      if (p.saltImage == 0u) { prop = eA; }
      else if (p.saltImage == 1u) { prop = eB; }
      else { prop = max(eA, eB); }
    }
    salt = clamp(salt - prop * p.saltBias * 0.75, 0.0, 1.0);
  }
  return salt;
}

fn irisMask(uv: vec2f) -> f32 {
  let d = (uv - p.irisFocus) * 1.4142;
  let r = length(d);
  let jit = (fbm(uv * 3.5 + p.seed * 0.21) - 0.5) * p.irisJitter * 0.3;
  return clamp(r + jit, 0.0, 1.0);
}

fn wetBleedMask(uv: vec2f, lA: f32, lB: f32) -> f32 {
  let base = mix(0.5, 0.5 + 0.5 * (lB - lA), 0.55);
  let aniso = mix(8.0, 28.0, p.bleedFinger);
  let fingUV = uv * vec2f(aniso, aniso * 0.35);
  let n1 = fbm(fingUV + p.seed * 0.3);
  let n2 = fbm(uv * 3.0 + p.seed * 0.7);
  let fingers = (n1 - 0.5) * p.bleedAmount * 0.8 + (n2 - 0.5) * 0.18;
  return clamp(base + fingers, 0.0, 1.0);
}

fn pigmentRunMask(uv: vec2f, lA: f32) -> f32 {
  let m = mix(lA, uv.y, p.runGravity);
  let n = (fbm(uv * 2.5 + p.seed * 0.11) - 0.5) * 0.06;
  return clamp(m + n, 0.0, 1.0);
}

fn organicMask(uv: vec2f, lA: f32, lB: f32, edge: f32) -> f32 {
  let n1 = fbm(uv * p.maskScale + p.seed * 0.13);
  let n2 = fbm(uv * p.maskScale * 2.3 + 17.0 + p.seed * 0.09);
  let noiseMask = mix(n1, n2, 0.35);
  let lumMask = 0.5 + 0.5 * (lB - lA);
  var m = mix(noiseMask, lumMask, p.organic);
  m = m - p.edges * edge * 0.45;
  return clamp(m, 0.0, 1.0);
}

// ---- main shader ------------------------------------------------------------

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;

  // Advection family (modes 10..14): the compute pipeline writes a state
  // texture each frame; here we just sample and present it.
  if (p.mode >= 10u && p.mode <= 14u) {
    return vec4f(textureSampleLevel(advState, samp, uv, 0.0).rgb, 1.0);
  }

  let t = applyCurve(p.t, p.curve);
  let env = pow(sin(3.14159265 * clamp(p.t, 0.0, 1.0)), 0.85);

  let cA = sampleFit(texA, uv, p.scaleA, p.offsetA, p.validA);
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB);
  let lA = luma(cA.rgb);
  let lB = luma(cB.rgb);

  // ---- pick a mask based on mode ----
  var mask = 0.0;
  if (p.mode == 2u) {
    mask = paperMask(uv);
    let lumMask = 0.5 + 0.5 * (lB - lA);
    mask = mix(mask, lumMask, p.organic * 0.35);
  } else if (p.mode == 3u) {
    mask = bloomsMask(uv);
  } else if (p.mode == 5u) {
    mask = sedimentMask(uv, cA.rgb, cB.rgb);
  } else if (p.mode == 6u) {
    mask = saltMask(uv, cA.rgb, cB.rgb);
  } else if (p.mode == 7u) {
    mask = irisMask(uv);
  } else if (p.mode == 8u) {
    mask = wetBleedMask(uv, lA, lB);
  } else if (p.mode == 9u) {
    mask = pigmentRunMask(uv, lA);
  } else {
    let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA);
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB);
    mask = organicMask(uv, lA, lB, max(eA, eB));
  }

  let sp = mix(0.05, 0.7, p.spread);
  var mixT = clamp(smoothstep(mask - sp, mask + sp, t), 0.0, 1.0);

  // ---- wet diffusion (mode 4): anticipatory tint of B into A ----
  var colA_eff = cA.rgb;
  if (p.mode == 4u && p.diffStrength > 0.001) {
    let anticipate = smoothstep(mask - 0.45, mask + 0.05, t);
    let bR = 0.025 + p.diffRadius * 0.08;
    // simple 12-tap soft blur of B
    var acc = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB).rgb * 0.35;
    var wsum = 0.35;
    for (var i = 0u; i < 12u; i = i + 1u) {
      let a = f32(i) * (6.2831853 / 12.0) + p.seed * 0.013;
      let d = vec2f(cos(a), sin(a));
      let rr = select(0.55, 1.0, (i % 2u) == 0u);
      let w = 1.0 - rr * 0.45;
      acc = acc + sampleFit(texB, uv + d * rr * bR, p.scaleB, p.offsetB, p.validB).rgb * w;
      wsum = wsum + w;
    }
    let bleedB = acc / wsum;
    let dry = 1.0 - mixT;
    colA_eff = mix(cA.rgb, bleedB, anticipate * dry * p.diffStrength * 0.55);
  }

  var outc = mix(colA_eff, cB.rgb, mixT);

  // ---- rim post-process (modes 1 and 3) ----
  if ((p.mode == 1u || p.mode == 3u) && env > 0.02) {
    let rimW = select(p.rimWidth * 0.4, p.bloomRim * 0.5, p.mode == 3u);
    if (rimW > 0.001) {
      let band = 1.0 - smoothstep(0.0, rimW, abs(t - mask));
      let base = mix(cA.rgb, cB.rgb, 0.65);
      let lm = luma(base);
      let chromaBoost = clamp(mix(vec3f(lm), base, 1.35), vec3f(0.0), vec3f(1.0));
      let rim = chromaBoost * 0.78;
      let darkness = select(p.rimDark, p.bloomRim, p.mode == 3u);
      let fade = env * env;
      outc = mix(outc, rim, band * darkness * fade * 0.85);
    }
  }

  // ---- wet bleed halo (mode 8) ----
  if (p.mode == 8u && p.bleedHalo > 0.001 && env > 0.02 && t > mask) {
    let haloW = 0.005 + p.bleedHalo * 0.06;
    let band = exp(-pow((t - mask) / haloW, 2.0));
    let base = mix(cA.rgb, cB.rgb, 0.75);
    let lm = luma(base);
    let saturated = clamp(mix(vec3f(lm), base, 1.5), vec3f(0.0), vec3f(1.0));
    outc = mix(outc, saturated, band * p.bleedHalo * env * 0.4);
  }

  // ---- pigment run drip (mode 9) ----
  if (p.mode == 9u && p.runDrip > 0.001 && env > 0.02 && t > mask) {
    let dripBand = exp(-pow((t - mask) / 0.08, 2.0));
    let dripB = sampleFit(texB, uv + vec2f(0.0, p.runDrip * 0.05), p.scaleB, p.offsetB, p.validB).rgb;
    outc = mix(outc, dripB, dripBand * p.runDrip * env * 0.35);
  }

  return vec4f(clamp(outc, vec3f(0.0), vec3f(1.0)), 1.0);
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
    { binding: 4, visibility: GPUShaderStage.FRAGMENT, texture: { sampleType: 'float' } },
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

// ============================================================================
// Advection sim — render-to-texture ping-pong (mode 10..14)
// ============================================================================
const STATE_FORMAT = 'rgba16float';
const SIM_SHADER = /* wgsl */`
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };

struct Params {
  t: f32, spread: f32, organic: f32, edges: f32,
  maskScale: f32, seed: f32, validA: u32, validB: u32,
  scaleA: vec2f, offsetA: vec2f, scaleB: vec2f, offsetB: vec2f,
  bg: vec3f, mode: u32,
  curve: u32, sedDirection: u32, sedSource: u32, saltSource: u32,
  rimWidth: f32, rimDark: f32,
  paperAngle: f32, paperAniso: f32, paperGranulation: f32,
  bloomCount: u32, bloomRim: f32, bloomRate: f32,
  diffStrength: f32, diffRadius: f32,
  sedBands: f32, sedSoftness: f32,
  saltDensity: f32, saltContrast: f32, saltBias: f32, saltImage: u32,
  irisFocus: vec2f, irisJitter: f32, _p0: f32,
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  advSeedCount: u32, advSeedRadius: f32, _p4: f32, _p5: f32,
};

@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var stateIn: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
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

fn hash21(q: vec2f) -> f32 {
  var x = fract(q * vec2f(123.34, 456.21));
  x += dot(x, x + 45.32);
  return fract(x.x * x.y);
}
fn vnoise(q: vec2f) -> f32 {
  let i = floor(q); let f = fract(q);
  let u = f * f * (3.0 - 2.0 * f);
  let a = hash21(i);
  let b = hash21(i + vec2f(1.0, 0.0));
  let c = hash21(i + vec2f(0.0, 1.0));
  let d = hash21(i + vec2f(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
fn fbm(q: vec2f) -> f32 {
  var v = 0.0; var amp = 0.5; var pp = q;
  for (var i = 0; i < 4; i = i + 1) { v += amp * vnoise(pp); pp *= 2.03; amp *= 0.5; }
  return v;
}
fn luma(c: vec3f) -> f32 { return dot(c, vec3f(0.299, 0.587, 0.114)); }
fn sampleFit(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32) -> vec4f {
  if (valid == 0u) { return vec4f(p.bg, 1.0); }
  let q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec4f(p.bg, 1.0); }
  return textureSampleLevel(tex, samp, q, 0.0);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  // 1 / dims (we sample state via sampler so we need pixel size in UV space)
  let dims = vec2f(textureDimensions(stateIn));
  let px = 1.0 / dims;

  // 8-tap neighbour average for soft isotropic diffusion (variant 0, baseline).
  let cur = textureSampleLevel(stateIn, samp, uv, 0.0).rgb;
  let nb = (
    textureSampleLevel(stateIn, samp, uv + vec2f(px.x, 0.0), 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv - vec2f(px.x, 0.0), 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv + vec2f(0.0, px.y), 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv - vec2f(0.0, px.y), 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv + px * 0.7071, 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv - px * 0.7071, 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv + vec2f(px.x, -px.y) * 0.7071, 0.0).rgb +
    textureSampleLevel(stateIn, samp, uv + vec2f(-px.x, px.y) * 0.7071, 0.0).rgb
  ) * 0.125;
  let diffused = mix(cur, nb, p.advVisc);

  // image-aware mask: noise + luma delta
  let cA = sampleFit(texA, uv, p.scaleA, p.offsetA, p.validA);
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB);
  let n1 = fbm(uv * p.maskScale + p.seed * 0.13);
  let n2 = fbm(uv * p.maskScale * 2.3 + 17.0 + p.seed * 0.09);
  let noiseMask = mix(n1, n2, 0.35);
  let lA = luma(cA.rgb); let lB = luma(cB.rgb);
  let lumMask = 0.5 + 0.5 * (lB - lA);
  let mask = mix(noiseMask, lumMask, p.organic);

  let sp = mix(0.1, 0.5, p.spread);
  let reveal = smoothstep(mask - sp, mask + sp * 0.3, p.t);
  let mixed = mix(diffused, cB.rgb, reveal * p.advRate);
  return vec4f(mixed, 1.0);
}
`;

const INIT_SHADER = /* wgsl */`
struct VSOut { @builtin(position) pos: vec4f, @location(0) uv: vec2f };
struct Params {
  // minimal — we only read fit + bg + validA here. Reuse full struct for layout.
  t: f32, spread: f32, organic: f32, edges: f32,
  maskScale: f32, seed: f32, validA: u32, validB: u32,
  scaleA: vec2f, offsetA: vec2f, scaleB: vec2f, offsetB: vec2f,
  bg: vec3f, mode: u32,
  curve: u32, sedDirection: u32, sedSource: u32, saltSource: u32,
  rimWidth: f32, rimDark: f32,
  paperAngle: f32, paperAniso: f32, paperGranulation: f32,
  bloomCount: u32, bloomRim: f32, bloomRate: f32,
  diffStrength: f32, diffRadius: f32,
  sedBands: f32, sedSoftness: f32,
  saltDensity: f32, saltContrast: f32, saltBias: f32, saltImage: u32,
  irisFocus: vec2f, irisJitter: f32, _p0: f32,
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  advSeedCount: u32, advSeedRadius: f32, _p4: f32, _p5: f32,
};
@group(0) @binding(0) var<uniform> p: Params;
@group(0) @binding(1) var texA: texture_2d<f32>;
@group(0) @binding(2) var texB: texture_2d<f32>;
@group(0) @binding(3) var samp: sampler;
@group(0) @binding(4) var stateIn: texture_2d<f32>;
@vertex fn vs(@builtin(vertex_index) idx: u32) -> VSOut {
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
@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  if (p.validA == 0u) { return vec4f(p.bg, 1.0); }
  let q = (uv - p.offsetA) / p.scaleA;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec4f(p.bg, 1.0); }
  return vec4f(textureSampleLevel(texA, samp, q, 0.0).rgb, 1.0);
}
`;

const simModule  = device.createShaderModule({ code: SIM_SHADER });
const initModule = device.createShaderModule({ code: INIT_SHADER });
const simPipeline = device.createRenderPipeline({
  label: 'sim-pipeline',
  layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
  vertex:   { module: simModule, entryPoint: 'vs' },
  fragment: { module: simModule, entryPoint: 'fs', targets: [{ format: STATE_FORMAT }] },
  primitive: { topology: 'triangle-list' },
});
const initPipeline = device.createRenderPipeline({
  label: 'init-pipeline',
  layout: device.createPipelineLayout({ bindGroupLayouts: [bindGroupLayout] }),
  vertex:   { module: initModule, entryPoint: 'vs' },
  fragment: { module: initModule, entryPoint: 'fs', targets: [{ format: STATE_FORMAT }] },
  primitive: { topology: 'triangle-list' },
});

let stateTexA = null, stateTexB = null;
let stateW = 0, stateH = 0;
const advec = { src: 'A', lastT: 0, needsReset: true };

function ensureStateTextures() {
  const w = canvas.width, h = canvas.height;
  if (!w || !h) return;
  if (w === stateW && h === stateH && stateTexA) return;
  if (stateTexA) stateTexA.destroy();
  if (stateTexB) stateTexB.destroy();
  stateTexA = device.createTexture({
    label: 'state-A',
    size: [w, h, 1], format: STATE_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  stateTexB = device.createTexture({
    label: 'state-B',
    size: [w, h, 1], format: STATE_FORMAT,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
  });
  stateW = w; stateH = h;
  advec.needsReset = true;
}

// Build bind groups dynamically — they must reference texA/texB/state textures
// that all change at runtime. Cheap to recreate, so do it each step.
function makeSimBindGroup(stateIn) {
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texA.createView() },
      { binding: 2, resource: texB.createView() },
      { binding: 3, resource: sampler },
      { binding: 4, resource: stateIn.createView() },
    ],
  });
}
function makeDisplayBindGroup(finalState) {
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texA.createView() },
      { binding: 2, resource: texB.createView() },
      { binding: 3, resource: sampler },
      { binding: 4, resource: finalState.createView() },
    ],
  });
}

// Uniform buffer — 208 bytes total, matches the Params struct in WGSL.
// Offsets (in 4-byte units, which is the JS Float32Array / Uint32Array index):
//   0  t            8   scaleA.x      16 bg.r          20 curve          24 rimWidth       32 diffStrength    40 irisFocus.x   44 bleedFinger    48 runDrip
//   1  spread       9   scaleA.y      17 bg.g          21 sedDirection   25 rimDark        33 diffRadius      41 irisFocus.y   45 bleedAmount    49 _p1
//   2  organic      10  offsetA.x     18 bg.b          22 sedSource      26 paperAngle     34 sedBands        42 irisJitter    46 bleedHalo      50 _p2
//   3  edges        11  offsetA.y     19 mode          23 saltSource     27 paperAniso     35 sedSoftness     43 _p0           47 runGravity     51 _p3
//   4  maskScale    12  scaleB.x                                          28 paperGran      36 saltDensity
//   5  seed         13  scaleB.y                                          29 bloomCount     37 saltContrast
//   6  validA       14  offsetB.x                                         30 bloomRim       38 saltBias
//   7  validB       15  offsetB.y                                         31 bloomRate      39 saltImage
const UBO_SIZE = 256;
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

// Placeholder state texture so the bind group is valid before sim runs.
// Replaced with real ping-pong textures on first advection frame.
let placeholderState = device.createTexture({
  label: 'state-placeholder',
  size: [1, 1, 1], format: STATE_FORMAT,
  usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.RENDER_ATTACHMENT,
});

function makeBindGroup(stateView) {
  return device.createBindGroup({
    layout: bindGroupLayout,
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: texA.createView() },
      { binding: 2, resource: texB.createView() },
      { binding: 3, resource: sampler },
      { binding: 4, resource: stateView || placeholderState.createView() },
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
  advec.needsReset = true;
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
  mode: 0,
  // mode-specific defaults (mirrors v1)
  rimWidth: 0.12, rimDark: 0.6,
  paperAngle: 0, paperAniso: 4, paperGranulation: 0.5,
  bloomCount: 8, bloomRim: 0.6, bloomRate: 0.55,
  diffStrength: 0.55, diffRadius: 0.45,
  sedBands: 6, sedSoftness: 0.35, sedDirection: 0, sedSource: 0,
  saltDensity: 0.55, saltContrast: 0.55,
  saltSource: 1, saltBias: 0.6, saltImage: 2,
  irisFocusX: 0.5, irisFocusY: 0.5, irisJitter: 0.35,
  bleedFinger: 0.5, bleedAmount: 0.45, bleedHalo: 0.5,
  runGravity: 0.5, runDrip: 0.35,
  // advection family
  advecVisc: 0.55, advecRate: 0.18, advecSteps: 3,
  advecGravity: 0.6, advecGravBias: 0.5,
  advecGravAngle: 0, advecGravStreak: 0.4, advecGravLateral: 0.3,
  advecCurlStr: 0.5, advecCurlScale: 2.5,
  advecBrushFollow: 0.7,
  advecSeedCount: 5, advecSeedRadius: 0.45,
  // style / framing
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

  // -- 0..7 --
  uboF32[0]  = state.t;
  uboF32[1]  = state.spread;
  uboF32[2]  = state.organic;
  uboF32[3]  = state.edges;
  uboF32[4]  = state.maskScale;
  uboF32[5]  = state.seed;
  uboU32[6]  = state.imgA ? 1 : 0;
  uboU32[7]  = state.imgB ? 1 : 0;
  // -- 8..15 --
  uboF32[8]  = fA.sx; uboF32[9]  = fA.sy;
  uboF32[10] = fA.ox; uboF32[11] = fA.oy;
  uboF32[12] = fB.sx; uboF32[13] = fB.sy;
  uboF32[14] = fB.ox; uboF32[15] = fB.oy;
  // -- 16..19 --
  uboF32[16] = bg[0]; uboF32[17] = bg[1]; uboF32[18] = bg[2];
  uboU32[19] = state.mode;
  // -- 20..23 -- enum-style u32s
  uboU32[20] = state.curve;
  uboU32[21] = state.sedDirection;
  uboU32[22] = state.sedSource;
  uboU32[23] = state.saltSource;
  // -- 24..31 -- rim, paper, blooms scalars
  uboF32[24] = state.rimWidth;
  uboF32[25] = state.rimDark;
  uboF32[26] = state.paperAngle;
  uboF32[27] = state.paperAniso;
  uboF32[28] = state.paperGranulation;
  uboU32[29] = state.bloomCount;
  uboF32[30] = state.bloomRim;
  uboF32[31] = state.bloomRate;
  // -- 32..39 -- diffusion, sediment, salt scalars
  uboF32[32] = state.diffStrength;
  uboF32[33] = state.diffRadius;
  uboF32[34] = state.sedBands;
  uboF32[35] = state.sedSoftness;
  uboF32[36] = state.saltDensity;
  uboF32[37] = state.saltContrast;
  uboF32[38] = state.saltBias;
  uboU32[39] = state.saltImage;
  // -- 40..43 -- iris
  uboF32[40] = state.irisFocusX;
  uboF32[41] = state.irisFocusY;
  uboF32[42] = state.irisJitter;
  uboF32[43] = 0;  // _p0
  // -- 44..47 -- bleed + run
  uboF32[44] = state.bleedFinger;
  uboF32[45] = state.bleedAmount;
  uboF32[46] = state.bleedHalo;
  uboF32[47] = state.runGravity;
  // -- 48..51 --
  uboF32[48] = state.runDrip;
  uboU32[49] = (state.mode >= 11 && state.mode <= 14) ? (state.mode - 10) : 0; // advVariant
  uboF32[50] = state.advecVisc;
  uboF32[51] = state.advecRate;
  // -- 52..55 -- gravity
  uboF32[52] = state.advecGravity;
  uboF32[53] = state.advecGravBias;
  uboF32[54] = state.advecGravAngle;
  uboF32[55] = state.advecGravStreak;
  // -- 56..59 -- lateral + curl + brush
  uboF32[56] = state.advecGravLateral;
  uboF32[57] = state.advecCurlStr;
  uboF32[58] = state.advecCurlScale;
  uboF32[59] = state.advecBrushFollow;
  // -- 60..63 -- seed
  uboU32[60] = state.advecSeedCount;
  uboF32[61] = state.advecSeedRadius;
  uboF32[62] = 0; uboF32[63] = 0;

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

    const isAdvec = state.mode >= 10 && state.mode <= 14;
    let finalState = null;
    const enc = device.createCommandEncoder();

    if (isAdvec) {
      ensureStateTextures();
      // Detect reset condition: first run, scrubbing backward, or t == 0.
      if (advec.needsReset || state.t < advec.lastT - 0.03 || state.t === 0) {
        // Init pass: render texA into stateTexA via initPipeline.
        const initPass = enc.beginRenderPass({
          colorAttachments: [{
            view: stateTexA.createView(),
            clearValue: { r: 0, g: 0, b: 0, a: 1 },
            loadOp: 'clear', storeOp: 'store',
          }],
        });
        initPass.setPipeline(initPipeline);
        initPass.setBindGroup(0, makeSimBindGroup(stateTexB));  // stateIn unused by init
        initPass.draw(6);
        initPass.end();
        advec.src = 'A';
        advec.lastT = 0;
        advec.needsReset = false;
        // Warm-up: run several sim steps so the state matches current t.
        const warm = Math.max(8, Math.round(state.advecSteps * 8));
        for (let i = 0; i < warm; i++) {
          const tAt = state.t * ((i + 1) / warm);
          runSimStepInto(enc, tAt);
        }
      } else {
        const N = Math.max(1, Math.round(state.advecSteps));
        const startT = advec.lastT, endT = state.t;
        for (let i = 0; i < N; i++) {
          runSimStepInto(enc, startT + (endT - startT) * ((i + 1) / N));
        }
      }
      advec.lastT = state.t;
      finalState = (advec.src === 'A') ? stateTexA : stateTexB;
    }

    const displayBG = isAdvec ? makeDisplayBindGroup(finalState) : bindGroup;
    const pass = enc.beginRenderPass({
      colorAttachments: [{
        view: ctx.getCurrentTexture().createView(),
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear', storeOp: 'store',
      }],
    });
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, displayBG);
    pass.draw(6);
    pass.end();
    device.queue.submit([enc.finish()]);
  }
  requestAnimationFrame(render);
}

// Run one sim step into the off-source ping-pong texture, then swap which is
// "current". Writes the current t into the uniform first (so successive steps
// can interpolate from lastT to current).
function runSimStepInto(encoder, tAt) {
  // Patch uniform t for this step (rest of params unchanged from writeUniforms).
  uboF32[0] = tAt;
  device.queue.writeBuffer(uniformBuffer, 0, uboHost, 0, 4);

  const srcTex = (advec.src === 'A') ? stateTexA : stateTexB;
  const dstTex = (advec.src === 'A') ? stateTexB : stateTexA;
  const pass = encoder.beginRenderPass({
    colorAttachments: [{
      view: dstTex.createView(),
      clearValue: { r: 0, g: 0, b: 0, a: 1 },
      loadOp: 'load', storeOp: 'store',
    }],
  });
  pass.setPipeline(simPipeline);
  pass.setBindGroup(0, makeSimBindGroup(srcTex));
  pass.draw(6);
  pass.end();
  advec.src = (advec.src === 'A') ? 'B' : 'A';
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

// ----- Watercolor mode + per-mode controls -----
const fWater = pane.addFolder({ title: 'Watercolor', expanded: true });
fWater.addBinding(state, 'mode', {
  label: 'mode',
  options: {
    'off (smooth)':    0,
    'pigment rim':     1,
    'paper grain':     2,
    'backrun blooms':  3,
    'wet diffusion':   4,
    'tonal sediment':  5,
    'salt':            6,
    'iris':            7,
    'wet bleed':       8,
    'pigment run':     9,
    'wet advection':   10,
  },
}).on('change', () => { updateModeFolders(); advec.needsReset = true; });

const fRim    = fWater.addFolder({ title: 'Pigment rim',    expanded: true });
fRim.addBinding(state, 'rimWidth', { min: 0, max: 0.4, step: 0.005, label: 'rim width' });
fRim.addBinding(state, 'rimDark',  { min: 0, max: 1, step: 0.01, label: 'rim dark' });

const fPaper  = fWater.addFolder({ title: 'Paper grain',    expanded: true });
fPaper.addBinding(state, 'paperAngle',       { min: 0, max: 1, step: 0.005, label: 'fiber angle' });
fPaper.addBinding(state, 'paperAniso',       { min: 1, max: 10, step: 0.1, label: 'anisotropy' });
fPaper.addBinding(state, 'paperGranulation', { min: 0, max: 1, step: 0.01, label: 'granulation' });

const fBlooms = fWater.addFolder({ title: 'Backrun blooms', expanded: true });
fBlooms.addBinding(state, 'bloomCount', { min: 1, max: 24, step: 1, label: 'count' });
fBlooms.addBinding(state, 'bloomRate',  { min: 0.1, max: 2, step: 0.01, label: 'growth rate' });
fBlooms.addBinding(state, 'bloomRim',   { min: 0, max: 1, step: 0.01, label: 'rim dark' });

const fDiff   = fWater.addFolder({ title: 'Wet diffusion',  expanded: true });
fDiff.addBinding(state, 'diffStrength', { min: 0, max: 1, step: 0.01, label: 'strength' });
fDiff.addBinding(state, 'diffRadius',   { min: 0, max: 1, step: 0.01, label: 'radius' });

const fSed    = fWater.addFolder({ title: 'Tonal sediment', expanded: true });
fSed.addBinding(state, 'sedSource', {
  label: 'decompose by',
  options: { 'luminance': 0, 'saturation': 1, 'hue': 2, 'detail': 3, 'temperature': 4 },
});
fSed.addBinding(state, 'sedBands',    { min: 1, max: 16, step: 1, label: 'bands' });
fSed.addBinding(state, 'sedSoftness', { min: 0, max: 1, step: 0.01, label: 'softness' });
fSed.addBinding(state, 'sedDirection', {
  label: 'order',
  options: { 'low → high': 0, 'high → low': 1 },
});

const fSalt   = fWater.addFolder({ title: 'Salt',           expanded: true });
fSalt.addBinding(state, 'saltDensity',  { min: 0, max: 1, step: 0.01, label: 'grain' });
fSalt.addBinding(state, 'saltContrast', { min: 0, max: 1, step: 0.01, label: 'contrast' });
fSalt.addBinding(state, 'saltSource', {
  label: 'reveal from',
  options: { 'random (none)': 0, 'light areas': 1, 'dark areas': 2, 'coloured areas': 3, 'edge detail': 4 },
});
fSalt.addBinding(state, 'saltImage', {
  label: 'sample',
  options: { 'A': 0, 'B': 1, 'both': 2 },
});
fSalt.addBinding(state, 'saltBias',     { min: 0, max: 1, step: 0.01, label: 'bias amount' });

const fIris   = fWater.addFolder({ title: 'Iris',           expanded: true });
fIris.addBinding(state, 'irisFocusX', { min: 0, max: 1, step: 0.005, label: 'focus x' });
fIris.addBinding(state, 'irisFocusY', { min: 0, max: 1, step: 0.005, label: 'focus y' });
fIris.addBinding(state, 'irisJitter', { min: 0, max: 1, step: 0.01, label: 'jitter' });

const fBleed  = fWater.addFolder({ title: 'Wet bleed',      expanded: true });
fBleed.addBinding(state, 'bleedFinger', { min: 0, max: 1, step: 0.01, label: 'finger' });
fBleed.addBinding(state, 'bleedAmount', { min: 0, max: 1, step: 0.01, label: 'amount' });
fBleed.addBinding(state, 'bleedHalo',   { min: 0, max: 1, step: 0.01, label: 'wet halo' });

const fRun    = fWater.addFolder({ title: 'Pigment run',    expanded: true });
fRun.addBinding(state, 'runGravity', { min: 0, max: 1, step: 0.01, label: 'gravity' });
fRun.addBinding(state, 'runDrip',    { min: 0, max: 1, step: 0.01, label: 'drip' });

const fAdvec  = fWater.addFolder({ title: 'Wet advection',  expanded: true });
fAdvec.addBinding(state, 'advecVisc',  { min: 0, max: 1, step: 0.01, label: 'viscosity' });
fAdvec.addBinding(state, 'advecRate',  { min: 0, max: 1, step: 0.01, label: 'mixing rate' });
fAdvec.addBinding(state, 'advecSteps', { min: 1, max: 8, step: 1, label: 'steps / frame' });
fAdvec.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

function updateModeFolders() {
  fRim.hidden    = state.mode !== 1;
  fPaper.hidden  = state.mode !== 2;
  fBlooms.hidden = state.mode !== 3;
  fDiff.hidden   = state.mode !== 4;
  fSed.hidden    = state.mode !== 5;
  fSalt.hidden   = state.mode !== 6;
  fIris.hidden   = state.mode !== 7;
  fBleed.hidden  = state.mode !== 8;
  fRun.hidden    = state.mode !== 9;
  fAdvec.hidden  = state.mode !== 10;
}
updateModeFolders();

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
