// WebGPU port of the transition tool.
//
// Milestone 1: scaffold + smooth organic crossfade.
// The JS shell (state, Tweakpane, slots, recorder, presets) mirrors the WebGL2
// version but the rendering path is rebuilt on WebGPU. Modes will be ported in
// subsequent milestones; right now only the default smooth dissolve runs.

import { Pane } from 'tweakpane';
import * as EssentialsPlugin from '@tweakpane/plugin-essentials';
import { Muxer, ArrayBufferTarget } from 'mp4-muxer';

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
ctx.configure({ device, format: presentationFormat, alphaMode: 'premultiplied' });

const GPU_MAX_TEX = device.limits.maxTextureDimension2D;
console.log('[transition-tool-v3] device limits.maxTextureDimension2D =', GPU_MAX_TEX);

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
  // -- 160..175 -- iris (vec2 align 8) + jitter + uniform-circle toggle
  irisFocus: vec2f, irisJitter: f32, irisUniform: u32,
  // -- 176..191 -- bleed, run scalars
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  // -- 192..207 -- run drip + advection-family params (start)
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  // -- 208..223 -- gravity params
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  // -- 224..239 -- gravity lateral + curl + brush
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  // -- 240..255 -- seed + canvas aspect (w/h) used by uniform-circle iris
  advSeedCount: u32, advSeedRadius: f32, canvasAspect: f32, _p5: f32,
  // -- 256..271 -- wet edge (mode 15): rect ingress
  weEdgeScale: f32, weEdgeWobble: f32, weDryRing: f32, weBleed: f32,
  // -- 272..287 -- wet edge: tendrils
  weTendrilCount: u32, weTendrilReach: f32, weTendrilWidth: f32, weTendrilStrength: f32,
  // -- 288..303 -- wet edge: detail bias + future padding
  weDetailBias: f32, moldTendrilsPerSeed: u32, weReverse: u32, weBDetailBias: f32,
  // -- 304..319 -- mold tendrils (mode 22): direct fbm-warped tendril paths
  moldWidth: f32, moldWobble: f32, moldSeedCount: u32, moldReach: f32,
  // -- 320..335 -- new painterly modes 16..21: stroke / glaze
  strokeScale: f32, strokeAniso: f32, glazeBands: f32, glazeSoftness: f32,
  // -- 336..351 -- glaze direction + warm tint / edge-first + dabs
  glazeDirection: u32, glazeWarm: f32, edgeFirstInk: f32, edgeFirstFade: f32,
  // -- 352..367 -- edge-first scale / flow / dabs count + reach
  edgeFirstScale: f32, flowAmount: f32, dabsCount: u32, dabsReach: f32,
  // -- 368..383 -- dabs wobble / density / global paper grain
  dabsWobble: f32, densityGravity: f32, densitySmear: f32, paperGrain: f32,
  formStrokeCount: u32, formStrokeSize: f32, formStrokeWobble: f32, _f1: f32,
  bloomLightBias: f32, bloomWobble: f32, bloomPaperShow: f32, _b1: f32,
  stageBands: f32, stageOverlap: f32, _s1: f32, _s2: f32,
  migrationStrength: f32, migrationDir: u32, migrationTurb: f32, _m1: f32,
  boundsEnable: u32, boundsCx: f32, boundsCy: f32, boundsW: f32,
  boundsH: f32, boundsSoftness: f32, weBLumaBias: f32, maskShift: f32,
  slotAColor: vec3f, keepAOutsideB: u32,
  slotBColor: vec3f, _slotBPad: f32,
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

fn sampleFit(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32, color: vec3f) -> vec4f {
  // valid encoding: 0 = no image (bg fallback), 1 = image, 2 = solid color, 3 = transparent.
  if (valid == 0u) { return vec4f(p.bg, 1.0); }
  if (valid == 2u) { return vec4f(color, 1.0); }
  if (valid == 3u) { return vec4f(0.0, 0.0, 0.0, 0.0); }
  let q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec4f(p.bg, 1.0); }
  // textureSampleLevel avoids the uniform-control-flow requirement of
  // textureSample (no implicit derivatives → safe inside conditionals).
  return textureSampleLevel(tex, samp, q, 0.0);
}

fn edgeMag(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32, color: vec3f) -> f32 {
  if (valid == 0u) { return 0.0; }
  let e = 0.0025;
  let cx1 = luma(sampleFit(tex, uv + vec2f( e, 0.0), scale, offset, valid, color).rgb);
  let cx2 = luma(sampleFit(tex, uv - vec2f( e, 0.0), scale, offset, valid, color).rgb);
  let cy1 = luma(sampleFit(tex, uv + vec2f(0.0, e), scale, offset, valid, color).rgb);
  let cy2 = luma(sampleFit(tex, uv - vec2f(0.0, e), scale, offset, valid, color).rgb);
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
    let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
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
      let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
      let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
      if (p.saltImage == 0u) { prop = eA; }
      else if (p.saltImage == 1u) { prop = eB; }
      else { prop = max(eA, eB); }
    }
    salt = clamp(salt - prop * p.saltBias * 0.75, 0.0, 1.0);
  }
  return salt;
}

fn irisMask(uv: vec2f) -> f32 {
  var d = uv - p.irisFocus;
  var norm = 1.4142;
  if (p.irisUniform == 1u) {
    // Aspect-correct so the iris is a circle in pixel space (not stretched
    // along the wider canvas axis). Renormalize so corners still reach r=1.
    d.x = d.x * p.canvasAspect;
    norm = 2.0 / sqrt(p.canvasAspect * p.canvasAspect + 1.0);
  }
  let r = length(d) * norm;
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

fn paperFiber(uv: vec2f) -> f32 {
  // Multi-octave fine fibers. Returns signed [-0.5..0.5]-ish modulation.
  let f1 = vnoise(uv * 300.0 + p.seed * 1.7) - 0.5;
  let f2 = vnoise(uv *  80.0 + p.seed * 0.4) - 0.5;
  let f3 = vnoise(uv * 700.0 + p.seed * 0.9) - 0.5;
  return f1 * 0.4 + f2 * 0.3 + f3 * 0.3;
}

fn strokeFollowMask(uv: vec2f) -> f32 {
  // Local gradient of B's luma → perpendicular is the local stroke direction.
  let e = 0.003;
  let gx = luma(sampleFit(texB, uv + vec2f( e, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
         - luma(sampleFit(texB, uv - vec2f( e, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
  let gy = luma(sampleFit(texB, uv + vec2f(0.0,  e), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
         - luma(sampleFit(texB, uv - vec2f(0.0,  e), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
  let grad = vec2f(gx, gy);
  let glen = length(grad);
  let strokeDir = select(vec2f(1.0, 0.0), vec2f(-grad.y, grad.x) / glen, glen > 1e-4);
  let perpDir = vec2f(-strokeDir.y, strokeDir.x);
  // Anisotropic noise space: long along strokes, narrow across them.
  let alongScale  = max(0.5, p.strokeScale);
  let acrossScale = alongScale * max(1.0, p.strokeAniso);
  let aco = vec2f(dot(uv - 0.5, strokeDir) * alongScale,
                  dot(uv - 0.5, perpDir)  * acrossScale) + p.seed * 0.13;
  let n = fbm(aco);
  let grain = (vnoise(uv * 110.0 + p.seed * 1.7) - 0.5) * 0.06;
  return clamp(n + grain, 0.0, 1.0);
}

fn tonalGlazeMask(uv: vec2f) -> f32 {
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  let L = luma(cB.rgb);
  let v = select(L, 1.0 - L, p.glazeDirection == 1u);
  let bands = max(2.0, p.glazeBands);
  let q = floor(v * bands) / max(1.0, bands - 1.0);
  // glazeSoftness = 1 → continuous wash; 0 → hard bands.
  let m = mix(q, v, p.glazeSoftness);
  // Per-band wet wobble (the v term shifts the noise per tonal region).
  let wob = (fbm(uv * 3.5 + v * 7.3 + p.seed * 0.21) - 0.5) * 0.12;
  let grain = (vnoise(uv * 130.0 + p.seed * 0.7) - 0.5) * 0.05;
  return clamp(m + wob + grain, 0.0, 1.0);
}

fn edgeFirstMask(uv: vec2f) -> f32 {
  let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  // Edges reveal early (low mask), flat areas reveal late.
  let base = clamp(1.0 - eB * 2.5, 0.0, 1.0);
  let wob = (fbm(uv * max(1.0, p.edgeFirstScale) + p.seed * 0.13) - 0.5) * 0.18;
  let grain = (vnoise(uv * 120.0 + p.seed * 1.3) - 0.5) * 0.05;
  return clamp(base + wob + grain, 0.0, 1.0);
}

fn dabsMask(uv: vec2f) -> f32 {
  var minReveal = 1.0;
  for (var i = 0u; i < 128u; i = i + 1u) {
    if (i >= p.dabsCount) { break; }
    let fi = f32(i) + p.seed * 0.07 + 1.0;
    let sp = vec2f(hash21(vec2f(fi * 1.3, 13.0)), hash21(vec2f(fi * 2.7, 47.0)));
    let startT = hash21(vec2f(fi, 91.0)) * 0.5;
    let sizeJit = 0.6 + 0.8 * hash21(vec2f(fi, 11.0));
    let d = distance(uv, sp);
    let w1 = (fbm(uv * 3.0 + fi * 5.0) - 0.5) * p.dabsWobble * 0.08;
    let w2 = (vnoise(uv * 12.0 + fi * 2.0) - 0.5) * p.dabsWobble * 0.02;
    let reveal = startT + (d + w1 + w2) * (1.0 / max(p.dabsReach * sizeJit, 0.05));
    minReveal = min(minReveal, reveal);
  }
  let grain = (vnoise(uv * 120.0 + p.seed * 1.7) - 0.5) * 0.05;
  return clamp(minReveal + grain, 0.0, 1.0);
}

fn wetDensityMask(uv: vec2f) -> f32 {
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  let mx = max(max(cB.r, cB.g), cB.b);
  let mn = min(min(cB.r, cB.g), cB.b);
  let sat = select(0.0, (mx - mn) / mx, mx > 1e-4);
  let density = clamp((1.0 - luma(cB.rgb)) * 0.65 + sat * 0.35, 0.0, 1.0);
  // uv.y near 1.0 is the bottom of the canvas in this UV layout. Bottom-heavy pigment pools earliest.
  let bottomBias = uv.y * p.densityGravity * 0.5;
  let m = 1.0 - density - bottomBias * density;
  let wob = (fbm(uv * 3.0 + p.seed * 0.13) - 0.5) * 0.12;
  let grain = (vnoise(uv * 120.0 + p.seed * 1.7) - 0.5) * 0.05;
  return clamp(m + wob + grain, 0.0, 1.0);
}

fn moldTendrilMask(uv: vec2f) -> f32 {
  // Each seed spawns N tendrils as fbm-warped lines growing outward, plus
  // one level of sub-branches per tendril. Mask = "how late this pixel
  // is touched by the closest tendril" — base of tendril reveals first,
  // tip reveals last.
  var minM = 1.0;
  let nSeed = max(1u, p.moldSeedCount);
  let nTend = max(1u, p.moldTendrilsPerSeed);
  let reach = max(0.02, p.moldReach);
  let halfW = max(0.003, p.moldWidth * 0.03);
  let wobAmp = p.moldWobble;

  for (var i = 0u; i < 16u; i = i + 1u) {
    if (i >= nSeed) { break; }
    let fi = f32(i) + p.seed * 0.07 + 1.0;
    let sp = vec2f(hash21(vec2f(fi * 1.3, 13.0)), hash21(vec2f(fi * 2.7, 47.0)));

    for (var j = 0u; j < 8u; j = j + 1u) {
      if (j >= nTend) { break; }
      let fj = f32(j) + 1.0;
      let angle = hash21(vec2f(fi * 2.0 + fj, 91.0)) * 6.2831853;
      let dir = vec2f(cos(angle), sin(angle));
      let perp = vec2f(-dir.y, dir.x);
      // fbm-warp the tendril path so it curves like a real hypha.
      let wig = (fbm(uv * 6.0 + fi * 3.0 + fj * 2.0) - 0.5) * wobAmp * 0.12;
      let qw = uv + perp * wig;
      let rel = qw - sp;
      let along = dot(rel, dir);
      let perpD = dot(rel, perp);
      if (along > 0.0 && along < reach) {
        let radial = along / reach;
        let m = max(radial, abs(perpD) / halfW);
        minM = min(minM, m);
      }

      // Three sub-branches per tendril, at random offsets along the parent,
      // each going off at ±~35° with shorter reach + thinner width.
      for (var k = 0u; k < 3u; k = k + 1u) {
        let fk = f32(k) + 1.0;
        let branchAt = 0.2 + hash21(vec2f(fi * 5.0 + fj * 3.0, fk)) * 0.6;
        let branchOff = (hash21(vec2f(fi + fj * 7.0, fk * 11.0)) - 0.5) * 1.2;
        let bAngle = angle + branchOff;
        let bdir = vec2f(cos(bAngle), sin(bAngle));
        let bperp = vec2f(-bdir.y, bdir.x);
        let bsp = sp + dir * branchAt * reach;
        let bwig = (fbm(uv * 9.0 + fk * 4.0 + fj * 1.7) - 0.5) * wobAmp * 0.08;
        let bqw = uv + bperp * bwig;
        let brel = bqw - bsp;
        let balong = dot(brel, bdir);
        let bperpD = dot(brel, bperp);
        let branchReach = reach * 0.45;
        let branchHalfW = halfW * 0.65;
        if (balong > 0.0 && balong < branchReach) {
          // Branch reveals AFTER the parent reaches its branch-off point.
          let bRadial = branchAt + (balong / branchReach) * (1.0 - branchAt);
          let bm = max(bRadial, abs(bperpD) / branchHalfW);
          minM = min(minM, bm);
        }
      }
    }
  }

  let grain = (vnoise(uv * 100.0 + p.seed * 1.7) - 0.5) * 0.05;
  return clamp(minM + grain, 0.0, 1.0);
}

fn watercolorFormationMask(uv: vec2f) -> f32 {
  // Many strokes painted across the canvas, each oriented along B's local gradient.
  var bestT = 1.0;
  let nStrokes = p.formStrokeCount;
  let size = max(0.005, p.formStrokeSize);
  for (var i = 0u; i < 64u; i = i + 1u) {
    if (i >= nStrokes) { break; }
    let fi = f32(i) + p.seed * 0.07 + 1.0;
    let sp = vec2f(hash21(vec2f(fi * 1.3, 13.0)), hash21(vec2f(fi * 2.7, 47.0)));
    let startT = hash21(vec2f(fi, 91.0)) * 0.7;
    // Orientation: perpendicular to B's gradient at the stroke origin.
    let e = 0.005;
    let gx = luma(sampleFit(texB, sp + vec2f(e, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, sp - vec2f(e, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let gy = luma(sampleFit(texB, sp + vec2f(0.0, e), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, sp - vec2f(0.0, e), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let grad = vec2f(gx, gy);
    let glen = length(grad);
    let strokeDir = select(vec2f(1.0, 0.0), vec2f(-grad.y, grad.x) / glen, glen > 1e-4);
    let perpDir = vec2f(-strokeDir.y, strokeDir.x);
    let sizeJit = 0.6 + 0.7 * hash21(vec2f(fi, 11.0));
    let strokeLen = size * 3.0 * sizeJit;
    let strokeWid = size * 1.0 * sizeJit;
    let rel = uv - sp;
    let along  = dot(rel, strokeDir) / strokeLen;
    let across = dot(rel, perpDir)   / strokeWid;
    // Wobble the elliptical edge with fbm so strokes have torn watercolor borders.
    let wob = (fbm(uv * 8.0 + fi * 3.0) - 0.5) * p.formStrokeWobble * 0.5;
    let d = sqrt(along * along + across * across) * (1.0 + wob);
    if (d < 1.0) {
      let revealT = startT + d * 0.2;
      bestT = min(bestT, revealT);
    }
  }
  let grain = (fbm(uv * 60.0 + p.seed * 1.7) - 0.5) * 0.05;
  return clamp(bestT + grain, 0.0, 1.0);
}

fn cauliflowerBloomMask(uv: vec2f) -> f32 {
  // B's lightest pixels are the bloom origins (paper-show-through). Multi-octave
  // wobble breaks the iso-luma contours into cauliflower shapes.
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  let L = luma(cB.rgb);
  let base = mix(0.5, 1.0 - L, p.bloomLightBias);
  let w1 = (fbm(uv * 2.0 + p.seed * 0.13) - 0.5) * 0.20 * p.bloomWobble;
  let w2 = (fbm(uv * 6.0 + p.seed * 0.27) - 0.5) * 0.10 * p.bloomWobble;
  let w3 = (vnoise(uv * 30.0 + p.seed * 0.71) - 0.5) * 0.04;
  let grain = (vnoise(uv * 130.0 + p.seed * 0.5) - 0.5) * 0.04;
  return clamp(base + w1 + w2 + w3 + grain, 0.0, 1.0);
}

fn wetStageMask(uv: vec2f) -> f32 {
  // Watercolor painting stages: lightest wash first, darkest accents last.
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  let L = luma(cB.rgb);
  let v = 1.0 - L;
  let bands = max(2.0, p.stageBands);
  let stage = floor(v * bands);
  let withinStage = fract(v * bands);
  let stageBlend = mix(stage / bands, (stage + withinStage) / bands, p.stageOverlap);
  let wob = (fbm(uv * 3.0 + v * 5.0 + p.seed * 0.21) - 0.5) * 0.15;
  let grain = (vnoise(uv * 130.0 + p.seed * 0.7) - 0.5) * 0.04;
  return clamp(stageBlend + wob + grain, 0.0, 1.0);
}

fn wetEdgeMask(uv: vec2f) -> f32 {
  // No B → nothing to bleed in.
  if (p.validB == 0u) { return 0.0; }
  // Work in B-local coords so the "rectangle" is always B's image bounds.
  let q = (uv - p.offsetB) / p.scaleB;
  // Outside B's rect → stay as A (mask > 1 keeps the smoothstep window above t).
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return 10.0; }

  // Distance from B's rectangular border, B-local. 0 at edge → 0.5 at center.
  let dbX = min(q.x, 1.0 - q.x);
  let dbY = min(q.y, 1.0 - q.y);
  let db  = min(dbX, dbY);

  // Wavy wet front — fbm perturbs the iso-distance contours.
  let wob = (fbm(q * p.weEdgeScale + p.seed * 0.13) - 0.5) * p.weEdgeWobble * 0.18;

  // Normalize: border → 0, center → ~1.
  var m = (db + wob) * 2.0;
  // Reverse direction: bleed from center outward instead of border inward.
  if (p.weReverse == 1u) {
    m = 1.0 - m;
  }

  // Capillary tendrils: pick seed points on the border, each growing inward.
  if (p.weTendrilCount > 0u && p.weTendrilStrength > 0.001) {
    var bestInfluence = 0.0;
    for (var i = 0u; i < 32u; i = i + 1u) {
      if (i >= p.weTendrilCount) { break; }
      let fi = f32(i) + p.seed * 0.07 + 1.0;
      let side = u32(hash21(vec2f(fi * 1.3, 3.1)) * 4.0) % 4u;
      let bp = hash21(vec2f(fi * 1.7, 7.2));
      var sp: vec2f;
      var dir: vec2f;
      if (side == 0u)      { sp = vec2f(bp,   0.0); dir = vec2f(0.0,  1.0); }
      else if (side == 1u) { sp = vec2f(1.0,  bp ); dir = vec2f(-1.0, 0.0); }
      else if (side == 2u) { sp = vec2f(bp,   1.0); dir = vec2f(0.0, -1.0); }
      else                  { sp = vec2f(0.0,  bp ); dir = vec2f(1.0,  0.0); }
      let perpDir = vec2f(-dir.y, dir.x);
      // Wobble the tendril's path so it curves like a paint feeler.
      let wig = (fbm(q * 5.5 + fi * 4.3) - 0.5) * 0.05;
      let qw = q + perpDir * wig;
      let rel = qw - sp;
      let along = dot(rel, dir);
      let perp  = dot(rel, perpDir);
      let reach = max(0.01, p.weTendrilReach);
      let width = max(0.002, p.weTendrilWidth * 0.04);
      if (along > 0.0 && along < reach) {
        let perpFall  = exp(-(perp * perp) / (width * width));
        let alongFall = 1.0 - along / reach;
        bestInfluence = max(bestInfluence, perpFall * alongFall);
      }
    }
    m = m - bestInfluence * p.weTendrilStrength * 0.5;
  }

  // Detail bias — A's high-detail regions reveal earlier (paint hangs in soft areas).
  if (p.weDetailBias > 0.001) {
    let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
    m = m - eA * p.weDetailBias * 0.35;
  }

  // B detail bias — front "reaches toward" B's focal points so they reveal first.
  if (p.weBDetailBias > 0.001) {
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
    m = m - eB * p.weBDetailBias * 0.35;
  }

  // B luma bias — positive: dark areas of B reveal first; negative: lights first.
  if (abs(p.weBLumaBias) > 0.001) {
    let lB = luma(sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    m = m - (lB - 0.5) * p.weBLumaBias * 0.5;
  }

  return clamp(m, 0.0, 1.0);
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

  // Stretch t so the per-pixel smoothstep window (mask±spread) is fully
  // traversed for t ∈ [0,1] — without this, pixels with mask near 0 or 1
  // never fully reveal at the timeline's endpoints.
  let sp = mix(0.05, 0.7, p.spread);
  let tCurve = applyCurve(p.t, p.curve);
  let t = tCurve * (1.0 + 2.0 * sp) - sp;
  let env = pow(sin(3.14159265 * clamp(p.t, 0.0, 1.0)), 0.85);

  let cA = sampleFit(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
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
  } else if (p.mode == 15u) {
    mask = wetEdgeMask(uv);
  } else if (p.mode == 16u) {
    mask = strokeFollowMask(uv);
  } else if (p.mode == 17u) {
    mask = tonalGlazeMask(uv);
  } else if (p.mode == 18u) {
    mask = edgeFirstMask(uv);
  } else if (p.mode == 19u) {
    // Painterly flow: simple organic mask. The character comes from the
    // gradient-warped B sample below, not the mask.
    let n = fbm(uv * 1.8 + p.seed * 0.13);
    let grain = (vnoise(uv * 120.0 + p.seed * 1.7) - 0.5) * 0.05;
    mask = clamp(n + grain, 0.0, 1.0);
  } else if (p.mode == 20u) {
    mask = dabsMask(uv);
  } else if (p.mode == 21u) {
    mask = wetDensityMask(uv);
  } else if (p.mode == 22u) {
    mask = moldTendrilMask(uv);
  } else if (p.mode == 23u) {
    mask = watercolorFormationMask(uv);
  } else if (p.mode == 24u) {
    mask = cauliflowerBloomMask(uv);
  } else if (p.mode == 25u) {
    mask = wetStageMask(uv);
  } else if (p.mode == 26u) {
    // Pigment migration uses a soft organic mask; the cB warp below does the work.
    let n = fbm(uv * 1.5 + p.seed * 0.13);
    let grain = (vnoise(uv * 120.0 + p.seed * 1.7) - 0.5) * 0.04;
    mask = clamp(n + grain, 0.0, 1.0);
  } else {
    let eA = edgeMag(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
    mask = organicMask(uv, lA, lB, max(eA, eB));
  }

  // Global mask shift: lets the user rebalance any mode's mask distribution
  // earlier (negative) or later (positive) without touching its inner logic.
  // Useful for image-driven masks whose values cluster around the source's
  // tonal distribution rather than spreading evenly across [0,1].
  mask = clamp(mask + p.maskShift, 0.0, 1.0);
  var mixT = clamp(smoothstep(mask - sp, mask + sp, t), 0.0, 1.0);

  // ---- wet diffusion (mode 4): anticipatory tint of B into A ----
  var colA_eff = cA.rgb;
  // ---- wet edge (mode 15): anticipatory bleed of B into A ahead of the front ----
  if (p.mode == 15u && p.weBleed > 0.001) {
    let anticipate = smoothstep(mask - 0.35, mask + 0.05, t);
    let bR = 0.02;
    var acc = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
    var wsum = 1.0;
    for (var i = 0u; i < 6u; i = i + 1u) {
      let a = f32(i) * (6.2831853 / 6.0) + p.seed * 0.017;
      let d = vec2f(cos(a), sin(a));
      acc = acc + sampleFit(texB, uv + d * bR, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
      wsum = wsum + 1.0;
    }
    let bleedB = acc / wsum;
    let dry = 1.0 - mixT;
    colA_eff = mix(cA.rgb, bleedB, anticipate * dry * p.weBleed * 0.4);
  }
  if (p.mode == 4u && p.diffStrength > 0.001) {
    let anticipate = smoothstep(mask - 0.45, mask + 0.05, t);
    let bR = 0.025 + p.diffRadius * 0.08;
    // simple 12-tap soft blur of B
    var acc = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb * 0.35;
    var wsum = 0.35;
    for (var i = 0u; i < 12u; i = i + 1u) {
      let a = f32(i) * (6.2831853 / 12.0) + p.seed * 0.013;
      let d = vec2f(cos(a), sin(a));
      let rr = select(0.55, 1.0, (i % 2u) == 0u);
      let w = 1.0 - rr * 0.45;
      acc = acc + sampleFit(texB, uv + d * rr * bR, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb * w;
      wsum = wsum + w;
    }
    let bleedB = acc / wsum;
    let dry = 1.0 - mixT;
    colA_eff = mix(cA.rgb, bleedB, anticipate * dry * p.diffStrength * 0.55);
  }

  // Mode 19 (painterly flow): sample B at a position warped along its own
  // gradient field, so paint "flows into place" as t→1.
  var cB_eff = cB.rgb;
  if (p.mode == 19u) {
    let ee = 0.005;
    let gx = luma(sampleFit(texB, uv + vec2f( ee, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, uv - vec2f( ee, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let gy = luma(sampleFit(texB, uv + vec2f(0.0,  ee), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, uv - vec2f(0.0,  ee), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let grad = vec2f(gx, gy);
    let glen = length(grad);
    if (glen > 1e-4) {
      let flowDir = vec2f(-grad.y, grad.x) / glen;
      let baseAmt = (1.0 - tCurve) * p.flowAmount * 0.18;
      // Wobble the warp amount so streaks aren't uniform — analog feel.
      let wob = (fbm(uv * 4.0 + p.seed * 0.21) - 0.5) * 0.5;
      cB_eff = sampleFit(texB, uv + flowDir * baseAmt * (1.0 + wob), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
    }
  }

  // Mode 26 (pigment migration): sample B at a position offset along (or
  // perpendicular to) B's own gradient; offset shrinks as t→1 so pigment
  // "flows into place".
  if (p.mode == 26u) {
    let ee = 0.005;
    let gx = luma(sampleFit(texB, uv + vec2f( ee, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, uv - vec2f( ee, 0.0), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let gy = luma(sampleFit(texB, uv + vec2f(0.0,  ee), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb)
           - luma(sampleFit(texB, uv - vec2f(0.0,  ee), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb);
    let grad = vec2f(gx, gy);
    let glen = length(grad);
    if (glen > 1e-4) {
      let dirAlong = select(grad / glen, vec2f(-grad.y, grad.x) / glen, p.migrationDir == 1u);
      let baseAmt = (1.0 - tCurve) * p.migrationStrength * 0.28;
      // Multi-scale turbulence on the displacement magnitude.
      let turb1 = (fbm(uv * 3.0  + p.seed * 0.21) - 0.5) * 0.6;
      let turb2 = (fbm(uv * 12.0 + p.seed * 0.47) - 0.5) * 0.3;
      let turbMul = 1.0 + (turb1 + turb2) * p.migrationTurb;
      cB_eff = sampleFit(texB, uv + dirAlong * baseAmt * turbMul, p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
    }
  }

  var outc = mix(colA_eff, cB_eff, mixT);

  // Tonal glaze warm tint (mode 17): pull glaze color slightly toward warm
  // pigment as it dries — a faint chromatic "settling" you see in real paint.
  if (p.mode == 17u && p.glazeWarm > 0.001) {
    let warmTint = outc * vec3f(1.04, 1.0, 0.94);
    outc = mix(outc, warmTint, p.glazeWarm * mixT);
  }

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

  // ---- wet edge dry-ring (mode 15): dark watercolor bead at the wet front ----
  if (p.mode == 15u && p.weDryRing > 0.001 && env > 0.02) {
    let ringW = 0.03;
    let band = exp(-pow((t - mask) / ringW, 2.0));
    let base = mix(cA.rgb, cB.rgb, 0.6);
    let lm = luma(base);
    let darker = clamp(base * 0.4 + vec3f(lm * 0.08), vec3f(0.0), vec3f(1.0));
    outc = mix(outc, darker, band * p.weDryRing * env * 0.55);
  }

  // ---- pigment run drip (mode 9) ----
  if (p.mode == 9u && p.runDrip > 0.001 && env > 0.02 && t > mask) {
    let dripBand = exp(-pow((t - mask) / 0.08, 2.0));
    let dripB = sampleFit(texB, uv + vec2f(0.0, p.runDrip * 0.05), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
    outc = mix(outc, dripB, dripBand * p.runDrip * env * 0.35);
  }

  // ---- edge underdrawing sketch overlay (mode 18) ----
  if (p.mode == 18u && p.edgeFirstInk > 0.001) {
    let eB = edgeMag(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
    // Ink ramps in fast, then fades as color floods in.
    let inUp = smoothstep(0.0, 0.18, p.t);
    let inDn = 1.0 - smoothstep(p.edgeFirstFade, p.edgeFirstFade + 0.18, p.t);
    let inkColor = vec3f(0.04, 0.03, 0.025);
    let inkAmt = clamp(eB * 3.0, 0.0, 1.0) * inUp * inDn * p.edgeFirstInk;
    outc = mix(outc, inkColor, inkAmt);
  }

  // ---- wet-density vertical smear (mode 21) ----
  if (p.mode == 21u && p.densitySmear > 0.001 && env > 0.02) {
    let anticipate = smoothstep(mask - 0.25, mask + 0.05, t) * (1.0 - mixT);
    let smearB = sampleFit(texB, uv - vec2f(0.0, 0.025), p.scaleB, p.offsetB, p.validB, p.slotBColor).rgb;
    outc = mix(outc, smearB, anticipate * p.densitySmear * env * 0.5);
  }

  // ---- watercolor character (modes 23..26): dark wet rim + paper-show pop + granulation ----
  if (p.mode >= 23u && p.mode <= 26u && env > 0.02) {
    // Cauliflower-style dark rim at the wet front
    let rimW = 0.04;
    let band = exp(-pow((t - mask) / rimW, 2.0));
    let baseRim = mix(cA.rgb, cB_eff, 0.55);
    let darker = clamp(baseRim * 0.42, vec3f(0.0), vec3f(1.0));
    outc = mix(outc, darker, band * env * 0.45);
    // Paper-show-through pop: B's brightest pixels briefly flash even brighter
    // (mimics paper exposed through the wash)
    let lumB = luma(cB_eff);
    let popThresh = 0.7;
    let popPhase = smoothstep(0.2, 0.5, mixT) * (1.0 - smoothstep(0.65, 0.95, mixT));
    let popAmt = max(0.0, lumB - popThresh) / max(1e-4, 1.0 - popThresh);
    outc = mix(outc, vec3f(0.95, 0.93, 0.88), popAmt * popPhase * 0.65);
    // Granulation: high-freq + mid-freq per-pixel value variation
    let g1 = (vnoise(uv * 220.0 + p.seed * 1.7) - 0.5) * 0.08;
    let g2 = (vnoise(uv *  70.0 + p.seed * 0.7) - 0.5) * 0.04;
    outc = clamp(outc + vec3f(g1 + g2) * mixT * 0.65, vec3f(0.0), vec3f(1.0));
  }

  // ---- global transition bounds: outside the box, restore unmodified A ----
  if (p.boundsEnable == 1u) {
    let hw = p.boundsW * 0.5;
    let hh = p.boundsH * 0.5;
    let dx = abs(uv.x - p.boundsCx) - hw;
    let dy = abs(uv.y - p.boundsCy) - hh;
    let outsideDist = max(dx, dy);
    var bMul = 1.0;
    if (p.boundsSoftness > 0.0001) {
      bMul = 1.0 - smoothstep(0.0, p.boundsSoftness, outsideDist);
    } else if (outsideDist > 0.0) {
      bMul = 0.0;
    }
    outc = mix(cA.rgb, outc, bMul);
  }

  // ---- global paper grain (all modes, opt-in) ----
  if (p.paperGrain > 0.001) {
    let fib = paperFiber(uv);
    let mul = 1.0 + fib * p.paperGrain * 0.18;
    outc = clamp(outc * mul, vec3f(0.0), vec3f(1.0));
    let warmShift = outc * vec3f(1.02, 0.995, 0.96);
    outc = mix(outc, warmShift, p.paperGrain * 0.15);
  }

  // Keep A as unchanged background outside B's rect when toggled — useful when
  // B is smaller than the canvas (e.g. contained / zoomed-out).
  var effMixT = mixT;
  if (p.keepAOutsideB == 1u) {
    let q = (uv - p.offsetB) / p.scaleB;
    if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) {
      outc = cA.rgb;
      effMixT = 0.0;
    }
  }
  // Per-slot alpha: 0 when that slot is in 'transparent' mode (valid==3u), 1 otherwise.
  // Final alpha mixes the same way as RGB; output premultiplied for correct
  // canvas compositing and AE imports.
  let alphaA = select(1.0, 0.0, p.validA == 3u);
  let alphaB = select(1.0, 0.0, p.validB == 3u);
  let alpha = mix(alphaA, alphaB, effMixT);
  let rgb = clamp(outc, vec3f(0.0), vec3f(1.0));
  return vec4f(rgb * alpha, alpha);
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
  irisFocus: vec2f, irisJitter: f32, irisUniform: u32,
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  advSeedCount: u32, advSeedRadius: f32, canvasAspect: f32, _p5: f32,
  weEdgeScale: f32, weEdgeWobble: f32, weDryRing: f32, weBleed: f32,
  weTendrilCount: u32, weTendrilReach: f32, weTendrilWidth: f32, weTendrilStrength: f32,
  weDetailBias: f32, moldTendrilsPerSeed: u32, weReverse: u32, weBDetailBias: f32,
  moldWidth: f32, moldWobble: f32, moldSeedCount: u32, moldReach: f32,
  strokeScale: f32, strokeAniso: f32, glazeBands: f32, glazeSoftness: f32,
  glazeDirection: u32, glazeWarm: f32, edgeFirstInk: f32, edgeFirstFade: f32,
  edgeFirstScale: f32, flowAmount: f32, dabsCount: u32, dabsReach: f32,
  dabsWobble: f32, densityGravity: f32, densitySmear: f32, paperGrain: f32,
  formStrokeCount: u32, formStrokeSize: f32, formStrokeWobble: f32, _f1: f32,
  bloomLightBias: f32, bloomWobble: f32, bloomPaperShow: f32, _b1: f32,
  stageBands: f32, stageOverlap: f32, _s1: f32, _s2: f32,
  migrationStrength: f32, migrationDir: u32, migrationTurb: f32, _m1: f32,
  boundsEnable: u32, boundsCx: f32, boundsCy: f32, boundsW: f32,
  boundsH: f32, boundsSoftness: f32, weBLumaBias: f32, maskShift: f32,
  slotAColor: vec3f, keepAOutsideB: u32,
  slotBColor: vec3f, _slotBPad: f32,
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
fn sampleFit(tex: texture_2d<f32>, uv: vec2f, scale: vec2f, offset: vec2f, valid: u32, color: vec3f) -> vec4f {
  if (valid == 0u) { return vec4f(p.bg, 1.0); }
  if (valid == 2u) { return vec4f(color, 1.0); }
  if (valid == 3u) { return vec4f(0.0, 0.0, 0.0, 0.0); }
  let q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) { return vec4f(p.bg, 1.0); }
  return textureSampleLevel(tex, samp, q, 0.0);
}

fn curlField(uv: vec2f) -> vec2f {
  let e = 0.004;
  let pos = uv * p.advCurlScale;
  let pyp = fbm(pos + vec2f(0.0, e));
  let pyn = fbm(pos - vec2f(0.0, e));
  let pxp = fbm(pos + vec2f(e, 0.0));
  let pxn = fbm(pos - vec2f(e, 0.0));
  return vec2f(pyp - pyn, pxn - pxp);
}

@fragment fn fs(in: VSOut) -> @location(0) vec4f {
  let uv = in.uv;
  let dims = vec2f(textureDimensions(stateIn));
  let px = 1.0 / dims;

  let cA = sampleFit(texA, uv, p.scaleA, p.offsetA, p.validA, p.slotAColor);
  let cB = sampleFit(texB, uv, p.scaleB, p.offsetB, p.validB, p.slotBColor);
  let lA = luma(cA.rgb); let lB = luma(cB.rgb);

  // ---- read previous state, optionally pre-advected (variant 2: curl) ----
  var cur: vec3f;
  if (p.advVariant == 2u && p.advCurlStr > 0.001) {
    let vel = curlField(uv) * p.advCurlStr * 0.06;
    cur = textureSampleLevel(stateIn, samp, uv - vel, 0.0).rgb;
  } else {
    cur = textureSampleLevel(stateIn, samp, uv, 0.0).rgb;
  }

  // ---- diffusion kernel ----
  var nb: vec3f;
  if (p.advVariant == 1u) {
    // Gravity: anisotropic kernel along the flow direction.
    let a = p.advGravAngle * 6.2831853;
    let flowDir = vec2f(sin(a), -cos(a));
    let perp = vec2f(-flowDir.y, flowDir.x);
    let streak = 1.0 + p.advGravStreak * 2.5;
    let upstep   = -flowDir * px.x * streak;
    let downstep =  flowDir * px.x * streak;
    let latstep  =  perp    * px.x;
    let wUp   = 1.0 + p.advGravity * 1.6;
    let wDown = max(0.0, 1.0 - p.advGravity * 0.85);
    let wLat  = 0.3 + p.advGravLateral * 1.4;
    let wDiagU = (wUp   + wLat) * 0.5;
    let wDiagD = (wDown + wLat) * 0.5;
    let wsum   = wUp + wDown + 2.0 * wLat + 2.0 * wDiagU + 2.0 * wDiagD;
    nb = (
      textureSampleLevel(stateIn, samp, uv + upstep, 0.0).rgb * wUp +
      textureSampleLevel(stateIn, samp, uv + downstep, 0.0).rgb * wDown +
      textureSampleLevel(stateIn, samp, uv + latstep, 0.0).rgb * wLat +
      textureSampleLevel(stateIn, samp, uv - latstep, 0.0).rgb * wLat +
      textureSampleLevel(stateIn, samp, uv + upstep * 0.7071 + latstep * 0.7071, 0.0).rgb * wDiagU +
      textureSampleLevel(stateIn, samp, uv + upstep * 0.7071 - latstep * 0.7071, 0.0).rgb * wDiagU +
      textureSampleLevel(stateIn, samp, uv + downstep * 0.7071 + latstep * 0.7071, 0.0).rgb * wDiagD +
      textureSampleLevel(stateIn, samp, uv + downstep * 0.7071 - latstep * 0.7071, 0.0).rgb * wDiagD
    ) / wsum;
  } else if (p.advVariant == 3u) {
    // Brush-channel: diffusion along A's local stroke direction.
    let e = 0.003;
    let gx = luma(sampleFit(texA, uv + vec2f(e, 0.0), p.scaleA, p.offsetA, p.validA, p.slotAColor).rgb) -
             luma(sampleFit(texA, uv - vec2f(e, 0.0), p.scaleA, p.offsetA, p.validA, p.slotAColor).rgb);
    let gy = luma(sampleFit(texA, uv + vec2f(0.0, e), p.scaleA, p.offsetA, p.validA, p.slotAColor).rgb) -
             luma(sampleFit(texA, uv - vec2f(0.0, e), p.scaleA, p.offsetA, p.validA, p.slotAColor).rgb);
    let grad = vec2f(gx, gy);
    let glen = length(grad);
    let sd = select(vec2f(1.0, 0.0), vec2f(-grad.y, grad.x) / glen, glen > 1e-4);
    let perp = vec2f(-sd.y, sd.x);
    let follow = p.advBrushFollow;
    let wAlong = 1.0 + follow * 1.5;
    let wPerp  = max(0.0, 1.0 - follow * 0.85);
    nb = (
      textureSampleLevel(stateIn, samp, uv + sd * px.x * 1.4, 0.0).rgb * wAlong +
      textureSampleLevel(stateIn, samp, uv - sd * px.x * 1.4, 0.0).rgb * wAlong +
      textureSampleLevel(stateIn, samp, uv + perp * px.x, 0.0).rgb * wPerp +
      textureSampleLevel(stateIn, samp, uv - perp * px.x, 0.0).rgb * wPerp
    ) / (2.0 * wAlong + 2.0 * wPerp + 1e-4);
  } else {
    // Isotropic 8-tap (variants 0, 2, 4)
    nb = (
      textureSampleLevel(stateIn, samp, uv + vec2f(px.x, 0.0), 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv - vec2f(px.x, 0.0), 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv + vec2f(0.0, px.y), 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv - vec2f(0.0, px.y), 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv + px * 0.7071, 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv - px * 0.7071, 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv + vec2f(px.x, -px.y) * 0.7071, 0.0).rgb +
      textureSampleLevel(stateIn, samp, uv + vec2f(-px.x, px.y) * 0.7071, 0.0).rgb
    ) * 0.125;
  }
  let diffused = mix(cur, nb, p.advVisc);

  // ---- mask (variant-dependent) ----
  var mask: f32;
  let n1 = fbm(uv * p.maskScale + p.seed * 0.13);
  let n2 = fbm(uv * p.maskScale * 2.3 + 17.0 + p.seed * 0.09);
  let noiseMask = mix(n1, n2, 0.35);
  let lumMask = 0.5 + 0.5 * (lB - lA);

  if (p.advVariant == 1u) {
    let a = p.advGravAngle * 6.2831853;
    let flowDir = vec2f(sin(a), -cos(a));
    let flowProgress = 0.5 - dot(uv - 0.5, flowDir);
    let gMask = mix(lA, flowProgress, p.advGravBias);
    mask = mix(noiseMask, gMask, p.organic);
  } else if (p.advVariant == 4u) {
    // Seed-point: random hash-based seed positions
    var minD = 9999.0;
    for (var i = 0u; i < 16u; i = i + 1u) {
      if (i >= p.advSeedCount) { break; }
      let fi = f32(i) + p.seed * 0.07 + 1.0;
      let sp = vec2f(hash21(vec2f(fi * 1.3, 13.0)), hash21(vec2f(fi * 2.7, 47.0)));
      minD = min(minD, distance(uv, sp));
    }
    mask = clamp(minD / max(p.advSeedRadius, 0.05), 0.0, 1.0);
  } else {
    mask = mix(noiseMask, lumMask, p.organic);
  }

  let sp = mix(0.1, 0.5, p.spread);
  // Stretch t so end-of-timeline pixels fully reveal — same fix as display shader.
  let tR = p.t * (1.0 + 2.0 * sp) - sp;
  let reveal = smoothstep(mask - sp, mask + sp * 0.3, tR);
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
  irisFocus: vec2f, irisJitter: f32, irisUniform: u32,
  bleedFinger: f32, bleedAmount: f32, bleedHalo: f32, runGravity: f32,
  runDrip: f32, advVariant: u32, advVisc: f32, advRate: f32,
  advGravity: f32, advGravBias: f32, advGravAngle: f32, advGravStreak: f32,
  advGravLateral: f32, advCurlStr: f32, advCurlScale: f32, advBrushFollow: f32,
  advSeedCount: u32, advSeedRadius: f32, canvasAspect: f32, _p5: f32,
  weEdgeScale: f32, weEdgeWobble: f32, weDryRing: f32, weBleed: f32,
  weTendrilCount: u32, weTendrilReach: f32, weTendrilWidth: f32, weTendrilStrength: f32,
  weDetailBias: f32, moldTendrilsPerSeed: u32, weReverse: u32, weBDetailBias: f32,
  moldWidth: f32, moldWobble: f32, moldSeedCount: u32, moldReach: f32,
  strokeScale: f32, strokeAniso: f32, glazeBands: f32, glazeSoftness: f32,
  glazeDirection: u32, glazeWarm: f32, edgeFirstInk: f32, edgeFirstFade: f32,
  edgeFirstScale: f32, flowAmount: f32, dabsCount: u32, dabsReach: f32,
  dabsWobble: f32, densityGravity: f32, densitySmear: f32, paperGrain: f32,
  formStrokeCount: u32, formStrokeSize: f32, formStrokeWobble: f32, _f1: f32,
  bloomLightBias: f32, bloomWobble: f32, bloomPaperShow: f32, _b1: f32,
  stageBands: f32, stageOverlap: f32, _s1: f32, _s2: f32,
  migrationStrength: f32, migrationDir: u32, migrationTurb: f32, _m1: f32,
  boundsEnable: u32, boundsCx: f32, boundsCy: f32, boundsW: f32,
  boundsH: f32, boundsSoftness: f32, weBLumaBias: f32, maskShift: f32,
  slotAColor: vec3f, keepAOutsideB: u32,
  slotBColor: vec3f, _slotBPad: f32,
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
const UBO_SIZE = 512;
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
  duration: 15.0,
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
  saltDensity: 0.0, saltContrast: 0.55,
  saltSource: 1, saltBias: 0.6, saltImage: 2,
  irisFocusX: 0.5, irisFocusY: 0.5, irisJitter: 0.35, irisUniform: true,
  bleedFinger: 0.5, bleedAmount: 0.45, bleedHalo: 0.5,
  runGravity: 0.5, runDrip: 0.35,
  // advection family
  advecVisc: 0.55, advecRate: 0.18, advecSteps: 3,
  advecGravity: 0.6, advecGravBias: 0.5,
  advecGravAngle: 0, advecGravStreak: 0.4, advecGravLateral: 0.3,
  advecCurlStr: 0.5, advecCurlScale: 2.5,
  advecBrushFollow: 0.7,
  advecSeedCount: 5, advecSeedRadius: 0.45,
  // wet edge (mode 15)
  weEdgeScale: 6.0, weEdgeWobble: 0.55,
  weDryRing: 0.45, weBleed: 0.5,
  weTendrilCount: 6, weTendrilReach: 0.4, weTendrilWidth: 0.5, weTendrilStrength: 0.55,
  weDetailBias: 0.35,
  weReverse: false, weBDetailBias: 0.0, weBLumaBias: 0.0,
  // stroke follow (mode 16)
  strokeScale: 6.0, strokeAniso: 4.0,
  // tonal glaze (mode 17)
  glazeBands: 3.0, glazeSoftness: 0.55, glazeDirection: 0, glazeWarm: 0.35,
  // edge underdrawing (mode 18)
  edgeFirstInk: 0.55, edgeFirstFade: 0.35, edgeFirstScale: 3.0,
  // painterly flow (mode 19)
  flowAmount: 0.55,
  // color-pool dabs (mode 20)
  dabsCount: 28, dabsReach: 0.32, dabsWobble: 0.6,
  // wet-density gravity (mode 21)
  densityGravity: 0.45, densitySmear: 0.45,
  // global paper grain (Style folder)
  paperGrain: 0.25,
  // mold tendrils (mode 22) — direct fbm-warped tendril paths from seeds
  moldSeedCount: 5, moldTendrilsPerSeed: 4,
  moldReach: 0.5, moldWidth: 0.35, moldWobble: 0.6,
  // mode 23 watercolor formation
  formStrokeCount: 32, formStrokeSize: 0.05, formStrokeWobble: 0.5,
  // mode 24 cauliflower bloom storm
  bloomLightBias: 0.85, bloomWobble: 0.5, bloomPaperShow: 0.6,
  // mode 25 wet-stage layering
  stageBands: 4, stageOverlap: 0.5,
  // mode 26 pigment migration
  migrationStrength: 0.6, migrationDir: 0, migrationTurb: 0.5,
  // global transition bounds (Style folder)
  boundsEnable: false, boundsCx: 0.5, boundsCy: 0.5, boundsW: 0.6, boundsH: 0.6, boundsSoftness: 0.03,
  // global mask timing shift (Dissolve folder)
  maskShift: 0,
  // per-slot fill modes: 'image' | 'solid' | 'transparent' (alpha output)
  slotAFillMode: 'image', slotAColor: '#000000',
  slotBFillMode: 'image', slotBColor: '#000000',
  // When on, anything outside B's rect stays as unmodified A — useful when B
  // is smaller than the canvas and you want A as a persistent background.
  keepAOutsideB: false,
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
    // True cover: scale image so the smaller-relative axis matches the canvas,
    // the larger axis extends past the canvas and gets cropped. Aspect preserved.
    if (ia > ca) {
      // Image wider than canvas → match canvas height, image overhangs left/right.
      const sx = ia / ca;
      return { sx, sy: 1, ox: (1 - sx) * 0.5, oy: 0 };
    }
    // Image more square / taller than canvas → match canvas width, overhangs top/bottom.
    const sy = ca / ia;
    return { sx: 1, sy, ox: 0, oy: (1 - sy) * 0.5 };
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
  // Size to whichever slot is actually rendering an image. When A is solid /
  // transparent the canvas snaps to B's dimensions so the recorder exports at
  // B's exact size (and vice versa).
  const aReal = state.slotAFillMode === 'image' && state.imgA;
  const bReal = state.slotBFillMode === 'image' && state.imgB;
  const ref = aReal ? state.imgA : (bReal ? state.imgB : (state.imgA || state.imgB));
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
  // valid encoding: 0=no image (bg), 1=image, 2=solid color, 3=transparent.
  uboU32[6]  = state.slotAFillMode === 'solid' ? 2 : state.slotAFillMode === 'transparent' ? 3 : (state.imgA ? 1 : 0);
  uboU32[7]  = state.slotBFillMode === 'solid' ? 2 : state.slotBFillMode === 'transparent' ? 3 : (state.imgB ? 1 : 0);
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
  uboU32[43] = state.irisUniform ? 1 : 0;
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
  // -- 60..63 -- seed + canvas aspect
  uboU32[60] = state.advecSeedCount;
  uboF32[61] = state.advecSeedRadius;
  uboF32[62] = ch > 0 ? cw / ch : 1.0;
  uboF32[63] = 0;
  // -- 64..67 -- wet edge (mode 15): rect ingress
  uboF32[64] = state.weEdgeScale;
  uboF32[65] = state.weEdgeWobble;
  uboF32[66] = state.weDryRing;
  uboF32[67] = state.weBleed;
  // -- 68..71 -- wet edge: tendrils
  uboU32[68] = state.weTendrilCount;
  uboF32[69] = state.weTendrilReach;
  uboF32[70] = state.weTendrilWidth;
  uboF32[71] = state.weTendrilStrength;
  // -- 72..75 -- detail bias + new wet-edge biases (reverse, B detail)
  uboF32[72] = state.weDetailBias;
  // slot 73 (moldTendrilsPerSeed) is written below in the mold block
  uboU32[74] = state.weReverse ? 1 : 0;
  uboF32[75] = state.weBDetailBias;
  // -- 73 -- mold tendrils per-seed count (slot was _p6)
  uboU32[73] = state.moldTendrilsPerSeed;
  // -- 76..79 -- mold tendrils (mode 22): direct path approach
  uboF32[76] = state.moldWidth;
  uboF32[77] = state.moldWobble;
  uboU32[78] = state.moldSeedCount;
  uboF32[79] = state.moldReach;
  // -- 96..111 -- new strong watercolor modes (23..26)
  uboU32[96]  = state.formStrokeCount;
  uboF32[97]  = state.formStrokeSize;
  uboF32[98]  = state.formStrokeWobble;
  uboF32[99]  = 0;
  uboF32[100] = state.bloomLightBias;
  uboF32[101] = state.bloomWobble;
  uboF32[102] = state.bloomPaperShow;
  uboF32[103] = 0;
  uboF32[104] = state.stageBands;
  uboF32[105] = state.stageOverlap;
  uboF32[106] = 0; uboF32[107] = 0;
  uboF32[108] = state.migrationStrength;
  uboU32[109] = state.migrationDir;
  uboF32[110] = state.migrationTurb;
  uboF32[111] = 0;
  // -- 112..119 -- global transition bounds
  uboU32[112] = state.boundsEnable ? 1 : 0;
  uboF32[113] = state.boundsCx;
  uboF32[114] = state.boundsCy;
  uboF32[115] = state.boundsW;
  uboF32[116] = state.boundsH;
  uboF32[117] = state.boundsSoftness;
  uboF32[118] = state.weBLumaBias;
  uboF32[119] = state.maskShift;
  // -- 120..127 -- per-slot solid colors (used when fill mode = 'solid')
  const ca = hexToRgb(state.slotAColor);
  const cb = hexToRgb(state.slotBColor);
  uboF32[120] = ca[0]; uboF32[121] = ca[1]; uboF32[122] = ca[2];
  uboU32[123] = state.keepAOutsideB ? 1 : 0;
  uboF32[124] = cb[0]; uboF32[125] = cb[1]; uboF32[126] = cb[2]; uboF32[127] = 0;
  // -- 80..95 -- new painterly modes (16..21) + global paper grain
  uboF32[80] = state.strokeScale;
  uboF32[81] = state.strokeAniso;
  uboF32[82] = state.glazeBands;
  uboF32[83] = state.glazeSoftness;
  uboU32[84] = state.glazeDirection;
  uboF32[85] = state.glazeWarm;
  uboF32[86] = state.edgeFirstInk;
  uboF32[87] = state.edgeFirstFade;
  uboF32[88] = state.edgeFirstScale;
  uboF32[89] = state.flowAmount;
  uboU32[90] = state.dabsCount;
  uboF32[91] = state.dabsReach;
  uboF32[92] = state.dabsWobble;
  uboF32[93] = state.densityGravity;
  uboF32[94] = state.densitySmear;
  uboF32[95] = state.paperGrain;

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
        if (typeof updateTransportLabels !== 'undefined') updateTransportLabels();
      }
    }
    state.t = state.reverse ? (1 - pT) : pT;
    if (typeof bT !== 'undefined') bT.refresh();
  }
  renderFrame();
  requestAnimationFrame(render);
}

// Synchronous GPU draw — used by the rAF render loop and by the recorder.
function renderFrame() {
  if (!state.imgA && !state.imgB) return;
  writeUniforms();

  const isAdvec = (state.mode >= 10 && state.mode <= 14);
  let finalState = null;
  const enc = device.createCommandEncoder();

  if (isAdvec) {
    ensureStateTextures();
    if (advec.needsReset || state.t < advec.lastT - 0.03 || state.t === 0) {
      const initPass = enc.beginRenderPass({
        colorAttachments: [{
          view: stateTexA.createView(),
          clearValue: { r: 0, g: 0, b: 0, a: 1 },
          loadOp: 'clear', storeOp: 'store',
        }],
      });
      initPass.setPipeline(initPipeline);
      initPass.setBindGroup(0, makeSimBindGroup(stateTexB));
      initPass.draw(6);
      initPass.end();
      advec.src = 'A';
      advec.lastT = 0;
      advec.needsReset = false;
      const warm = Math.max(8, Math.round(state.advecSteps * 8));
      for (let i = 0; i < warm; i++) {
        runSimStepInto(enc, state.t * ((i + 1) / warm));
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

// IndexedDB: 'images' tracks the last A/B (key 'imageA'/'imageB'); 'library'
// is the persistent gallery of every image the user has ever loaded.
const IDB_NAME = 'transition-tool-v3';
const IDB_STORE = 'images';
const IDB_LIB_STORE = 'library';
function idbOpen() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = () => {
      const db = r.result;
      if (!db.objectStoreNames.contains(IDB_STORE)) db.createObjectStore(IDB_STORE);
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
    return await new Promise((resolve, reject) => {
      const req = db.transaction(IDB_STORE, 'readonly').objectStore(IDB_STORE).get(key);
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  } catch { return null; }
}
async function idbPut(key, value) {
  try {
    const db = await idbOpen();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).put(value, key);
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {}
}
async function idbClearAll() {
  try {
    const db = await idbOpen();
    return await new Promise((resolve, reject) => {
      const tx = db.transaction(IDB_STORE, 'readwrite');
      tx.objectStore(IDB_STORE).clear();
      tx.oncomplete = () => resolve();
      tx.onerror = () => reject(tx.error);
    });
  } catch {}
}

// ---- library store (persistent gallery of all uploaded images) ----
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
// Generate a small JPEG thumbnail (~256px wide) from any image blob/file.
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

function loadFile(file, slot) {
  if (!file.type.startsWith('image/')) return;
  idbPut(slot === 'A' ? 'imageA' : 'imageB', file);  // persist the blob
  loadFromUrl(URL.createObjectURL(file), slot);
  // Also persist into the library (one entry per new file).
  addToLibrary(file).then(() => renderLibrary());
}
async function addToLibrary(file) {
  const thumb = await makeThumb(file, 256);
  if (!thumb) return null;
  return libAdd({
    blob: file,
    thumb,
    addedAt: Date.now(),
    name: file.name || 'untitled',
  });
}

// ---- library UI ----
const libGridEl = document.getElementById('library-grid');
const _libThumbUrls = new Map();  // id -> object URL (revoked on re-render)
let _libCache = [];

function libRevokeAll() {
  for (const url of _libThumbUrls.values()) URL.revokeObjectURL(url);
  _libThumbUrls.clear();
}
function activeBlobIds() {
  // Track which library entries correspond to the slots' current images. We
  // compare by blob identity since loadFile keeps a single source-of-truth.
  return { A: state._libIdA || null, B: state._libIdB || null };
}
async function renderLibrary() {
  _libCache = await libList();
  libRevokeAll();
  libGridEl.innerHTML = '';
  if (_libCache.length === 0) {
    const empty = document.createElement('div');
    empty.className = 'library-empty';
    empty.textContent = 'drop or pick images into A/B — they show here';
    libGridEl.appendChild(empty);
    return;
  }
  const { A: idA, B: idB } = activeBlobIds();
  for (const entry of _libCache) {
    const url = URL.createObjectURL(entry.thumb);
    _libThumbUrls.set(entry.id, url);
    const tile = document.createElement('div');
    tile.className = 'library-thumb' + (entry.id === idA ? ' in-A' : '') + (entry.id === idB ? ' in-B' : '');
    tile.title = `${entry.name}\nclick = A · shift-click = B · right-click to delete`;
    tile.dataset.libId = entry.id;
    const img = document.createElement('img');
    img.src = url;
    tile.appendChild(img);
    libGridEl.appendChild(tile);
  }
}
libGridEl.addEventListener('click', e => {
  const tile = e.target.closest('.library-thumb');
  if (!tile) return;
  const id = parseInt(tile.dataset.libId, 10);
  const entry = _libCache.find(x => x.id === id);
  if (!entry) return;
  const slot = e.shiftKey ? 'B' : 'A';
  // Don't recurse through loadFile's addToLibrary — load directly + persist as last-A/B.
  state['_libId' + slot] = id;
  idbPut(slot === 'A' ? 'imageA' : 'imageB', entry.blob);
  loadFromUrl(URL.createObjectURL(entry.blob), slot);
  renderLibrary();
});
libGridEl.addEventListener('contextmenu', async e => {
  const tile = e.target.closest('.library-thumb');
  if (!tile) return;
  e.preventDefault();
  const id = parseInt(tile.dataset.libId, 10);
  if (!confirm('Delete this image from the library?')) return;
  await libDelete(id);
  if (state._libIdA === id) state._libIdA = null;
  if (state._libIdB === id) state._libIdB = null;
  renderLibrary();
});
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
  if (typeof updateTransportLabels !== 'undefined') updateTransportLabels();
}
// On startup: try to restore persisted A/B from IndexedDB; fall back to the
// bundled defaults if a slot has nothing stored. Then render the library grid.
async function seedLibraryWithDefaultsIfEmpty() {
  const list = await libList();
  if (list.length > 0) return;
  for (const path of ['./defaults/lofoten_A.jpg', './defaults/lofoten_B.jpg']) {
    try {
      const resp = await fetch(path);
      if (!resp.ok) continue;
      const blob = await resp.blob();
      await addToLibrary(new File([blob], path.split('/').pop(), { type: blob.type }));
    } catch {}
  }
}
(async () => {
  const [blobA, blobB] = await Promise.all([idbGet('imageA'), idbGet('imageB')]);
  loadFromUrl(blobA ? URL.createObjectURL(blobA) : './defaults/lofoten_A.jpg', 'A');
  loadFromUrl(blobB ? URL.createObjectURL(blobB) : './defaults/lofoten_B.jpg', 'B');
  await seedLibraryWithDefaultsIfEmpty();
  renderLibrary();
})();

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
document.getElementById('clear').addEventListener('click', async () => {
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
  await idbClearAll();
});

document.getElementById('reset').addEventListener('click', async () => {
  await idbClearAll();
  loadFromUrl('./defaults/lofoten_A.jpg', 'A');
  loadFromUrl('./defaults/lofoten_B.jpg', 'B');
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
pane.registerPlugin(EssentialsPlugin);

// Wrap addBinding / addFolder so every tracked binding records its options,
// letting randomizeMode pick values in each control's actual UI range without
// duplicating range tables. Applies recursively to all sub-folders.
const _bindOpts = new WeakMap();
const _bindKey  = new WeakMap();
function _patchAdders(folder) {
  const origBind = folder.addBinding.bind(folder);
  folder.addBinding = (target, key, opts) => {
    const api = origBind(target, key, opts);
    // Only track bindings against the main state object — rating bindings use
    // a local object so they never appear in the randomize walk.
    if (target === state) {
      _bindOpts.set(api, opts || {});
      _bindKey.set(api, key);
    }
    return api;
  };
  const origFolder = folder.addFolder.bind(folder);
  folder.addFolder = (opts) => _patchAdders(origFolder(opts));
  if (folder.addTab) {
    const origTab = folder.addTab.bind(folder);
    folder.addTab = (opts) => {
      const tab = origTab(opts);
      for (const page of tab.pages) _patchAdders(page);
      return tab;
    };
  }
  return folder;
}
_patchAdders(pane);

// ---- per-mode starred flag (simple boolean) ----
const STARRED_LS_KEY = 'transition-tool-v3:starred';
const starred = (() => { try { return JSON.parse(localStorage.getItem(STARRED_LS_KEY)) || {}; } catch { return {}; } })();
function saveStarred() { try { localStorage.setItem(STARRED_LS_KEY, JSON.stringify(starred)); } catch {} }
function setStarred(modeId, on) { if (on) starred[modeId] = true; else delete starred[modeId]; saveStarred(); }
function isStarred(modeId)      { return !!starred[modeId]; }

// Randomize the mode-specific controls of a folder using each binding's actual
// UI range (recorded by _patchAdders). Rating + button-grid + button children
// aren't tracked, so they're skipped naturally.
function randomizeMode(modeId, folder) {
  for (const child of folder.children) {
    if (!_bindKey.has(child)) continue;
    const key  = _bindKey.get(child);
    const opts = _bindOpts.get(child) || {};
    if (opts.options) {
      const values = Object.values(opts.options);
      state[key] = values[Math.floor(Math.random() * values.length)];
    } else if (typeof opts.min === 'number' && typeof opts.max === 'number') {
      const step = opts.step || 0.001;
      const v = opts.min + Math.random() * (opts.max - opts.min);
      state[key] = Math.round(v / step) * step;
    } else if (typeof state[key] === 'boolean') {
      state[key] = Math.random() < 0.5;
    }
  }
  state.seed = Math.floor(Math.random() * 999);
  if (modeId >= 10 && modeId <= 14) advec.needsReset = true;
  pane.refresh();
}

const fPlay = pane.addFolder({ title: 'Playback', expanded: true });
const bT = fPlay.addBinding(state, 't', { min: 0, max: 1, step: 0.001, label: 'progress' });
fPlay.addBinding(state, 'duration', { min: 0.5, max: 30, step: 0.1 });

function togglePlay() {
  if (state.playing) { state.playing = false; }
  else {
    state.playing = true;
    const consumed = state.reverse ? (1 - state.t) : state.t;
    state.startTime = performance.now() - consumed * state.duration * 1000;
  }
  updateTransportLabels();
}
function restartPlayback() {
  state.t = 0; state.reverse = false; state.playing = true;
  state.startTime = performance.now();
  updateTransportLabels();
}
function toggleLoop() {
  state.loop = !state.loop;
  updateTransportLabels();
}
const transportGrid = fPlay.addBlade({
  view: 'buttongrid',
  size: [3, 1],
  cells: (x) => ({ title: [state.playing ? 'Pause' : 'Play', 'Restart', 'Loop: ' + (state.loop ? 'on' : 'off')][x] }),
  label: '',
});
transportGrid.on('click', e => {
  const idx = e.index[0];
  if (idx === 0) togglePlay();
  else if (idx === 1) restartPlayback();
  else if (idx === 2) toggleLoop();
});
function updateTransportLabels() {
  const btns = transportGrid.element.querySelectorAll('button');
  if (btns[0]) btns[0].textContent = state.playing ? 'Pause' : 'Play';
  if (btns[2]) btns[2].textContent = 'Loop: ' + (state.loop ? 'on' : 'off');
}

// ---- top-level tabs (Playback stays above; everything else goes in a tab) ----
const tabs = pane.addTab({
  pages: [
    { title: 'Mode' },
    { title: 'Frame' },
    { title: 'Output' },
    { title: 'Saved' },
  ],
});
const tabMode   = tabs.pages[0];
const tabFrame  = tabs.pages[1];
const tabOutput = tabs.pages[2];
const tabSaved  = tabs.pages[3];

// Per-mode default values — used by the "Reset defaults" button in each
// mode folder to restore that mode's params without touching anything else.
const MODE_DEFAULTS = {
  1:  { rimWidth: 0.12, rimDark: 0.6 },
  2:  { paperAngle: 0, paperAniso: 4, paperGranulation: 0.5 },
  3:  { bloomCount: 8, bloomRim: 0.6, bloomRate: 0.55 },
  4:  { diffStrength: 0.55, diffRadius: 0.45 },
  5:  { sedBands: 6, sedSoftness: 0.35, sedDirection: 0, sedSource: 0 },
  6:  { saltDensity: 0.0, saltContrast: 0.55, saltSource: 1, saltBias: 0.6, saltImage: 2 },
  7:  { irisFocusX: 0.5, irisFocusY: 0.5, irisJitter: 0.35, irisUniform: true },
  8:  { bleedFinger: 0.5, bleedAmount: 0.45, bleedHalo: 0.5 },
  9:  { runGravity: 0.5, runDrip: 0.35 },
  10: { advecVisc: 0.55, advecRate: 0.18, advecSteps: 3 },
  11: { advecGravity: 0.6, advecGravBias: 0.5, advecGravAngle: 0, advecGravStreak: 0.4, advecGravLateral: 0.3 },
  12: { advecCurlStr: 0.5, advecCurlScale: 2.5 },
  13: { advecBrushFollow: 0.7 },
  14: { advecSeedCount: 5, advecSeedRadius: 0.45 },
  15: {
    weEdgeScale: 6.0, weEdgeWobble: 0.55,
    weDryRing: 0.45, weBleed: 0.5,
    weTendrilCount: 6, weTendrilReach: 0.4, weTendrilWidth: 0.5, weTendrilStrength: 0.55,
    weDetailBias: 0.35,
    weReverse: false, weBDetailBias: 0.0, weBLumaBias: 0.0,
  },
  16: { strokeScale: 6.0, strokeAniso: 4.0 },
  17: { glazeBands: 3.0, glazeSoftness: 0.55, glazeDirection: 0, glazeWarm: 0.35 },
  18: { edgeFirstInk: 0.55, edgeFirstFade: 0.35, edgeFirstScale: 3.0 },
  19: { flowAmount: 0.55 },
  20: { dabsCount: 28, dabsReach: 0.32, dabsWobble: 0.6 },
  21: { densityGravity: 0.45, densitySmear: 0.45 },
  22: { moldSeedCount: 5, moldTendrilsPerSeed: 4, moldReach: 0.5, moldWidth: 0.35, moldWobble: 0.6 },
  23: { formStrokeCount: 32, formStrokeSize: 0.05, formStrokeWobble: 0.5 },
  24: { bloomLightBias: 0.85, bloomWobble: 0.5, bloomPaperShow: 0.6 },
  25: { stageBands: 4, stageOverlap: 0.5 },
  26: { migrationStrength: 0.6, migrationDir: 0, migrationTurb: 0.5 },
};
function resetModeDefaults(modeId) {
  const d = MODE_DEFAULTS[modeId];
  if (!d) return;
  for (const [k, v] of Object.entries(d)) state[k] = v;
  if (modeId >= 10 && modeId <= 14) advec.needsReset = true;
  pane.refresh();
}
function addModeFooter(folder, modeId) {
  // Single-cell star toggle — starred or not.
  const starGrid = folder.addBlade({
    view: 'buttongrid',
    size: [1, 1],
    cells: () => ({ title: isStarred(modeId) ? '★ starred' : '☆ star this mode' }),
    label: '',
  });
  const paintStar = () => {
    const btn = starGrid.element.querySelector('button');
    if (!btn) return;
    const on = isStarred(modeId);
    btn.textContent = on ? '★ starred' : '☆ star this mode';
    btn.classList.toggle('rating-active', on);
  };
  starGrid.on('click', () => { setStarred(modeId, !isStarred(modeId)); paintStar(); });
  queueMicrotask(paintStar);
  // Randomize + Reset side-by-side.
  const actionGrid = folder.addBlade({
    view: 'buttongrid',
    size: [2, 1],
    cells: (x) => ({ title: ['🎲 Randomize', 'Reset'][x] }),
    label: '',
  });
  actionGrid.on('click', e => {
    if (e.index[0] === 0) randomizeMode(modeId, folder);
    else resetModeDefaults(modeId);
  });
}

// ----- Watercolor mode + per-mode controls -----
const fWater = tabMode.addFolder({ title: 'Watercolor', expanded: true });
const MODE_OPTIONS = {
  '— off (smooth)':                       0,
  'Painterly — pigment rim':              1,
  'Painterly — paper grain':              2,
  'Painterly — backrun blooms':           3,
  'Painterly — wet diffusion':            4,
  'Painterly — tonal sediment':           5,
  'Painterly — salt':                     6,
  'Painterly — iris':                     7,
  'Painterly — wet bleed':                8,
  'Painterly — pigment run':              9,
  'Advection — wet':                      10,
  'Advection — gravity':                  11,
  'Advection — curl-noise eddies':        12,
  'Advection — brush-channel':            13,
  'Advection — seed-point injection':     14,
  'Wet edge — rect ingress':              15,
  'Image-driven — stroke-follow':         16,
  'Image-driven — tonal wash':            17,
  'Image-driven — edge underdrawing':     18,
  'Image-driven — painterly flow':        19,
  'Image-driven — color-pool dabs':       20,
  'Image-driven — wet-density gravity':   21,
  'Decay — mold tendrils':                22,
  'Strong watercolor — formation':        23,
  'Strong watercolor — cauliflower bloom':24,
  'Strong watercolor — wet-stage layer':  25,
  'Strong watercolor — pigment migration':26,
};
const MODE_NAMES_FULL = Object.fromEntries(Object.entries(MODE_OPTIONS).map(([n, id]) => [id, n]));
fWater.addBinding(state, 'mode', {
  label: 'mode',
  options: MODE_OPTIONS,
}).on('change', () => { updateModeFolders(); advec.needsReset = true; });

const fRim    = fWater.addFolder({ title: 'Pigment rim',    expanded: true });
fRim.addBinding(state, 'rimWidth', { min: 0, max: 0.4, step: 0.005, label: 'rim width' });
fRim.addBinding(state, 'rimDark',  { min: 0, max: 1, step: 0.01, label: 'rim dark' });
addModeFooter(fRim, 1);

const fPaper  = fWater.addFolder({ title: 'Paper grain',    expanded: true });
fPaper.addBinding(state, 'paperAngle',       { min: 0, max: 1, step: 0.005, label: 'fiber angle' });
fPaper.addBinding(state, 'paperAniso',       { min: 1, max: 10, step: 0.1, label: 'anisotropy' });
fPaper.addBinding(state, 'paperGranulation', { min: 0, max: 1, step: 0.01, label: 'granulation' });
addModeFooter(fPaper, 2);

const fBlooms = fWater.addFolder({ title: 'Backrun blooms', expanded: true });
fBlooms.addBinding(state, 'bloomCount', { min: 1, max: 24, step: 1, label: 'count' });
fBlooms.addBinding(state, 'bloomRate',  { min: 0.1, max: 2, step: 0.01, label: 'growth rate' });
fBlooms.addBinding(state, 'bloomRim',   { min: 0, max: 1, step: 0.01, label: 'rim dark' });
addModeFooter(fBlooms, 3);

const fDiff   = fWater.addFolder({ title: 'Wet diffusion',  expanded: true });
fDiff.addBinding(state, 'diffStrength', { min: 0, max: 1, step: 0.01, label: 'strength' });
fDiff.addBinding(state, 'diffRadius',   { min: 0, max: 1, step: 0.01, label: 'radius' });
addModeFooter(fDiff, 4);

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
addModeFooter(fSed, 5);

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
addModeFooter(fSalt, 6);

const fIris   = fWater.addFolder({ title: 'Iris',           expanded: true });
fIris.addBinding(state, 'irisUniform', { label: 'uniform circle' });
fIris.addBinding(state, 'irisFocusX', { min: 0, max: 1, step: 0.005, label: 'focus x' });
fIris.addBinding(state, 'irisFocusY', { min: 0, max: 1, step: 0.005, label: 'focus y' });
fIris.addBinding(state, 'irisJitter', { min: 0, max: 1, step: 0.01, label: 'jitter' });
addModeFooter(fIris, 7);

const fBleed  = fWater.addFolder({ title: 'Wet bleed',      expanded: true });
fBleed.addBinding(state, 'bleedFinger', { min: 0, max: 1, step: 0.01, label: 'finger' });
fBleed.addBinding(state, 'bleedAmount', { min: 0, max: 1, step: 0.01, label: 'amount' });
fBleed.addBinding(state, 'bleedHalo',   { min: 0, max: 1, step: 0.01, label: 'wet halo' });
addModeFooter(fBleed, 8);

const fRun    = fWater.addFolder({ title: 'Pigment run',    expanded: true });
fRun.addBinding(state, 'runGravity', { min: 0, max: 1, step: 0.01, label: 'gravity' });
fRun.addBinding(state, 'runDrip',    { min: 0, max: 1, step: 0.01, label: 'drip' });
addModeFooter(fRun, 9);

const fAdvec  = fWater.addFolder({ title: 'Wet advection',  expanded: true });
fAdvec.addBinding(state, 'advecVisc',  { min: 0, max: 1, step: 0.01, label: 'viscosity' });
fAdvec.addBinding(state, 'advecRate',  { min: 0, max: 1, step: 0.01, label: 'mixing rate' });
fAdvec.addBinding(state, 'advecSteps', { min: 1, max: 8, step: 1, label: 'steps / frame' });
addModeFooter(fAdvec, 10);
fAdvec.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

const fAdvecG = fWater.addFolder({ title: 'Gravity advection', expanded: true });
fAdvecG.addBinding(state, 'advecGravAngle',   { min: 0, max: 1, step: 0.005, label: 'flow angle' });
fAdvecG.addBinding(state, 'advecGravity',     { min: 0, max: 1, step: 0.01, label: 'gravity' });
fAdvecG.addBinding(state, 'advecGravStreak',  { min: 0, max: 1, step: 0.01, label: 'streak' });
fAdvecG.addBinding(state, 'advecGravLateral', { min: 0, max: 1, step: 0.01, label: 'lateral spread' });
fAdvecG.addBinding(state, 'advecGravBias',    { min: 0, max: 1, step: 0.01, label: 'shadow ↔ flow' });
addModeFooter(fAdvecG, 11);
fAdvecG.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

const fAdvecC = fWater.addFolder({ title: 'Curl-noise eddies', expanded: true });
fAdvecC.addBinding(state, 'advecCurlStr',   { min: 0, max: 1, step: 0.01, label: 'eddy strength' });
fAdvecC.addBinding(state, 'advecCurlScale', { min: 0.5, max: 8, step: 0.1, label: 'eddy scale' });
addModeFooter(fAdvecC, 12);
fAdvecC.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

const fAdvecB = fWater.addFolder({ title: 'Brush-channel advection', expanded: true });
fAdvecB.addBinding(state, 'advecBrushFollow', { min: 0, max: 1, step: 0.01, label: 'follow strokes' });
addModeFooter(fAdvecB, 13);
fAdvecB.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

const fAdvecS = fWater.addFolder({ title: 'Seed-point injection', expanded: true });
fAdvecS.addBinding(state, 'advecSeedCount',  { min: 1, max: 16, step: 1, label: 'seed count' });
fAdvecS.addBinding(state, 'advecSeedRadius', { min: 0.1, max: 1, step: 0.01, label: 'reach' });
addModeFooter(fAdvecS, 14);
fAdvecS.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

const fWetEdge = fWater.addFolder({ title: 'Wet edge (rect)', expanded: true });
fWetEdge.addBinding(state, 'weEdgeScale',       { min: 1,    max: 16, step: 0.1,  label: 'edge scale' });
fWetEdge.addBinding(state, 'weEdgeWobble',      { min: 0,    max: 1,  step: 0.01, label: 'edge wobble' });
fWetEdge.addBinding(state, 'weTendrilCount',    { min: 0,    max: 32, step: 1,    label: 'tendril count' });
fWetEdge.addBinding(state, 'weTendrilReach',    { min: 0.02, max: 1,  step: 0.01, label: 'tendril reach' });
fWetEdge.addBinding(state, 'weTendrilWidth',    { min: 0.02, max: 1,  step: 0.01, label: 'tendril width' });
fWetEdge.addBinding(state, 'weTendrilStrength', { min: 0,    max: 1,  step: 0.01, label: 'tendril strength' });
fWetEdge.addBinding(state, 'weDetailBias',      { min: 0,    max: 1,  step: 0.01, label: 'detail bias (A)' });
fWetEdge.addBinding(state, 'weBDetailBias',     { min: 0,    max: 1,  step: 0.01, label: 'detail bias (B)' });
fWetEdge.addBinding(state, 'weBLumaBias',       { min: -1,   max: 1,  step: 0.01, label: 'B luma bias' });
fWetEdge.addBinding(state, 'weReverse',         { label: 'reverse (center→out)' });
fWetEdge.addBinding(state, 'weDryRing',         { min: 0,    max: 1,  step: 0.01, label: 'dry-ring dark' });
fWetEdge.addBinding(state, 'weBleed',           { min: 0,    max: 1,  step: 0.01, label: 'anticipatory bleed' });
addModeFooter(fWetEdge, 15);

const fStroke = fWater.addFolder({ title: 'Stroke-follow', expanded: true });
fStroke.addBinding(state, 'strokeScale', { min: 0.5, max: 20, step: 0.1, label: 'stroke scale' });
fStroke.addBinding(state, 'strokeAniso', { min: 1,   max: 12, step: 0.1, label: 'anisotropy' });
addModeFooter(fStroke, 16);

const fGlaze = fWater.addFolder({ title: 'Tonal wash', expanded: true });
fGlaze.addBinding(state, 'glazeBands',    { min: 2, max: 8, step: 1,    label: 'washes' });
fGlaze.addBinding(state, 'glazeSoftness', { min: 0, max: 1, step: 0.01, label: 'softness' });
fGlaze.addBinding(state, 'glazeDirection', {
  label: 'order',
  options: { 'darks first': 0, 'lights first': 1 },
});
fGlaze.addBinding(state, 'glazeWarm', { min: 0, max: 1, step: 0.01, label: 'warm dry-shift' });
addModeFooter(fGlaze, 17);

const fEdgeFirst = fWater.addFolder({ title: 'Edge underdrawing', expanded: true });
fEdgeFirst.addBinding(state, 'edgeFirstInk',   { min: 0,    max: 1,  step: 0.01, label: 'ink' });
fEdgeFirst.addBinding(state, 'edgeFirstFade',  { min: 0.05, max: 0.9, step: 0.01, label: 'sketch fades at t=' });
fEdgeFirst.addBinding(state, 'edgeFirstScale', { min: 1,    max: 10, step: 0.1,  label: 'mask scale' });
addModeFooter(fEdgeFirst, 18);

const fFlow = fWater.addFolder({ title: 'Painterly flow', expanded: true });
fFlow.addBinding(state, 'flowAmount', { min: 0, max: 1, step: 0.01, label: 'flow amount' });
addModeFooter(fFlow, 19);

const fDabs = fWater.addFolder({ title: 'Color-pool dabs', expanded: true });
fDabs.addBinding(state, 'dabsCount',  { min: 1,    max: 128, step: 1,    label: 'dab count' });
fDabs.addBinding(state, 'dabsReach',  { min: 0.05, max: 1,   step: 0.01, label: 'reach' });
fDabs.addBinding(state, 'dabsWobble', { min: 0,    max: 1,   step: 0.01, label: 'edge wobble' });
addModeFooter(fDabs, 20);

const fDensity = fWater.addFolder({ title: 'Wet-density gravity', expanded: true });
fDensity.addBinding(state, 'densityGravity', { min: 0, max: 1, step: 0.01, label: 'gravity bias' });
fDensity.addBinding(state, 'densitySmear',   { min: 0, max: 1, step: 0.01, label: 'wet smear' });
addModeFooter(fDensity, 21);

const fMold = fWater.addFolder({ title: 'Mold tendrils', expanded: true });
fMold.addBinding(state, 'moldSeedCount',        { min: 1,    max: 16, step: 1,    label: 'seed count' });
fMold.addBinding(state, 'moldTendrilsPerSeed',  { min: 1,    max: 8,  step: 1,    label: 'tendrils / seed' });
fMold.addBinding(state, 'moldReach',            { min: 0.05, max: 1,  step: 0.01, label: 'reach' });
fMold.addBinding(state, 'moldWidth',            { min: 0.05, max: 1,  step: 0.01, label: 'tendril width' });
fMold.addBinding(state, 'moldWobble',           { min: 0,    max: 1,  step: 0.01, label: 'wobble' });
addModeFooter(fMold, 22);

const fForm = fWater.addFolder({ title: 'Watercolor formation', expanded: true });
fForm.addBinding(state, 'formStrokeCount',  { min: 1,    max: 64,  step: 1,    label: 'stroke count' });
fForm.addBinding(state, 'formStrokeSize',   { min: 0.01, max: 0.2, step: 0.005, label: 'stroke size' });
fForm.addBinding(state, 'formStrokeWobble', { min: 0,    max: 1,   step: 0.01, label: 'edge wobble' });
addModeFooter(fForm, 23);

const fBloom = fWater.addFolder({ title: 'Cauliflower bloom storm', expanded: true });
fBloom.addBinding(state, 'bloomLightBias',  { min: 0, max: 1, step: 0.01, label: 'light bias (B)' });
fBloom.addBinding(state, 'bloomWobble',     { min: 0, max: 1, step: 0.01, label: 'bloom wobble' });
fBloom.addBinding(state, 'bloomPaperShow',  { min: 0, max: 1, step: 0.01, label: 'paper-show pop' });
addModeFooter(fBloom, 24);

const fStage = fWater.addFolder({ title: 'Wet-stage layering', expanded: true });
fStage.addBinding(state, 'stageBands',   { min: 2, max: 8, step: 1,    label: 'stages' });
fStage.addBinding(state, 'stageOverlap', { min: 0, max: 1, step: 0.01, label: 'stage overlap' });
addModeFooter(fStage, 25);

const fMig = fWater.addFolder({ title: 'Pigment migration', expanded: true });
fMig.addBinding(state, 'migrationStrength', { min: 0, max: 1, step: 0.01, label: 'strength' });
fMig.addBinding(state, 'migrationDir', {
  label: 'direction',
  options: { 'along gradient': 0, 'perpendicular': 1 },
});
fMig.addBinding(state, 'migrationTurb', { min: 0, max: 1, step: 0.01, label: 'turbulence' });
addModeFooter(fMig, 26);

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
  fAdvec.hidden   = state.mode !== 10;
  fAdvecG.hidden  = state.mode !== 11;
  fAdvecC.hidden  = state.mode !== 12;
  fAdvecB.hidden  = state.mode !== 13;
  fAdvecS.hidden  = state.mode !== 14;
  fWetEdge.hidden = state.mode !== 15;
  fStroke.hidden    = state.mode !== 16;
  fGlaze.hidden     = state.mode !== 17;
  fEdgeFirst.hidden = state.mode !== 18;
  fFlow.hidden      = state.mode !== 19;
  fDabs.hidden      = state.mode !== 20;
  fDensity.hidden   = state.mode !== 21;
  fMold.hidden      = state.mode !== 22;
  fForm.hidden      = state.mode !== 23;
  fBloom.hidden     = state.mode !== 24;
  fStage.hidden     = state.mode !== 25;
  fMig.hidden       = state.mode !== 26;
}
updateModeFolders();

const fDis = tabMode.addFolder({ title: 'Dissolve', expanded: true });
fDis.addBinding(state, 'organic',   { min: 0, max: 1, step: 0.01 });
fDis.addBinding(state, 'edges',     { min: -1, max: 1, step: 0.01 });
fDis.addBinding(state, 'spread',    { min: 0, max: 1, step: 0.01 });
fDis.addBinding(state, 'maskScale', { min: 0.3, max: 4, step: 0.05, label: 'mask scale' });
fDis.addBinding(state, 'curve', { options: { 'linear': 0, 'ease-in-out': 1, 'ease-in': 2, 'ease-out': 3 } });
fDis.addBinding(state, 'seed', { min: 0, max: 999, step: 1 });
fDis.addBinding(state, 'maskShift', { min: -0.5, max: 0.5, step: 0.005, label: 'mask shift' });

const fImg = tabFrame.addFolder({ title: 'Framing', expanded: true });
fImg.addBinding(state, 'zoomA', { min: 0.5, max: 4, step: 0.01, label: 'A zoom' });
fImg.addBinding(state, 'panAx', { min: -1, max: 1, step: 0.005, label: 'A pan x' });
fImg.addBinding(state, 'panAy', { min: -1, max: 1, step: 0.005, label: 'A pan y' });
fImg.addBinding(state, 'zoomB', { min: 0.5, max: 4, step: 0.01, label: 'B zoom' });
fImg.addBinding(state, 'panBx', { min: -1, max: 1, step: 0.005, label: 'B pan x' });
fImg.addBinding(state, 'panBy', { min: -1, max: 1, step: 0.005, label: 'B pan y' });

// ----- Export / Record -----
state.exportFps = 24;
state.exportSizeMode = '1920';
state.exportPadBottom = 0;  // 0 = no padding; 1 = add full-height black below; 1.416 ≈ Elverket floor ratio

// Prefer HEVC (H.265) over H.264 — HEVC headroom is ~7680 vs ~3840 for AVC, so
// wide panoramas survive without aggressive downscaling. Falls back gracefully.
const RECORDER_MIMES = [
  'video/mp4;codecs=hvc1.1.6.L120.B0',  // HEVC Main L4 — Chrome 126+ on macOS
  'video/mp4;codecs=hev1.1.6.L120.B0',
  'video/mp4;codecs=hvc1',
  'video/mp4;codecs=hev1',
  'video/mp4;codecs=avc1.42E01E',        // H.264 fallback
  'video/mp4;codecs=avc1',
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];
function pickRecorderMime() {
  for (const m of RECORDER_MIMES) if (MediaRecorder.isTypeSupported(m)) return m;
  return 'video/webm';
}
const mimeToExt = m => m.startsWith('video/mp4') ? 'mp4' : 'webm';
// Encoder-specific maximum dimension. H.265 hardware encoders handle 8K, H.264
// usually 4K, VP9 ~8K. We probe with the actual picked mime so the cap matches.
function encoderMaxDim(mime) {
  if (/hev|hvc/.test(mime)) return 7680;
  if (/vp9/.test(mime))     return 7680;
  if (/avc|h264/.test(mime)) return 3840;
  return 3840;
}

const MODE_NAMES_V2 = {
  0: 'off', 1: 'rim', 2: 'paper', 3: 'blooms', 4: 'diffusion',
  5: 'sediment', 6: 'salt', 7: 'iris', 8: 'wet-bleed', 9: 'pigment-run',
  10: 'advec', 15: 'wet-edge',
  16: 'stroke', 17: 'glaze', 18: 'edge-first', 19: 'flow', 20: 'dabs', 21: 'density',
  22: 'mold',
  23: 'wc-form', 24: 'bloom-storm', 25: 'wet-stage', 26: 'pig-migration',
};
const SED_SOURCE_NAMES = ['luma','sat','hue','detail','temp'];
const SALT_SOURCE_NAMES = ['random','light','dark','col','edge'];
const fx = (v, n = 2) => (Math.round(v * Math.pow(10, n)) / Math.pow(10, n)).toString();

function makeFilenameV2() {
  const m = state.mode;
  const parts = [MODE_NAMES_V2[m] || `mode${m}`];
  if      (m === 1)  parts.push(`rimW=${fx(state.rimWidth)}`, `dark=${fx(state.rimDark)}`);
  else if (m === 2)  parts.push(`ang=${fx(state.paperAngle)}`, `aniso=${fx(state.paperAniso,1)}`, `gran=${fx(state.paperGranulation)}`);
  else if (m === 3)  parts.push(`n=${state.bloomCount}`, `rate=${fx(state.bloomRate)}`, `rim=${fx(state.bloomRim)}`);
  else if (m === 4)  parts.push(`str=${fx(state.diffStrength)}`, `r=${fx(state.diffRadius)}`);
  else if (m === 5)  parts.push(`by=${SED_SOURCE_NAMES[state.sedSource] || 'luma'}`, `bands=${state.sedBands}`, `soft=${fx(state.sedSoftness)}`);
  else if (m === 6)  parts.push(`from=${SALT_SOURCE_NAMES[state.saltSource] || 'random'}`, `grain=${fx(state.saltDensity)}`, `bias=${fx(state.saltBias)}`);
  else if (m === 7)  parts.push(`focus=${fx(state.irisFocusX)}-${fx(state.irisFocusY)}`, `jit=${fx(state.irisJitter)}`, state.irisUniform ? 'uniform' : 'stretched');
  else if (m === 8)  parts.push(`fing=${fx(state.bleedFinger)}`, `amt=${fx(state.bleedAmount)}`, `halo=${fx(state.bleedHalo)}`);
  else if (m === 9)  parts.push(`grav=${fx(state.runGravity)}`, `drip=${fx(state.runDrip)}`);
  else if (m === 10) parts.push(`visc=${fx(state.advecVisc)}`, `rate=${fx(state.advecRate)}`);
  else if (m === 15) {
    parts.push(`wob=${fx(state.weEdgeWobble)}`, `tend=${state.weTendrilCount}`, `det=${fx(state.weDetailBias)}`, `ring=${fx(state.weDryRing)}`);
    if (state.weReverse) parts.push('rev');
    if (state.weBDetailBias > 0.001) parts.push(`detB=${fx(state.weBDetailBias)}`);
    if (Math.abs(state.weBLumaBias) > 0.001) parts.push(`lumB=${fx(state.weBLumaBias)}`);
  }
  else if (m === 16) parts.push(`sc=${fx(state.strokeScale,1)}`, `aniso=${fx(state.strokeAniso,1)}`);
  else if (m === 17) parts.push(`bands=${state.glazeBands}`, `soft=${fx(state.glazeSoftness)}`, state.glazeDirection ? 'lights-first' : 'darks-first', `warm=${fx(state.glazeWarm)}`);
  else if (m === 18) parts.push(`ink=${fx(state.edgeFirstInk)}`, `fade=${fx(state.edgeFirstFade)}`);
  else if (m === 19) parts.push(`flow=${fx(state.flowAmount)}`);
  else if (m === 20) parts.push(`n=${state.dabsCount}`, `reach=${fx(state.dabsReach)}`, `wob=${fx(state.dabsWobble)}`);
  else if (m === 21) parts.push(`grav=${fx(state.densityGravity)}`, `smear=${fx(state.densitySmear)}`);
  else if (m === 22) parts.push(`seeds=${state.moldSeedCount}`, `per=${state.moldTendrilsPerSeed}`, `reach=${fx(state.moldReach)}`, `wob=${fx(state.moldWobble)}`);
  else if (m === 23) parts.push(`n=${state.formStrokeCount}`, `sz=${fx(state.formStrokeSize)}`, `wob=${fx(state.formStrokeWobble)}`);
  else if (m === 24) parts.push(`bias=${fx(state.bloomLightBias)}`, `wob=${fx(state.bloomWobble)}`, `paper=${fx(state.bloomPaperShow)}`);
  else if (m === 25) parts.push(`bands=${state.stageBands}`, `over=${fx(state.stageOverlap)}`);
  else if (m === 26) parts.push(`str=${fx(state.migrationStrength)}`, state.migrationDir ? 'perp' : 'along', `turb=${fx(state.migrationTurb)}`);
  if (state.paperGrain > 0.001) parts.push(`paper=${fx(state.paperGrain)}`);
  // duration / fps / dimensions / pad are appended by the recorder using
  // the actual output values (after any encoder downscale).
  return `transition__${parts.join('__')}`;
}

let recording = false;
const fExp = tabOutput.addFolder({ title: 'Export', expanded: true });
fExp.addBinding(state, 'exportFps', {
  label: 'fps', options: { '24 fps': 24, '25 fps': 25, '30 fps': 30, '50 fps': 50, '60 fps': 60 },
});
fExp.addBinding(state, 'exportSizeMode', {
  label: 'size',
  options: {
    'source (full)': 'src', '5120 wide': '5120', '3840 wide': '3840',
    '2560 wide': '2560', '1920 wide': '1920', '1280 wide': '1280', '960 wide': '960',
  },
});
// Preset dropdown that writes into state.exportPadBottom on change. Slider
// stays available for fine-tuning.
const padPresets = { _v: state.exportPadBottom };
const bPadPreset = fExp.addBinding(padPresets, '_v', {
  label: 'pad preset',
  options: {
    'none':                  0,
    'half (0.5)':            0.5,
    'full (1.0)':            1.0,
    'Elverket panorama':     1.416,
    'double (2.0)':          2.0,
  },
});
bPadPreset.on('change', e => {
  state.exportPadBottom = e.value;
  pane.refresh();
});
const bPad = fExp.addBinding(state, 'exportPadBottom', { min: 0, max: 3, step: 0.001, label: 'pad below (× h)' });

// ---- output folder (File System Access API, Chromium-only) ----
const HAS_FS_ACCESS = typeof window.showDirectoryPicker === 'function';
let outputDirHandle = null;
const outputFolderProxy = { name: 'browser default' };
(async () => {
  const saved = await idbGet('outputDir');
  if (saved) {
    outputDirHandle = saved;
    outputFolderProxy.name = saved.name;
    try { pane.refresh(); } catch {}
  }
})();
async function getOutputDirHandleWithPermission() {
  if (!outputDirHandle) return null;
  try {
    let perm = await outputDirHandle.queryPermission({ mode: 'readwrite' });
    if (perm !== 'granted') perm = await outputDirHandle.requestPermission({ mode: 'readwrite' });
    return perm === 'granted' ? outputDirHandle : null;
  } catch { return null; }
}
async function saveBlobToOutputFolder(blob, filename) {
  const dir = await getOutputDirHandleWithPermission();
  if (!dir) return false;
  try {
    const fileHandle = await dir.getFileHandle(filename, { create: true });
    const writable = await fileHandle.createWritable();
    await writable.write(blob);
    await writable.close();
    return true;
  } catch (e) {
    console.error('[output folder save]', e);
    return false;
  }
}
fExp.addBinding(outputFolderProxy, 'name', { readonly: true, label: 'output folder' });
fExp.addBlade({
  view: 'buttongrid',
  size: [2, 1],
  cells: (x) => ({ title: HAS_FS_ACCESS ? ['📁 Pick folder', 'Use default'][x] : ['(not supported in this browser)', ''][x] }),
  label: '',
}).on('click', async e => {
  if (!HAS_FS_ACCESS) return;
  if (e.index[0] === 0) {
    try {
      const handle = await window.showDirectoryPicker({ mode: 'readwrite' });
      outputDirHandle = handle;
      outputFolderProxy.name = handle.name;
      await idbPut('outputDir', handle);
      pane.refresh();
    } catch (err) { if (err.name !== 'AbortError') alert('Folder pick failed: ' + err.message); }
  } else {
    outputDirHandle = null;
    outputFolderProxy.name = 'browser default';
    await idbPut('outputDir', null);
    pane.refresh();
  }
});
// Slider → preset: keep the dropdown showing the matching preset (or 'none')
// when the slider lands on a value we have a preset for.
bPad.on('change', () => {
  const v = state.exportPadBottom;
  const presets = [0, 0.5, 1.0, 1.416, 2.0];
  const match = presets.find(p => Math.abs(p - v) < 0.001);
  padPresets._v = match !== undefined ? match : 0;
  bPadPreset.refresh();
});
const btnRecord = fExp.addButton({ title: 'Record video' });
btnRecord.on('click', () => startRecording());

// Try a series of VideoEncoder configs in descending order of profile/level so
// we pick the highest-headroom one this machine actually supports.
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
      const cfg = {
        codec: c.codec, width, height, framerate, bitrate,
        hardwareAcceleration: 'prefer-hardware',
      };
      const r = await VideoEncoder.isConfigSupported(cfg);
      if (r && r.supported) return { config: cfg, muxerCodec: c.muxer };
    } catch {}
  }
  return null;
}

// Maximum dimension a given codec / level can encode. Use these to scale the
// recording before configuring the encoder.
function codecMaxDim(codecString) {
  if (codecString.includes('L153')) return 8192;   // HEVC L5.1
  if (codecString.includes('L120')) return 4096;   // HEVC L4 or AVC L4
  if (codecString.includes('L033')) return 8192;   // (informational)
  if (codecString.includes('640033')) return 8192; // AVC High L5.1
  if (codecString.includes('640028')) return 4096; // AVC High L4
  return 3840;
}

async function startRecording(opts = {}) {
  if (recording) return;
  if (!state.imgA || !state.imgB) return;

  const fps = state.exportFps;
  const sizeMode = state.exportSizeMode;
  let recW = canvas.width, recH = canvas.height;
  if (sizeMode !== 'src') {
    const w = parseInt(sizeMode, 10);
    const h = Math.round(w * canvas.height / canvas.width);
    recW = w; recH = h;
  }
  const padPx0 = Math.round(recH * state.exportPadBottom);

  recording = true;
  const originalTitle = btnRecord.title;
  btnRecord.title = 'Preparing…';

  // Probe encoder support at the requested size; scale down if needed and re-probe.
  let scale = 1;
  let pick = null;
  for (let attempt = 0; attempt < 4; attempt++) {
    const tryW = (recW + (recW % 2));
    const tryH = ((recH + padPx0 * scale) + ((recH + padPx0 * scale) % 2));
    pick = await pickEncoderConfig(tryW, Math.round(tryH), fps, 12_000_000);
    if (pick) {
      const cap = codecMaxDim(pick.config.codec);
      if (Math.max(tryW, tryH) <= cap) break;
    }
    scale *= 0.75;
    recW = Math.round(recW * 0.75);
    recH = Math.round(recH * 0.75);
  }
  if (!pick) {
    btnRecord.title = 'FAILED — no usable video encoder';
    setTimeout(() => { btnRecord.title = originalTitle; }, 4000);
    recording = false;
    return;
  }

  const padPx = Math.round(recH * state.exportPadBottom);
  const totalH = (recH + padPx) + ((recH + padPx) % 2);
  const offW = recW + (recW % 2);
  const off = document.createElement('canvas');
  off.width = offW; off.height = totalH;
  const offCtx = off.getContext('2d');
  offCtx.fillStyle = '#000';
  offCtx.fillRect(0, 0, off.width, off.height);

  console.log(`[record] codec ${pick.config.codec}  ${offW}×${totalH}  ${fps}fps` + (scale < 1 ? `  (scaled ×${scale.toFixed(2)} from canvas ${canvas.width}×${canvas.height})` : ''));

  // Set up muxer + encoder.
  const muxer = new Muxer({
    target: new ArrayBufferTarget(),
    video: { codec: pick.muxerCodec, width: offW, height: totalH, frameRate: fps },
    fastStart: 'in-memory',
  });
  const encoder = new VideoEncoder({
    output: (chunk, meta) => muxer.addVideoChunk(chunk, meta),
    error: e => console.error('[encoder]', e),
  });
  encoder.configure({ ...pick.config, width: offW, height: totalH });

  const totalFrames = Math.max(2, Math.round(state.duration * fps));
  const frameDuration = 1_000_000 / fps; // microseconds
  const wasPlaying = state.playing;
  const prevT = state.t;
  state.playing = false;
  if (state.mode >= 10 && state.mode <= 14) advec.needsReset = true;

  btnRecord.title = scale < 1
    ? `Scaled to ${offW}×${totalH}. Recording…`
    : 'Recording…';

  for (let i = 0; i < totalFrames; i++) {
    state.t = i / (totalFrames - 1);
    renderFrame();
    // wait one rAF so the WebGPU swap-chain commits the just-submitted frame
    await new Promise(r => requestAnimationFrame(r));
    offCtx.drawImage(canvas, 0, 0, recW, recH);

    const vf = new VideoFrame(off, {
      timestamp: Math.round(i * frameDuration),
      duration: Math.round(frameDuration),
    });
    encoder.encode(vf);
    vf.close();

    btnRecord.title = `frame ${i + 1} / ${totalFrames}`;
    // Back-pressure: if the encoder queue is getting long, let it drain.
    if (encoder.encodeQueueSize > 16) {
      await new Promise(r => setTimeout(r, 0));
    }
  }
  await encoder.flush();
  muxer.finalize();

  recording = false;
  state.t = prevT;
  state.playing = wasPlaying;

  const blob = new Blob([muxer.target.buffer], { type: 'video/mp4' });
  if (blob.size < 1024) {
    btnRecord.title = 'FAILED — empty output';
    setTimeout(() => { btnRecord.title = originalTitle; }, 4000);
    return;
  }

  // Build filename with duration, fps, actual output dimensions, and pad
  // (appended here so the actual encoded size is reflected, not the
  // pre-scale request).
  let base = opts.filename || makeFilenameV2();
  if (!/\.mp4$/i.test(base)) {
    const tail = [
      `${Math.round(state.duration)}s`,
      `${fps}fps`,
      `${offW}x${totalH}`,
    ];
    if (state.exportPadBottom > 0) tail.push(`pad=${fx(state.exportPadBottom)}`);
    base = `${base}__${tail.join('__')}`;
  }
  const filename = /\.mp4$/i.test(base) ? base : `${base}.mp4`;

  // Try the persistent output folder first; fall back to a browser download.
  const savedToFolder = await saveBlobToOutputFolder(blob, filename);
  let where = '';
  if (savedToFolder) {
    where = ` → ${outputDirHandle?.name || 'folder'}`;
  } else {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = filename; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  btnRecord.title = `saved (${(blob.size / 1024 / 1024).toFixed(1)} MB)${where}`;
  setTimeout(() => { btnRecord.title = originalTitle; }, 2500);
}

const fStyle = tabFrame.addFolder({ title: 'Style', expanded: true });
fStyle.addBinding(state, 'fit', {
  options: { 'cover (crop)': 'cover', 'contain': 'contain', 'stretch': 'stretch' },
});
fStyle.addBinding(state, 'bg', { view: 'color' });
fStyle.addBinding(state, 'paperGrain', { min: 0, max: 1, step: 0.01, label: 'paper grain' });

const fSlotFill = fStyle.addFolder({ title: 'A / B fill', expanded: false });
fSlotFill.addBinding(state, 'slotAFillMode', {
  label: 'A fill',
  options: { 'image': 'image', 'solid color': 'solid', 'transparent (alpha)': 'transparent' },
}).on('change', () => resizeCanvas());
fSlotFill.addBinding(state, 'slotAColor', { view: 'color', label: 'A color' });
fSlotFill.addBinding(state, 'slotBFillMode', {
  label: 'B fill',
  options: { 'image': 'image', 'solid color': 'solid', 'transparent (alpha)': 'transparent' },
}).on('change', () => resizeCanvas());
fSlotFill.addBinding(state, 'slotBColor', { view: 'color', label: 'B color' });
fSlotFill.addBinding(state, 'keepAOutsideB', { label: 'keep A outside B' });

const fBounds = fStyle.addFolder({ title: 'Transition bounds', expanded: false });
fBounds.addBinding(state, 'boundsEnable',   { label: 'limit to box' });
fBounds.addBinding(state, 'boundsCx',       { min: 0, max: 1,    step: 0.005, label: 'center x' });
fBounds.addBinding(state, 'boundsCy',       { min: 0, max: 1,    step: 0.005, label: 'center y' });
fBounds.addBinding(state, 'boundsW',        { min: 0.02, max: 1, step: 0.005, label: 'width' });
fBounds.addBinding(state, 'boundsH',        { min: 0.02, max: 1, step: 0.005, label: 'height' });
fBounds.addBinding(state, 'boundsSoftness', { min: 0, max: 0.3,  step: 0.005, label: 'edge softness' });

// ----- Presets -----
const PRESET_KEYS = [
  'duration', 'mode', 'curve', 'seed',
  'rimWidth', 'rimDark',
  'paperAngle', 'paperAniso', 'paperGranulation',
  'bloomCount', 'bloomRim', 'bloomRate',
  'diffStrength', 'diffRadius',
  'sedBands', 'sedSoftness', 'sedDirection', 'sedSource',
  'saltDensity', 'saltContrast', 'saltSource', 'saltBias', 'saltImage',
  'irisFocusX', 'irisFocusY', 'irisJitter', 'irisUniform',
  'bleedFinger', 'bleedAmount', 'bleedHalo',
  'runGravity', 'runDrip',
  'advecVisc', 'advecRate', 'advecSteps',
  'advecGravity', 'advecGravBias', 'advecGravAngle', 'advecGravStreak', 'advecGravLateral',
  'advecCurlStr', 'advecCurlScale',
  'advecBrushFollow',
  'advecSeedCount', 'advecSeedRadius',
  'weEdgeScale', 'weEdgeWobble', 'weDryRing', 'weBleed',
  'weTendrilCount', 'weTendrilReach', 'weTendrilWidth', 'weTendrilStrength',
  'weDetailBias', 'weReverse', 'weBDetailBias', 'weBLumaBias',
  'strokeScale', 'strokeAniso',
  'glazeBands', 'glazeSoftness', 'glazeDirection', 'glazeWarm',
  'edgeFirstInk', 'edgeFirstFade', 'edgeFirstScale',
  'flowAmount',
  'dabsCount', 'dabsReach', 'dabsWobble',
  'densityGravity', 'densitySmear',
  'paperGrain',
  'moldSeedCount', 'moldTendrilsPerSeed', 'moldReach', 'moldWidth', 'moldWobble',
  'formStrokeCount', 'formStrokeSize', 'formStrokeWobble',
  'bloomLightBias', 'bloomWobble', 'bloomPaperShow',
  'stageBands', 'stageOverlap',
  'migrationStrength', 'migrationDir', 'migrationTurb',
  'boundsEnable', 'boundsCx', 'boundsCy', 'boundsW', 'boundsH', 'boundsSoftness',
  'organic', 'edges', 'spread', 'maskScale', 'maskShift',
  'zoomA', 'panAx', 'panAy', 'zoomB', 'panBx', 'panBy',
];

const FACTORY_PRESETS = {
  'Smooth dreamy': {
    duration: 5, mode: 0, curve: 1, seed: 42,
    organic: 0.65, edges: 0.25, spread: 0.55, maskScale: 0.9,
  },
  'Paper grain — cold press': {
    duration: 7, mode: 2, curve: 1, seed: 42,
    paperAngle: 0, paperAniso: 6, paperGranulation: 0.7,
    organic: 0.3, edges: 0.15, spread: 0.35, maskScale: 0.9,
  },
  'Backruns — dramatic': {
    duration: 12, mode: 3, curve: 2, seed: 12,
    bloomCount: 4, bloomRate: 0.45, bloomRim: 0.75,
    organic: 0.5, edges: 0.2, spread: 0.5, maskScale: 0.9,
  },
  'Sediment — hue': {
    duration: 8, mode: 5, curve: 0, seed: 42,
    sedSource: 2, sedBands: 5, sedSoftness: 0.35,
    organic: 0.5, spread: 0.5,
  },
  'Salt — light bias': {
    duration: 6, mode: 6, curve: 0, seed: 42,
    saltDensity: 0.55, saltContrast: 0.55, saltSource: 1, saltBias: 0.75, saltImage: 2,
  },
  'Wet advection (smooth)': {
    duration: 10, mode: 10, curve: 0, seed: 42,
    advecVisc: 0.55, advecRate: 0.18, advecSteps: 3,
    organic: 0.6, spread: 0.55, maskScale: 0.9,
  },
  'Gravity advection — down': {
    duration: 10, mode: 11, curve: 0, seed: 42,
    advecGravAngle: 0, advecGravity: 0.8, advecGravStreak: 0.5,
    advecGravLateral: 0.3, advecGravBias: 0.5,
    advecVisc: 0.55, advecRate: 0.18,
  },
  'Curl-noise eddies': {
    duration: 10, mode: 12, curve: 0, seed: 42,
    advecCurlStr: 0.8, advecCurlScale: 1.5,
    advecVisc: 0.55, advecRate: 0.18,
  },
};

const LS_KEY = 'transition-tool-v3:presets';
const loadUserPresets = () => { try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; } };
const saveUserPresetsToLS = (o) => localStorage.setItem(LS_KEY, JSON.stringify(o));

function snapshotState() {
  const out = {};
  for (const k of PRESET_KEYS) out[k] = state[k];
  return out;
}
function applyPreset(id) {
  const [kind, name] = id.split(':');
  const src = kind === 'factory' ? FACTORY_PRESETS[name] : loadUserPresets()[name];
  if (!src) return;
  for (const k of PRESET_KEYS) if (k in src) state[k] = src[k];
  if (state.mode >= 10 && state.mode <= 14) advec.needsReset = true;
  pane.refresh();
  updateModeFolders();
}

const presetUI = { current: '', newName: '' };
const fPresets = tabSaved.addFolder({ title: 'Presets', expanded: true });

function buildPresetOptions() {
  const opts = { '— select —': '' };
  for (const k of Object.keys(FACTORY_PRESETS)) opts['★ ' + k] = 'factory:' + k;
  const user = loadUserPresets();
  for (const k of Object.keys(user))            opts['user · ' + k] = 'user:' + k;
  return opts;
}

function rebuildPresetsFolder() {
  while (fPresets.children.length) fPresets.children[0].dispose();
  presetUI.current = '';
  fPresets.addBinding(presetUI, 'current', { label: 'load', options: buildPresetOptions() })
    .on('change', e => { if (e.value) applyPreset(e.value); });
  fPresets.addBinding(presetUI, 'newName', { label: 'name' });
  fPresets.addButton({ title: 'Save current as preset' }).on('click', () => {
    const name = presetUI.newName.trim();
    if (!name) return;
    const user = loadUserPresets();
    user[name] = snapshotState();
    saveUserPresetsToLS(user);
    presetUI.newName = '';
    rebuildPresetsFolder();
  });
  fPresets.addButton({ title: 'Delete selected (user only)' }).on('click', () => {
    if (!presetUI.current.startsWith('user:')) return;
    const name = presetUI.current.slice(5);
    const user = loadUserPresets();
    delete user[name];
    saveUserPresetsToLS(user);
    rebuildPresetsFolder();
  });
}
rebuildPresetsFolder();

// ----- Starred modes export -----
const fStarred = tabSaved.addFolder({ title: 'Starred modes', expanded: true });
fStarred.addButton({ title: '📋 Copy starred summary' }).on('click', async () => {
  const entries = Object.keys(starred)
    .map(id => ({ id: +id, name: MODE_NAMES_FULL[+id] || `mode ${id}` }))
    .sort((a, b) => a.id - b.id);
  if (entries.length === 0) {
    alert('No starred modes yet — star some first.');
    return;
  }
  const text = `# Starred modes (transition-v3, ${new Date().toISOString().slice(0, 10)})\n` +
    entries.map(e => `★  ${e.name} (mode ${e.id})`).join('\n');
  try {
    await navigator.clipboard.writeText(text);
    alert(`Copied ${entries.length} starred mode${entries.length === 1 ? '' : 's'} to clipboard.\nPaste it back to Claude.`);
  } catch {
    console.log(text);
    alert('Clipboard blocked — see browser console for the summary.');
  }
});
fStarred.addButton({ title: 'Clear all stars' }).on('click', () => {
  if (!confirm('Clear all starred modes?')) return;
  for (const k of Object.keys(starred)) delete starred[k];
  saveStarred();
});

// Keyboard
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === ' ')          { e.preventDefault(); togglePlay(); }
  if (e.key === 'Tab')        { e.preventDefault(); setMinimized(!document.body.classList.contains('minimized')); }
  if (e.key === 'ArrowLeft')  { state.t = Math.max(0, state.t - 0.02); pane.refresh(); }
  if (e.key === 'ArrowRight') { state.t = Math.min(1, state.t + 0.02); pane.refresh(); }
});

// Expose for headless / automation experiments
// ----- Auto-persist all settings to localStorage -----
const SESSION_LS_KEY = 'transition-tool-v3:session';
const PERSIST_KEYS = [
  ...PRESET_KEYS,
  'fit', 'bg',
  'exportFps', 'exportSizeMode', 'exportPadBottom',
  'slotAFillMode', 'slotAColor', 'slotBFillMode', 'slotBColor', 'keepAOutsideB',
];
function saveSession() {
  try {
    const out = {};
    for (const k of PERSIST_KEYS) out[k] = state[k];
    localStorage.setItem(SESSION_LS_KEY, JSON.stringify(out));
  } catch {}
}
function loadSession() {
  try {
    const s = JSON.parse(localStorage.getItem(SESSION_LS_KEY) || 'null');
    if (!s) return;
    for (const k of PERSIST_KEYS) if (k in s) state[k] = s[k];
    // sync the pad-preset dropdown to whatever the restored slider value is
    const presets = [0, 0.5, 1.0, 1.416, 2.0];
    const match = presets.find(p => Math.abs(p - state.exportPadBottom) < 0.001);
    padPresets._v = match !== undefined ? match : 0;
    pane.refresh();
    updateModeFolders();
    if (state.mode >= 10 && state.mode <= 14) advec.needsReset = true;
  } catch {}
}
loadSession();
pane.on('change', () => saveSession());

window.__tool = { state, pane, device, adapter };
console.log('[transition-tool-v3] WebGPU ready, format:', presentationFormat);
