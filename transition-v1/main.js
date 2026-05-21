// Dreamy A→B transition driven entirely by the paintings' own content —
// luminance + edge masks shape the dissolve, with soft defocus, bloom, warmth and vignette
// breathing in at the midpoint. No directional displacement.

import { Pane } from 'tweakpane';

const canvas = document.getElementById('canvas');
const gl = canvas.getContext('webgl2', { premultipliedAlpha: false, preserveDrawingBuffer: true });
if (!gl) throw new Error('WebGL2 not supported');

// ---------- shaders ----------
const VS = `#version 300 es
in vec2 a_pos;
out vec2 v_uv;
void main() {
  v_uv = a_pos * 0.5 + 0.5;
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;

const FS = `#version 300 es
precision highp float;

uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform vec2 u_aspectA;     // (sx, sy) UV scale to fit image in canvas
uniform vec2 u_offsetA;     // UV offset to centre it
uniform vec2 u_aspectB;
uniform vec2 u_offsetB;
uniform float u_t;          // transition progress 0..1
uniform float u_softness;   // peak disc-blur radius (0..1)
uniform float u_glow;       // local bright-lift at peak
uniform float u_bloom;      // wide soft halo from bright strokes at peak
uniform float u_warmth;     // -1..1 colour-temperature shift at peak
uniform float u_vignette;   // radial darken at peak
uniform float u_organic;    // 0=pure noise mask, 1=pure luminance mask
uniform float u_maskScale;  // noise scale for the dissolve mask
uniform float u_spread;     // dissolve softness: low = crisp, high = wide fade
uniform float u_edges;      // edge-driven bias on the dissolve mask
uniform int   u_curve;      // 0 linear, 1 ease-in-out, 2 ease-in, 3 ease-out
uniform float u_seed;

// watercolor mode:
//   0 off, 1 rim, 2 paper, 3 blooms, 4 diffusion,
//   5 sediment, 6 salt, 7 iris,
//   8 wet bleed (fingering), 9 pigment run (gravity),
//   10 wet advection, 11 gravity advection, 12 curl-noise eddies,
//   13 brush-channel advection, 14 seed-point injection
uniform int   u_mode;
uniform float u_rimWidth;
uniform float u_rimDark;
uniform float u_paperAngle;        // 0..1 (0=horizontal fibers, 0.5=vertical)
uniform float u_paperAniso;        // 1..8 stretch along fibers
uniform float u_paperGranulation;  // high-freq speckle amount
uniform int   u_bloomCount;
uniform float u_bloomRim;
uniform float u_bloomRate;         // how fast each bloom grows
uniform float u_diffStrength;      // how strongly B bleeds into A's region
uniform float u_diffRadius;        // bleed kernel radius
uniform float u_sedBands;          // sediment: number of luminance bands (1..16)
uniform float u_sedSoftness;       // sediment: band-edge softness (0..1)
uniform int   u_sedDirection;      // sediment: 0=low values first, 1=high values first
uniform int   u_sedSource;         // sediment: 0 luma, 1 saturation, 2 hue, 3 detail, 4 temperature
uniform float u_saltDensity;       // salt: 0=few large crystals, 1=fine grain
uniform float u_saltContrast;      // salt: spot-vs-spread contrast
uniform float u_saltSparkle;       // salt: brightness pop when a crystal opens
uniform int   u_saltSource;        // salt: 0 none, 1 light, 2 dark, 3 colored, 4 edges
uniform float u_saltBias;          // salt: how strongly source biases reveal timing
uniform int   u_saltImage;         // salt: 0=A, 1=B, 2=both averaged
uniform vec2  u_irisFocus;         // iris: focal point in UV space
uniform float u_irisJitter;        // iris: front irregularity
// Wet bleed (fingering)
uniform float u_bleedFinger;       // finger anisotropy / freq
uniform float u_bleedAmount;       // perturbation strength
uniform float u_bleedHalo;         // saturated wet halo at the front
// Pigment run (gravity)
uniform float u_runGravity;        // 0 = pure luma, 1 = full top-to-bottom
uniform float u_runDrip;           // vertical sample offset
// Wet advection (FBO sim) — display path reads from the simulated state tex
uniform sampler2D u_advecState;

uniform vec3  u_bg;
uniform int   u_validA;
uniform int   u_validB;

in vec2 v_uv;
out vec4 frag;

// hash + fbm (cheap value noise)
float hash(vec2 p) {
  p = fract(p * vec2(123.34, 456.21));
  p += dot(p, p + 45.32);
  return fract(p.x * p.y);
}
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 5; i++) {
    v += a * vnoise(p);
    p *= 2.03;
    a *= 0.5;
  }
  return v;
}

float applyCurve(float x, int mode) {
  x = clamp(x, 0.0, 1.0);
  if (mode == 1) return x * x * (3.0 - 2.0 * x);          // ease-in-out
  if (mode == 2) return x * x;                             // ease-in
  if (mode == 3) return 1.0 - (1.0 - x) * (1.0 - x);       // ease-out
  return x;                                                 // linear
}

vec4 sampleFit(sampler2D tex, vec2 uv, vec2 scale, vec2 offset, int valid) {
  if (valid == 0) return vec4(u_bg, 1.0);
  vec2 q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return vec4(u_bg, 1.0);
  return texture(tex, q);
}

// soft disc blur — isotropic defocus that feels dreamy rather than smeared
vec4 softBlur(sampler2D tex, vec2 uv, float radius, vec2 scale, vec2 offset, int valid) {
  if (radius < 0.0005) return sampleFit(tex, uv, scale, offset, valid);
  vec4 acc = sampleFit(tex, uv, scale, offset, valid) * 0.35;
  float wsum = 0.35;
  // 12-tap ring at two radii — cheap soft disc kernel
  for (int i = 0; i < 12; i++) {
    float a = float(i) * (6.2831853 / 12.0) + u_seed * 0.013;
    vec2 d = vec2(cos(a), sin(a));
    float r = (i % 2 == 0) ? 1.0 : 0.55;
    float w = 1.0 - r * 0.45;
    acc += sampleFit(tex, uv + d * r * radius, scale, offset, valid) * w;
    wsum += w;
  }
  return acc / wsum;
}

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// brush-edge magnitude — high where the painting has detail/strokes
float edgeMag(sampler2D tex, vec2 uv, vec2 scale, vec2 offset, int valid) {
  if (valid == 0) return 0.0;
  float e = 0.0025;
  float cx1 = luma(sampleFit(tex, uv + vec2(e, 0.0), scale, offset, valid).rgb);
  float cx2 = luma(sampleFit(tex, uv - vec2(e, 0.0), scale, offset, valid).rgb);
  float cy1 = luma(sampleFit(tex, uv + vec2(0.0, e), scale, offset, valid).rgb);
  float cy2 = luma(sampleFit(tex, uv - vec2(0.0, e), scale, offset, valid).rgb);
  return clamp(length(vec2(cx1 - cx2, cy1 - cy2)) * 4.0, 0.0, 1.0);
}

// Paper-grain mask: anisotropic fbm along the fiber direction + fine granulation.
float paperMask(vec2 uv) {
  float ang = u_paperAngle * 3.14159265;
  float ca = cos(ang), sa = sin(ang);
  mat2 rot = mat2(ca, sa, -sa, ca);
  vec2 g = rot * (uv - 0.5);
  // stretch along the fiber axis → long, wavy striations
  vec2 stretched = vec2(g.x * u_maskScale, g.y * u_maskScale * u_paperAniso);
  float base = fbm(stretched + u_seed * 0.13);
  // tooth: high-freq speckle, simulates pigment grains in paper hollows
  float tooth = fbm(uv * (u_maskScale * 14.0) + u_seed * 1.7) - 0.5;
  return clamp(base + tooth * u_paperGranulation * 0.35, 0.0, 1.0);
}

// Backrun-bloom mask: distance to nearest seed, scaled by per-seed growth.
// minRevealT = the earliest time any bloom front reaches this pixel.
float bloomsMask(vec2 uv) {
  float minReveal = 1.0;
  for (int i = 0; i < 24; i++) {
    if (i >= u_bloomCount) break;
    float fi = float(i) + u_seed * 0.07 + 1.0;
    vec2 sp = vec2(hash(vec2(fi * 1.3, 13.0)), hash(vec2(fi * 2.7, 47.0)));
    float startT = hash(vec2(fi, 91.0)) * 0.4;     // staggered starts
    float jitter = 0.85 + 0.3 * hash(vec2(fi, 11.0)); // per-bloom growth rate
    float d = distance(uv, sp);
    // perturb the front shape with low-freq noise → irregular blob outline
    float wob = (fbm(uv * 4.0 + fi * 3.0) - 0.5) * 0.08;
    float reveal = startT + (d + wob) * (1.0 / max(u_bloomRate, 0.05)) * jitter;
    minReveal = min(minReveal, reveal);
  }
  return clamp(minReveal, 0.0, 1.0);
}

// Standard organic mask (modes 0, 1, 4) — noise + luminance + edges.
float organicMask(vec2 uv, float lumA, float lumB, float edge) {
  float n1 = fbm(uv * u_maskScale + u_seed * 0.13);
  float n2 = fbm(uv * u_maskScale * 2.3 + 17.0 + u_seed * 0.09);
  float noiseMask = mix(n1, n2, 0.35);
  float lumMask = 0.5 + 0.5 * (lumB - lumA);
  float m = mix(noiseMask, lumMask, u_organic);
  m -= u_edges * edge * 0.45;
  return clamp(m, 0.0, 1.0);
}

// Tonal sediment: a per-pixel image property quantized into bands, so distinct
// value-ranges reveal in stages — like watercolor sediment settling out.
// The "source" can be luminance, saturation, hue, edge-detail, or temperature.
float sedimentMask(vec2 uv, vec3 cA, vec3 cB) {
  vec3 src = (cA + cB) * 0.5;
  float v;

  if (u_sedSource == 0) {                 // luminance
    v = luma(src);
  } else if (u_sedSource == 1) {          // saturation (HSV chroma/max)
    float mx = max(max(src.r, src.g), src.b);
    float mn = min(min(src.r, src.g), src.b);
    v = (mx > 1e-4) ? (mx - mn) / mx : 0.0;
  } else if (u_sedSource == 2) {          // hue angle, mapped to [0,1]
    float mx = max(max(src.r, src.g), src.b);
    float mn = min(min(src.r, src.g), src.b);
    float c  = mx - mn;
    float h  = 0.0;
    if (c > 1e-4) {
      if (mx == src.r)       h = mod((src.g - src.b) / c, 6.0);
      else if (mx == src.g)  h = ((src.b - src.r) / c) + 2.0;
      else                   h = ((src.r - src.g) / c) + 4.0;
      h /= 6.0;
    }
    v = h;
  } else if (u_sedSource == 3) {          // local detail (edge magnitude)
    float eA = edgeMag(u_texA, uv, u_aspectA, u_offsetA, u_validA);
    float eB = edgeMag(u_texB, uv, u_aspectB, u_offsetB, u_validB);
    v = max(eA, eB);
  } else {                                 // temperature: warm ↔ cool
    v = clamp(0.5 + (src.r - src.b) * 0.7, 0.0, 1.0);
  }

  if (u_sedDirection == 1) v = 1.0 - v;

  float bands = max(1.0, u_sedBands);
  float quantized = floor(v * bands) / max(1.0, bands - 1.0);
  return clamp(mix(quantized, v, u_sedSoftness), 0.0, 1.0);
}

// Salt: scattered crystal-like dissolves. High-freq noise, contrasted so it
// reads as discrete sparkles rather than smooth speckle. Each "crystal"
// reveals at a slightly different t. Optionally biased by image content so
// crystals open earlier in bright / dark / saturated / edge-rich regions of
// the source paintings.
float saltMask(vec2 uv, vec3 colA, vec3 colB) {
  float density = 6.0 + u_saltDensity * 90.0;       // crystals per UV-unit
  float n1 = vnoise(uv * density + u_seed * 1.7);
  float n2 = vnoise(uv * density * 0.35 + u_seed * 0.3);
  float n = mix(n2, n1, 0.75);
  float k = 0.5 + u_saltContrast * 5.0;
  float m = clamp(0.5 + (n - 0.5) * k, 0.0, 1.0);
  float salt = 1.0 - m;

  // Image-content bias.
  if (u_saltSource != 0 && u_saltBias > 0.001) {
    vec3 src;
    if (u_saltImage == 0)      src = colA;
    else if (u_saltImage == 1) src = colB;
    else                       src = (colA + colB) * 0.5;

    float prop = 0.0;
    if (u_saltSource == 1) {              // light areas reveal first
      prop = luma(src);
    } else if (u_saltSource == 2) {       // dark areas reveal first
      prop = 1.0 - luma(src);
    } else if (u_saltSource == 3) {       // saturated/coloured areas
      float maxC = max(max(src.r, src.g), src.b);
      float minC = min(min(src.r, src.g), src.b);
      prop = (maxC > 1e-4) ? (maxC - minC) / maxC : 0.0;
    } else if (u_saltSource == 4) {       // edges of the chosen image(s)
      float eA = edgeMag(u_texA, uv, u_aspectA, u_offsetA, u_validA);
      float eB = edgeMag(u_texB, uv, u_aspectB, u_offsetB, u_validB);
      prop = (u_saltImage == 0) ? eA : (u_saltImage == 1) ? eB : max(eA, eB);
    }

    // higher prop → reveal earlier → lower mask
    salt = clamp(salt - prop * u_saltBias * 0.75, 0.0, 1.0);
  }

  return salt;
}

// Iris: radial reveal from a focal point, with optional jitter so the front
// breathes rather than expanding as a perfect circle.
float irisMask(vec2 uv) {
  vec2 d = uv - u_irisFocus;
  float r = length(d) * 1.4142;  // normalise so corners reach ~1
  float jitter = (fbm(uv * 3.5 + u_seed * 0.21) - 0.5) * u_irisJitter * 0.3;
  return clamp(r + jitter, 0.0, 1.0);
}

// Wet bleed mask: luminance-based base + anisotropic high-freq noise that
// produces finger-like protrusions at the dissolve front (Saffman-Taylor
// instability look).
float wetBleedMask(vec2 uv, float lumA, float lumB) {
  float base = mix(0.5, 0.5 + 0.5 * (lumB - lumA), 0.55);
  // anisotropic noise stretched in one axis for fingering
  float aniso = mix(8.0, 28.0, u_bleedFinger);
  vec2 fingUV = uv * vec2(aniso, aniso * 0.35);
  float n1 = fbm(fingUV + u_seed * 0.3);
  // low-freq branching
  float n2 = fbm(uv * 3.0 + u_seed * 0.7);
  float fingers = (n1 - 0.5) * u_bleedAmount * 0.8 + (n2 - 0.5) * 0.18;
  return clamp(base + fingers, 0.0, 1.0);
}

// Pigment run: mask = mix of A's luminance (shadows of A reveal first, as if
// they're attracting the wash) with vertical position (gravity — bottom of
// canvas reveals before top). u_runGravity blends the two.
float pigmentRunMask(vec2 uv, float lumA) {
  // shadows-of-A first → low mask in dark areas: mask = lumA
  // gravity: bottom reveals first → mask = uv.y
  float m = mix(lumA, uv.y, u_runGravity);
  // small noise jitter so the front isn't a sharp horizontal line
  float n = (fbm(uv * 2.5 + u_seed * 0.11) - 0.5) * 0.06;
  return clamp(m + n, 0.0, 1.0);
}

// Lightweight mask used by the diffusion mode to peek at neighbours
// (skip the edge term — too many texture taps otherwise).
float maskAt(vec2 uv) {
  if (u_mode == 2) return paperMask(uv);
  if (u_mode == 3) return bloomsMask(uv);
  float lA = luma(sampleFit(u_texA, uv, u_aspectA, u_offsetA, u_validA).rgb);
  float lB = luma(sampleFit(u_texB, uv, u_aspectB, u_offsetB, u_validB).rgb);
  float n1 = fbm(uv * u_maskScale + u_seed * 0.13);
  float n2 = fbm(uv * u_maskScale * 2.3 + 17.0 + u_seed * 0.09);
  float noiseMask = mix(n1, n2, 0.35);
  float lumMask = 0.5 + 0.5 * (lB - lA);
  return clamp(mix(noiseMask, lumMask, u_organic), 0.0, 1.0);
}

void main() {
  vec2 uv = v_uv;

  // Advection family (modes 10..14): the heavy lifting happens in the sim
  // shader writing to a ping-pong FBO. Here we just sample the simulated
  // state and output it.
  if (u_mode >= 10 && u_mode <= 14) {
    frag = vec4(texture(u_advecState, uv).rgb, 1.0);
    return;
  }

  // shape the timeline. Stretch t so the per-pixel smoothstep window
  // (mask ± spread) is fully traversed for t ∈ [0,1] — otherwise pixels
  // with mask near 0 or 1 never reach a full reveal at the endpoints.
  float sp = mix(0.05, 0.7, u_spread);
  float tCurve = applyCurve(u_t, u_curve);
  float t = tCurve * (1.0 + 2.0 * sp) - sp;
  float env = pow(sin(3.14159265 * clamp(u_t, 0.0, 1.0)), 0.85);

  // soft defocus that breathes in at the midpoint
  float blurR = u_softness * env * 0.03;
  vec4 colA = softBlur(u_texA, uv, blurR, u_aspectA, u_offsetA, u_validA);
  vec4 colB = softBlur(u_texB, uv, blurR, u_aspectB, u_offsetB, u_validB);

  // ---- dissolve mask (mode-dependent) ----
  float mask;
  float lumA = luma(colA.rgb);
  float lumB = luma(colB.rgb);
  if (u_mode == 2) {
    mask = paperMask(uv);
    float lumMask = 0.5 + 0.5 * (lumB - lumA);
    mask = mix(mask, lumMask, u_organic * 0.35);
  } else if (u_mode == 3) {
    mask = bloomsMask(uv);
  } else if (u_mode == 5) {
    mask = sedimentMask(uv, colA.rgb, colB.rgb);
  } else if (u_mode == 6) {
    mask = saltMask(uv, colA.rgb, colB.rgb);
  } else if (u_mode == 7) {
    mask = irisMask(uv);
  } else if (u_mode == 8) {
    mask = wetBleedMask(uv, lumA, lumB);
  } else if (u_mode == 9) {
    mask = pigmentRunMask(uv, lumA);
  } else {
    float eA = edgeMag(u_texA, uv, u_aspectA, u_offsetA, u_validA);
    float eB = edgeMag(u_texB, uv, u_aspectB, u_offsetB, u_validB);
    mask = organicMask(uv, lumA, lumB, max(eA, eB));
  }

  // spread: low = crisp dissolve, high = wide gentle fade. sp computed above.
  float mixT = clamp(smoothstep(mask - sp, mask + sp, t), 0.0, 1.0);

  // ---- diffusion mode: B's pigment seeps into A as a soft anticipatory tint ----
  // The previous version pulled raw B pixels from neighbours, which produced
  // scratchy/streaky edges. Here we instead pre-blur B (= a "wet pigment pool")
  // and feather it into A in a region just ahead of the dissolve front. Result
  // is a soft wet-paper bleed without directional artefacts.
  vec3 colA_eff = colA.rgb;
  if (u_mode == 4 && u_diffStrength > 0.001) {
    // anticipation: 0 far from the front, 1 just before the pixel flips
    float anticipate = smoothstep(mask - 0.45, mask + 0.05, t);
    float br = 0.025 + u_diffRadius * 0.08;
    vec3 bleedB = softBlur(u_texB, uv, br, u_aspectB, u_offsetB, u_validB).rgb;
    float dry = 1.0 - mixT;
    colA_eff = mix(colA.rgb, bleedB, anticipate * dry * u_diffStrength * 0.55);
  }

  vec4 outc = vec4(mix(colA_eff, colB.rgb, mixT), 1.0);

  // ---- pigment rim: at the wet front, the local pigment looks slightly more
  // concentrated. We derive the rim colour from whatever's settling at this
  // pixel (mostly B's local hue) and apply a gentle multiplicative darken +
  // saturation boost — never a flat dark band — so the rim feels like the
  // painting's own pigment pooling, not a foreign stroke.
  if ((u_mode == 1 || u_mode == 3) && env > 0.02) {
    float rimW = (u_mode == 3) ? u_bloomRim * 0.5 : u_rimWidth * 0.4;
    if (rimW > 0.001) {
      float band = 1.0 - smoothstep(0.0, rimW, abs(t - mask));
      // local pigment about to settle here: lean toward B since B is the
      // arriving wash, but keep some A so the rim blends with what was there.
      vec3 base = mix(colA.rgb, colB.rgb, 0.65);
      // concentrate the local hue: slight chroma boost, gentle darken.
      float lum = luma(base);
      vec3 chromaBoost = clamp(mix(vec3(lum), base, 1.35), 0.0, 1.0);
      vec3 rim = chromaBoost * 0.78;
      float darkness = (u_mode == 3) ? u_bloomRim : u_rimDark;
      // env² for a softer ramp at start/end so the rim doesn't pop.
      float fade = env * env;
      outc.rgb = mix(outc.rgb, rim, band * darkness * fade * 0.85);
    }
  }

  // ---- wet bleed: saturated wet halo just inside the front ----
  if (u_mode == 8 && u_bleedHalo > 0.001 && env > 0.02) {
    float haloW = 0.005 + u_bleedHalo * 0.06;
    // halo only where the pixel has already started revealing (inside B's territory)
    if (t > mask) {
      float band = exp(-pow((t - mask) / haloW, 2.0));
      vec3 base = mix(colA.rgb, colB.rgb, 0.75);
      float lum = luma(base);
      vec3 saturated = clamp(mix(vec3(lum), base, 1.5), 0.0, 1.0);
      outc.rgb = mix(outc.rgb, saturated, band * u_bleedHalo * env * 0.4);
    }
  }

  // ---- pigment run: vertical drip — pixels just past the front carry a
  // little of B's pigment from slightly above (as if running down). ----
  if (u_mode == 9 && u_runDrip > 0.001 && env > 0.02 && t > mask) {
    float dripBand = exp(-pow((t - mask) / 0.08, 2.0));
    vec3 dripB = sampleFit(u_texB, uv + vec2(0.0, u_runDrip * 0.05),
                           u_aspectB, u_offsetB, u_validB).rgb;
    outc.rgb = mix(outc.rgb, dripB, dripBand * u_runDrip * env * 0.35);
  }

  // ---- bloom halo: wide blur of the bright regions, screen-blended back at peak ----
  if (u_bloom > 0.001 && env > 0.02) {
    float bR = 0.025 + u_bloom * 0.05;
    vec4 bA = softBlur(u_texA, uv, bR, u_aspectA, u_offsetA, u_validA);
    vec4 bB = softBlur(u_texB, uv, bR, u_aspectB, u_offsetB, u_validB);
    vec3 bloomCol = mix(bA.rgb, bB.rgb, mixT);
    float bMask = smoothstep(0.45, 0.95, luma(bloomCol));
    outc.rgb = 1.0 - (1.0 - outc.rgb) * (1.0 - bloomCol * bMask * u_bloom * env * 0.7); // screen
  }

  // ---- local bright-lift glow ----
  float lum = luma(outc.rgb);
  float lift = u_glow * env * smoothstep(0.25, 0.85, lum);
  outc.rgb += lift * 0.16;

  // ---- warmth: subtle temperature shift at peak ----
  outc.rgb += vec3(u_warmth, u_warmth * 0.25, -u_warmth) * env * 0.08;

  // ---- vignette breathing at peak ----
  float d = length(uv - 0.5) * 1.4142;
  float vig = smoothstep(0.55, 1.0, d) * u_vignette * env;
  outc.rgb *= 1.0 - vig * 0.55;

  frag = vec4(clamp(outc.rgb, 0.0, 1.0), 1.0);
}`;

// ---------- gl helpers ----------
function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s), src);
    throw new Error('shader compile failed');
  }
  return s;
}
function program(vs, fs) {
  const p = gl.createProgram();
  gl.attachShader(p, compile(gl.VERTEX_SHADER, vs));
  gl.attachShader(p, compile(gl.FRAGMENT_SHADER, fs));
  gl.linkProgram(p);
  if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
    console.error(gl.getProgramInfoLog(p));
    throw new Error('program link failed');
  }
  return p;
}

const prog = program(VS, FS);
gl.useProgram(prog);

const quad = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quad);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1,-1, 1,-1, -1,1,  -1,1, 1,-1, 1,1]), gl.STATIC_DRAW);
const posLoc = gl.getAttribLocation(prog, 'a_pos');
gl.enableVertexAttribArray(posLoc);
gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

const uni = name => gl.getUniformLocation(prog, name);
const U = {
  texA: uni('u_texA'), texB: uni('u_texB'),
  aspA: uni('u_aspectA'), offA: uni('u_offsetA'),
  aspB: uni('u_aspectB'), offB: uni('u_offsetB'),
  t: uni('u_t'),
  softness: uni('u_softness'), glow: uni('u_glow'), bloom: uni('u_bloom'),
  warmth: uni('u_warmth'), vignette: uni('u_vignette'),
  organic: uni('u_organic'), maskScale: uni('u_maskScale'),
  spread: uni('u_spread'), edges: uni('u_edges'), curve: uni('u_curve'),
  seed: uni('u_seed'),
  mode: uni('u_mode'),
  rimWidth: uni('u_rimWidth'), rimDark: uni('u_rimDark'),
  paperAngle: uni('u_paperAngle'), paperAniso: uni('u_paperAniso'),
  paperGranulation: uni('u_paperGranulation'),
  bloomCount: uni('u_bloomCount'), bloomRim: uni('u_bloomRim'), bloomRate: uni('u_bloomRate'),
  diffStrength: uni('u_diffStrength'), diffRadius: uni('u_diffRadius'),
  sedBands: uni('u_sedBands'), sedSoftness: uni('u_sedSoftness'), sedDirection: uni('u_sedDirection'),
  sedSource: uni('u_sedSource'),
  saltDensity: uni('u_saltDensity'), saltContrast: uni('u_saltContrast'), saltSparkle: uni('u_saltSparkle'),
  saltSource: uni('u_saltSource'), saltBias: uni('u_saltBias'), saltImage: uni('u_saltImage'),
  irisFocus: uni('u_irisFocus'), irisJitter: uni('u_irisJitter'),
  bleedFinger: uni('u_bleedFinger'), bleedAmount: uni('u_bleedAmount'), bleedHalo: uni('u_bleedHalo'),
  runGravity: uni('u_runGravity'), runDrip: uni('u_runDrip'),
  advecState: uni('u_advecState'),
  bg: uni('u_bg'), validA: uni('u_validA'), validB: uni('u_validB')
};
gl.uniform1i(U.texA, 0);
gl.uniform1i(U.texB, 1);
gl.uniform1i(U.advecState, 2);

function makeTex() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  // 1x1 placeholder
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0,0,0,255]));
  return t;
}
const texA = makeTex();
const texB = makeTex();

function uploadImage(tex, img) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
}

// ============================================================================
// Wet advection (FBO ping-pong): one diffusion step per pass, run K passes
// per frame to advance from lastT toward state.t. Scrubbing backward triggers
// a re-init of the state to A and a warm-up to the current t.
// ============================================================================

const SIM_FS = `#version 300 es
precision highp float;
uniform sampler2D u_state;
uniform sampler2D u_texA;
uniform sampler2D u_texB;
uniform vec2 u_aspectA, u_offsetA;
uniform vec2 u_aspectB, u_offsetB;
uniform int  u_validA, u_validB;
uniform vec2 u_pixel;
uniform float u_t;
uniform float u_visc;
uniform float u_rate;
uniform float u_seed;
uniform float u_maskScale;
uniform float u_organic;
uniform float u_spread;
uniform vec3  u_bg;
// Variant selector: 0 default, 1 gravity, 2 curl, 3 brush-channel, 4 seed
uniform int   u_variant;
uniform float u_advGravity;     // gravity: along-flow weight bias
uniform float u_advGravBias;    // gravity: mix shadows-of-A vs flow direction in the mask
uniform float u_advGravAngle;   // gravity: flow direction (0=down, 0.25=right, 0.5=up, 0.75=left)
uniform float u_advGravStreak;  // gravity: along-flow anisotropy (longer streaks at high values)
uniform float u_advGravLateral; // gravity: perpendicular spread
uniform float u_advCurlStr;     // curl: velocity strength
uniform float u_advCurlScale;   // curl: noise frequency
uniform float u_advBrushFollow; // brush-channel: anisotropy strength
uniform int   u_advSeedCount;   // seed: number of seeds
uniform float u_advSeedRadius;  // seed: per-seed injection radius
uniform int   u_advSeedFromImg; // seed: 0 random hash, 1 use uploaded positions
uniform vec2  u_advSeeds[16];   // seed positions in UV space (only used when u_advSeedFromImg == 1)

in vec2 v_uv;
out vec4 frag;

float hash(vec2 p) { p = fract(p * vec2(123.34, 456.21)); p += dot(p, p + 45.32); return fract(p.x * p.y); }
float vnoise(vec2 p) {
  vec2 i = floor(p), f = fract(p);
  vec2 u = f * f * (3.0 - 2.0 * f);
  float a = hash(i);
  float b = hash(i + vec2(1.0, 0.0));
  float c = hash(i + vec2(0.0, 1.0));
  float d = hash(i + vec2(1.0, 1.0));
  return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}
float fbm(vec2 p) {
  float v = 0.0, a = 0.5;
  for (int i = 0; i < 4; i++) { v += a * vnoise(p); p *= 2.03; a *= 0.5; }
  return v;
}
float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }
vec4 sampleFit(sampler2D tex, vec2 uv, vec2 scale, vec2 offset, int valid) {
  if (valid == 0) return vec4(u_bg, 1.0);
  vec2 q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return vec4(u_bg, 1.0);
  return texture(tex, q);
}

// 2D curl of a scalar fbm field — gives a divergence-free velocity field.
vec2 curlField(vec2 uv) {
  float e = 0.004;
  vec2 p = uv * u_advCurlScale;
  float p_yp = fbm(p + vec2(0.0, e));
  float p_yn = fbm(p - vec2(0.0, e));
  float p_xp = fbm(p + vec2(e, 0.0));
  float p_xn = fbm(p - vec2(e, 0.0));
  return vec2(p_yp - p_yn, p_xn - p_xp);
}

void main() {
  vec2 uv = v_uv;

  // ---- A and B colour samples (shared across variants) ----
  vec3 cA = sampleFit(u_texA, uv, u_aspectA, u_offsetA, u_validA).rgb;
  vec3 cB = sampleFit(u_texB, uv, u_aspectB, u_offsetB, u_validB).rgb;
  float lA = luma(cA), lB = luma(cB);

  // ---- Read previous state, possibly advected ----
  vec3 cur;
  if (u_variant == 2 && u_advCurlStr > 0.001) {
    // Curl-noise: read state from upstream (uv - velocity)
    vec2 vel = curlField(uv) * u_advCurlStr * 0.06;
    cur = texture(u_state, uv - vel).rgb;
  } else {
    cur = texture(u_state, uv).rgb;
  }

  // ---- Diffusion step (variant-dependent kernel) ----
  vec3 nb;
  if (u_variant == 1) {
    // Gravity with arbitrary direction. flowDir = (sin(a), -cos(a)) so:
    //   angle 0    → down  (0,-1)
    //   angle 0.25 → right (1, 0)
    //   angle 0.5  → up    (0, 1)
    //   angle 0.75 → left (-1, 0)
    // Pigment "falls" along +flowDir; the neighbour kernel weights the
    // upstream direction (against flow) higher so pigment accumulates here.
    float a = u_advGravAngle * 6.2831853;
    vec2 flowDir = vec2(sin(a), -cos(a));
    vec2 perp    = vec2(-flowDir.y, flowDir.x);

    // Anisotropy: streak elongates the kernel along the flow axis.
    float streak = 1.0 + u_advGravStreak * 2.5;
    vec2 upstep   = -flowDir * u_pixel.x * streak;  // sample upstream
    vec2 downstep =  flowDir * u_pixel.x * streak;
    vec2 latstep  =  perp    * u_pixel.x;

    float wUp   = 1.0 + u_advGravity * 1.6;
    float wDown = max(0.0, 1.0 - u_advGravity * 0.85);
    float wLat  = 0.3 + u_advGravLateral * 1.4;
    float wDiagU = (wUp   + wLat) * 0.5;
    float wDiagD = (wDown + wLat) * 0.5;
    float wsum   = wUp + wDown + 2.0 * wLat + 2.0 * wDiagU + 2.0 * wDiagD;

    nb = (
      texture(u_state, uv + upstep).rgb                 * wUp +
      texture(u_state, uv + downstep).rgb               * wDown +
      texture(u_state, uv + latstep).rgb                * wLat +
      texture(u_state, uv - latstep).rgb                * wLat +
      texture(u_state, uv + upstep * 0.7071 + latstep * 0.7071).rgb   * wDiagU +
      texture(u_state, uv + upstep * 0.7071 - latstep * 0.7071).rgb   * wDiagU +
      texture(u_state, uv + downstep * 0.7071 + latstep * 0.7071).rgb * wDiagD +
      texture(u_state, uv + downstep * 0.7071 - latstep * 0.7071).rgb * wDiagD
    ) / wsum;
  } else if (u_variant == 3) {
    // Brush-channel: anisotropic diffusion along A's stroke direction
    // (perpendicular to its luma gradient).
    float e = 0.003;
    float gx = luma(sampleFit(u_texA, uv + vec2(e, 0.0), u_aspectA, u_offsetA, u_validA).rgb) -
               luma(sampleFit(u_texA, uv - vec2(e, 0.0), u_aspectA, u_offsetA, u_validA).rgb);
    float gy = luma(sampleFit(u_texA, uv + vec2(0.0, e), u_aspectA, u_offsetA, u_validA).rgb) -
               luma(sampleFit(u_texA, uv - vec2(0.0, e), u_aspectA, u_offsetA, u_validA).rgb);
    vec2 grad = vec2(gx, gy);
    float glen = length(grad);
    vec2 sd = (glen > 1e-4) ? vec2(-grad.y, grad.x) / glen : vec2(1.0, 0.0);
    // Weight along ±sd higher than perpendicular.
    float follow = u_advBrushFollow;
    vec2 perp = vec2(-sd.y, sd.x);
    float wAlong = 1.0 + follow * 1.5;
    float wPerp  = max(0.0, 1.0 - follow * 0.85);
    nb = (
      texture(u_state, uv + sd * u_pixel.x * 1.4).rgb * wAlong +
      texture(u_state, uv - sd * u_pixel.x * 1.4).rgb * wAlong +
      texture(u_state, uv + perp * u_pixel.x).rgb * wPerp +
      texture(u_state, uv - perp * u_pixel.x).rgb * wPerp
    ) / (2.0 * wAlong + 2.0 * wPerp + 1e-4);
  } else {
    // Default isotropic 8-tap (used by variants 0, 2, 4).
    nb = (
      texture(u_state, uv + vec2(u_pixel.x, 0.0)).rgb +
      texture(u_state, uv - vec2(u_pixel.x, 0.0)).rgb +
      texture(u_state, uv + vec2(0.0, u_pixel.y)).rgb +
      texture(u_state, uv - vec2(0.0, u_pixel.y)).rgb +
      texture(u_state, uv + u_pixel * 0.7071).rgb +
      texture(u_state, uv - u_pixel * 0.7071).rgb +
      texture(u_state, uv + vec2(u_pixel.x, -u_pixel.y) * 0.7071).rgb +
      texture(u_state, uv + vec2(-u_pixel.x, u_pixel.y) * 0.7071).rgb
    ) * 0.125;
  }
  vec3 diffused = mix(cur, nb, u_visc);

  // ---- Mask & B-injection (variant-dependent) ----
  float mask;
  float n = fbm(uv * u_maskScale + u_seed * 0.13);
  float lumMask = 0.5 + 0.5 * (lB - lA);

  if (u_variant == 1) {
    // Bias toward shadows-of-A AND flow progress (pixels far along the flow
    // direction reveal first). flowDir is recomputed here.
    float a = u_advGravAngle * 6.2831853;
    vec2 flowDir = vec2(sin(a), -cos(a));
    // projection of (uv - centre) onto flowDir: positive = far along flow
    float flowProgress = 0.5 - dot(uv - 0.5, flowDir);
    float shadow = lA;
    float gMask = mix(shadow, flowProgress, u_advGravBias);
    mask = mix(n, gMask, u_organic);
  } else if (u_variant == 4) {
    // Seed-point: distance to nearest seed → reveal threshold. Seeds come from
    // either uploaded image-derived positions or hash-based randomness.
    float minD = 9999.0;
    for (int i = 0; i < 16; i++) {
      if (i >= u_advSeedCount) break;
      vec2 sp;
      if (u_advSeedFromImg == 1) {
        sp = u_advSeeds[i];
      } else {
        float fi = float(i) + u_seed * 0.07 + 1.0;
        sp = vec2(hash(vec2(fi * 1.3, 13.0)), hash(vec2(fi * 2.7, 47.0)));
      }
      float d = distance(uv, sp);
      minD = min(minD, d);
    }
    mask = clamp(minD / max(u_advSeedRadius, 0.05), 0.0, 1.0);
  } else {
    // Default / curl / brush: noise + lum mix
    mask = mix(n, lumMask, u_organic);
  }

  float sp = mix(0.1, 0.5, u_spread);
  // Stretch t so end-of-timeline pixels fully reveal.
  float tR = u_t * (1.0 + 2.0 * sp) - sp;
  float reveal = smoothstep(mask - sp, mask + sp * 0.3, tR);
  vec3 mixed = mix(diffused, cB, reveal * u_rate);

  frag = vec4(mixed, 1.0);
}`;

const INIT_FS = `#version 300 es
precision highp float;
uniform sampler2D u_texA;
uniform vec2 u_aspectA, u_offsetA;
uniform int u_validA;
uniform vec3 u_bg;
in vec2 v_uv;
out vec4 frag;
vec4 sampleFit(sampler2D tex, vec2 uv, vec2 scale, vec2 offset, int valid) {
  if (valid == 0) return vec4(u_bg, 1.0);
  vec2 q = (uv - offset) / scale;
  if (q.x < 0.0 || q.x > 1.0 || q.y < 0.0 || q.y > 1.0) return vec4(u_bg, 1.0);
  return texture(tex, q);
}
void main() {
  frag = vec4(sampleFit(u_texA, v_uv, u_aspectA, u_offsetA, u_validA).rgb, 1.0);
}`;

const simProg = program(VS, SIM_FS);
gl.useProgram(simProg);
{
  const pl = gl.getAttribLocation(simProg, 'a_pos');
  gl.enableVertexAttribArray(pl);
  gl.vertexAttribPointer(pl, 2, gl.FLOAT, false, 0, 0);
}
const simUni = {
  state:     gl.getUniformLocation(simProg, 'u_state'),
  texA:      gl.getUniformLocation(simProg, 'u_texA'),
  texB:      gl.getUniformLocation(simProg, 'u_texB'),
  aspA:      gl.getUniformLocation(simProg, 'u_aspectA'),
  offA:      gl.getUniformLocation(simProg, 'u_offsetA'),
  aspB:      gl.getUniformLocation(simProg, 'u_aspectB'),
  offB:      gl.getUniformLocation(simProg, 'u_offsetB'),
  validA:    gl.getUniformLocation(simProg, 'u_validA'),
  validB:    gl.getUniformLocation(simProg, 'u_validB'),
  pixel:     gl.getUniformLocation(simProg, 'u_pixel'),
  t:         gl.getUniformLocation(simProg, 'u_t'),
  visc:      gl.getUniformLocation(simProg, 'u_visc'),
  rate:      gl.getUniformLocation(simProg, 'u_rate'),
  seed:      gl.getUniformLocation(simProg, 'u_seed'),
  maskScale: gl.getUniformLocation(simProg, 'u_maskScale'),
  organic:   gl.getUniformLocation(simProg, 'u_organic'),
  spread:    gl.getUniformLocation(simProg, 'u_spread'),
  bg:        gl.getUniformLocation(simProg, 'u_bg'),
  variant:    gl.getUniformLocation(simProg, 'u_variant'),
  gravity:    gl.getUniformLocation(simProg, 'u_advGravity'),
  gravBias:   gl.getUniformLocation(simProg, 'u_advGravBias'),
  gravAngle:  gl.getUniformLocation(simProg, 'u_advGravAngle'),
  gravStreak: gl.getUniformLocation(simProg, 'u_advGravStreak'),
  gravLateral: gl.getUniformLocation(simProg, 'u_advGravLateral'),
  curlStr:    gl.getUniformLocation(simProg, 'u_advCurlStr'),
  curlScale:  gl.getUniformLocation(simProg, 'u_advCurlScale'),
  brushFollow: gl.getUniformLocation(simProg, 'u_advBrushFollow'),
  seedCount:    gl.getUniformLocation(simProg, 'u_advSeedCount'),
  seedRadius:   gl.getUniformLocation(simProg, 'u_advSeedRadius'),
  seedFromImg:  gl.getUniformLocation(simProg, 'u_advSeedFromImg'),
  seeds:        gl.getUniformLocation(simProg, 'u_advSeeds'),
};
gl.uniform1i(simUni.state, 2);
gl.uniform1i(simUni.texA, 0);
gl.uniform1i(simUni.texB, 1);

const initProg = program(VS, INIT_FS);
gl.useProgram(initProg);
{
  const pl = gl.getAttribLocation(initProg, 'a_pos');
  gl.enableVertexAttribArray(pl);
  gl.vertexAttribPointer(pl, 2, gl.FLOAT, false, 0, 0);
}
const initUni = {
  texA:   gl.getUniformLocation(initProg, 'u_texA'),
  aspA:   gl.getUniformLocation(initProg, 'u_aspectA'),
  offA:   gl.getUniformLocation(initProg, 'u_offsetA'),
  validA: gl.getUniformLocation(initProg, 'u_validA'),
  bg:     gl.getUniformLocation(initProg, 'u_bg'),
};
gl.uniform1i(initUni.texA, 0);

// FBO pair
let fboA = null, fboB = null;
let fboTexA = null, fboTexB = null;
let fboW = 0, fboH = 0;

function makeFBOTex(w, h) {
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  return tex;
}
function makeFBO(tex) {
  const fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
  return fbo;
}
function ensureFBOs() {
  const w = canvas.width, h = canvas.height;
  if (w === fboW && h === fboH && fboA) return;
  if (fboTexA) { gl.deleteTexture(fboTexA); gl.deleteFramebuffer(fboA); }
  if (fboTexB) { gl.deleteTexture(fboTexB); gl.deleteFramebuffer(fboB); }
  fboTexA = makeFBOTex(w, h);
  fboA    = makeFBO(fboTexA);
  fboTexB = makeFBOTex(w, h);
  fboB    = makeFBO(fboTexB);
  fboW = w; fboH = h;
  advec.needsReset = true;
}

// Which FBO currently holds "source" state. Read from src, write to !src,
// then swap so the just-written FBO becomes the next source.
const advec = { src: 'A', lastT: 0, needsReset: true };

function bindImageTextures() {
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);
}

function initAdvecState() {
  ensureFBOs();
  const dst = (advec.src === 'A') ? fboA : fboB;
  gl.bindFramebuffer(gl.FRAMEBUFFER, dst);
  gl.viewport(0, 0, fboW, fboH);
  gl.useProgram(initProg);
  bindImageTextures();
  const fA = composedFit('A', fboW, fboH);
  gl.uniform2f(initUni.aspA, fA.sx, fA.sy);
  gl.uniform2f(initUni.offA, fA.ox, fA.oy);
  gl.uniform1i(initUni.validA, state.imgA ? 1 : 0);
  const bg = hexToRgb(state.bg);
  gl.uniform3f(initUni.bg, bg[0], bg[1], bg[2]);
  gl.drawArrays(gl.TRIANGLES, 0, 6);
  advec.lastT = 0;
  advec.needsReset = false;
}

function advecStep(tAt) {
  const srcTex = (advec.src === 'A') ? fboTexA : fboTexB;
  const dstFBO = (advec.src === 'A') ? fboB : fboA;
  gl.bindFramebuffer(gl.FRAMEBUFFER, dstFBO);
  gl.viewport(0, 0, fboW, fboH);
  gl.useProgram(simProg);
  bindImageTextures();
  gl.activeTexture(gl.TEXTURE2);
  gl.bindTexture(gl.TEXTURE_2D, srcTex);

  const fA = composedFit('A', fboW, fboH);
  const fB = composedFit('B', fboW, fboH);
  gl.uniform2f(simUni.aspA, fA.sx, fA.sy); gl.uniform2f(simUni.offA, fA.ox, fA.oy);
  gl.uniform2f(simUni.aspB, fB.sx, fB.sy); gl.uniform2f(simUni.offB, fB.ox, fB.oy);
  gl.uniform1i(simUni.validA, state.imgA ? 1 : 0);
  gl.uniform1i(simUni.validB, state.imgB ? 1 : 0);
  gl.uniform2f(simUni.pixel, 1.0 / fboW, 1.0 / fboH);
  gl.uniform1f(simUni.t, tAt);
  gl.uniform1f(simUni.visc, state.advecVisc);
  gl.uniform1f(simUni.rate, state.advecRate);
  gl.uniform1f(simUni.seed, state.seed);
  gl.uniform1f(simUni.maskScale, state.maskScale);
  gl.uniform1f(simUni.organic, state.organic);
  gl.uniform1f(simUni.spread, state.spread);
  // variant = state.mode - 10
  gl.uniform1i(simUni.variant, state.mode - 10);
  gl.uniform1f(simUni.gravity, state.advecGravity);
  gl.uniform1f(simUni.gravBias, state.advecGravBias);
  gl.uniform1f(simUni.gravAngle, state.advecGravAngle);
  gl.uniform1f(simUni.gravStreak, state.advecGravStreak);
  gl.uniform1f(simUni.gravLateral, state.advecGravLateral);
  gl.uniform1f(simUni.curlStr, state.advecCurlStr);
  gl.uniform1f(simUni.curlScale, state.advecCurlScale);
  gl.uniform1f(simUni.brushFollow, state.advecBrushFollow);
  gl.uniform1i(simUni.seedCount, state.advecSeedCount);
  gl.uniform1f(simUni.seedRadius, state.advecSeedRadius);
  gl.uniform1i(simUni.seedFromImg, state.advecSeedSource === 0 ? 0 : 1);
  gl.uniform2fv(simUni.seeds, seedPositions);
  const bg = hexToRgb(state.bg);
  gl.uniform3f(simUni.bg, bg[0], bg[1], bg[2]);

  gl.drawArrays(gl.TRIANGLES, 0, 6);
  advec.src = (advec.src === 'A') ? 'B' : 'A';
}

function runAdvection() {
  ensureFBOs();
  if (!state.imgA || !state.imgB) return null;

  // Reset if scrubbing backward, jumping, or first run.
  if (advec.needsReset || state.t < advec.lastT - 0.03 || state.t === 0) {
    initAdvecState();
    const warm = Math.max(8, Math.round(state.advecSteps * 8));
    for (let i = 0; i < warm; i++) {
      advecStep(state.t * ((i + 1) / warm));
    }
  } else {
    // Catch up from lastT → t, in N small steps
    const N = Math.max(1, Math.round(state.advecSteps));
    const startT = advec.lastT;
    const endT = state.t;
    for (let i = 0; i < N; i++) {
      advecStep(startT + (endT - startT) * ((i + 1) / N));
    }
  }
  advec.lastT = state.t;
  // Return the texture that holds the current state
  return (advec.src === 'A') ? fboTexA : fboTexB;
}

// ---------- state ----------
const state = {
  imgA: null,            // HTMLImageElement
  imgB: null,
  t: 0,                  // 0..1
  playing: false,
  reverse: false,
  loop: true,
  startTime: 0,
  duration: 15.0,
  softness: 0.0,
  glow: 0.0,
  bloom: 0.0,
  warmth: 0.0,
  vignette: 0.0,
  organic: 0.65,
  maskScale: 0.9,
  spread: 0.55,
  edges: 0.25,
  curve: 0, // linear
  seed: 42,
  mode: 0,                   // 0 off, 1 rim, 2 paper, 3 blooms, 4 diffusion
  rimWidth: 0.12,
  rimDark: 0.6,
  paperAngle: 0.0,           // 0 = horizontal fibers
  paperAniso: 4.0,
  paperGranulation: 0.5,
  bloomCount: 8,
  bloomRim: 0.6,
  bloomRate: 0.55,
  diffStrength: 0.55,
  diffRadius: 0.45,
  sedBands: 6,
  sedSoftness: 0.35,
  sedDirection: 0,                 // 0 low values first, 1 high values first
  sedSource: 0,                    // 0 luma, 1 saturation, 2 hue, 3 detail, 4 temperature
  saltDensity: 0.0,
  saltContrast: 0.55,
  saltSparkle: 0.0,
  saltSource: 1,      // 0 none, 1 light, 2 dark, 3 coloured, 4 edges
  saltBias: 0.6,
  saltImage: 2,       // 0 = A, 1 = B, 2 = both
  irisFocusX: 0.5,
  irisFocusY: 0.5,
  irisJitter: 0.35,
  bleedFinger: 0.5,
  bleedAmount: 0.45,
  bleedHalo: 0.5,
  runGravity: 0.5,
  runDrip: 0.35,
  advecVisc: 0.55,
  advecRate: 0.18,
  advecSteps: 3,
  advecGravity: 0.6,
  advecGravBias: 0.5,
  advecGravAngle: 0,        // 0 = straight down
  advecGravStreak: 0.4,
  advecGravLateral: 0.3,
  advecCurlStr: 0.5,
  advecCurlScale: 2.5,
  advecBrushFollow: 0.7,
  advecSeedCount: 5,
  advecSeedRadius: 0.45,
  advecSeedSource: 1,    // 0 random, 1 light, 2 dark, 3 coloured, 4 edges
  advecSeedImage: 2,     // 0 A, 1 B, 2 both
  fit: 'cover',
  bg: '#000000',
  zoomA: 1.0, panAx: 0.0, panAy: 0.0,
  zoomB: 1.0, panBx: 0.0, panBy: 0.0,
  exportFps: 24,
  exportSizeMode: 'src',
  exportPadBottom: 0,  // 0 = no padding; 1 = full-h black below; 1.416 ≈ Elverket floor:wall ratio
  recording: false,
};

function hexToRgb(hex) {
  const m = hex.match(/^#([0-9a-f]{6})$/i);
  if (!m) return [0, 0, 0];
  const n = parseInt(m[1], 16);
  return [((n >> 16) & 255) / 255, ((n >> 8) & 255) / 255, (n & 255) / 255];
}

// ---------- canvas sizing ----------
function fitInfo(img, cw, ch, mode) {
  if (!img) return { sx: 1, sy: 1, ox: 0, oy: 0 };
  const ia = img.naturalWidth / img.naturalHeight;
  const ca = cw / ch;
  if (mode === 'stretch') return { sx: 1, sy: 1, ox: 0, oy: 0 };
  if (mode === 'cover') {
    // image fills canvas, may crop
    if (ia > ca) { // image wider than canvas
      const sx = ca / ia;
      return { sx: sx, sy: 1, ox: (1 - sx) * 0.5, oy: 0 };
    } else {
      const sy = ia / ca;
      return { sx: 1, sy: sy, ox: 0, oy: (1 - sy) * 0.5 };
    }
  }
  // contain: letterbox — but our shader's UV is canvas-space, so flip the logic
  if (ia > ca) {
    const sy = ca / ia;
    return { sx: 1, sy: sy, ox: 0, oy: (1 - sy) * 0.5 };
  } else {
    const sx = ia / ca;
    return { sx: sx, sy: 1, ox: (1 - sx) * 0.5, oy: 0 };
  }
}

// Compose the user's per-image zoom + pan with the fit-mode transform.
// Math: the shader maps canvasUV → imageUV via q = (canvasUV - offset) / scale.
// We pre-apply user zoom Z and pan p before the fit, which gives:
//   scale'  = scale * Z
//   offset' = Z * offset + 0.5 * (1 - Z) - Z * pan
function composedFit(slot, cw, ch) {
  const img = slot === 'A' ? state.imgA : state.imgB;
  const z   = slot === 'A' ? state.zoomA : state.zoomB;
  const px  = slot === 'A' ? state.panAx : state.panBx;
  const py  = slot === 'A' ? state.panAy : state.panBy;
  const f = fitInfo(img, cw, ch, state.fit);
  return {
    sx: f.sx * z,
    sy: f.sy * z,
    ox: z * f.ox + 0.5 * (1 - z) - z * px,
    oy: z * f.oy + 0.5 * (1 - z) - z * py,
  };
}

// Sanity-cap the internal render size to the GPU's reported max-texture-size
// (or 8192, whichever is smaller). We still pass the source resolution through
// when it's smaller. Heights up to 1920, widths essentially unlimited up to
// the GPU cap. This lets very-wide source images render at 1:1.
// Cap to whatever the GPU reports — 4090/16384, M-series Macs/16384, older
// integrated/4096. The shader auto-downsamples sources larger than this.
const GL_MAX_TEX = gl.getParameter(gl.MAX_TEXTURE_SIZE);
console.log('[transition-tool] GL MAX_TEXTURE_SIZE =', GL_MAX_TEX);

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
  // Cap whichever dimension is larger to GL_MAX_TEX, preserving aspect.
  const longer = Math.max(w, h);
  if (longer > GL_MAX_TEX) {
    const scale = GL_MAX_TEX / longer;
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  canvas.width = w;
  canvas.height = h;
  const fit = Math.min(maxW / w, maxH / h, 1);
  canvas.style.width = (w * fit) + 'px';
  canvas.style.height = (h * fit) + 'px';
  gl.viewport(0, 0, w, h);
}

// ---------- render ----------
let timeStart = performance.now();
function render() {
  if (!state.imgA && !state.imgB) {
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    requestAnimationFrame(render);
    return;
  }

  if (state.playing) {
    const now = performance.now();
    const elapsed = (now - state.startTime) / 1000;
    let p = elapsed / state.duration;
    if (p >= 1) {
      if (state.loop) {
        state.startTime = now;
        p = 0;
        state.reverse = !state.reverse;
      } else {
        p = 1;
        state.playing = false;
        if (typeof btnPlay !== 'undefined') btnPlay.title = 'Play';
      }
    }
    state.t = state.reverse ? (1 - p) : p;
    // refresh only the t slider so the rest of the panel doesn't re-render each frame
    if (typeof bT !== 'undefined') bT.refresh();
  }

  // Advection mode runs its sim into FBOs first, then we display the result.
  let advecTex = null;
  if (state.mode >= 10 && state.mode <= 14) {
    advecTex = runAdvection();
  }

  // Switch back to the default framebuffer (canvas)
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0, 0, canvas.width, canvas.height);

  gl.useProgram(prog);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.activeTexture(gl.TEXTURE1);
  gl.bindTexture(gl.TEXTURE_2D, texB);
  if (advecTex) {
    gl.activeTexture(gl.TEXTURE2);
    gl.bindTexture(gl.TEXTURE_2D, advecTex);
  }
  // Reselect the main program's attribute layout (sim/init may have left a
  // different buffer bound — be safe and re-point the position attrib)
  gl.bindBuffer(gl.ARRAY_BUFFER, quad);
  gl.enableVertexAttribArray(posLoc);
  gl.vertexAttribPointer(posLoc, 2, gl.FLOAT, false, 0, 0);

  const cw = canvas.width, ch = canvas.height;
  const fA = composedFit('A', cw, ch);
  const fB = composedFit('B', cw, ch);
  gl.uniform2f(U.aspA, fA.sx, fA.sy);
  gl.uniform2f(U.offA, fA.ox, fA.oy);
  gl.uniform2f(U.aspB, fB.sx, fB.sy);
  gl.uniform2f(U.offB, fB.ox, fB.oy);

  pushUniforms();
  gl.drawArrays(gl.TRIANGLES, 0, 6);

  requestAnimationFrame(render);
}

function pushUniforms() {
  gl.uniform1f(U.t, state.t);
  gl.uniform1f(U.softness, state.softness);
  gl.uniform1f(U.glow, state.glow);
  gl.uniform1f(U.bloom, state.bloom);
  gl.uniform1f(U.warmth, state.warmth);
  gl.uniform1f(U.vignette, state.vignette);
  gl.uniform1f(U.organic, state.organic);
  gl.uniform1f(U.maskScale, state.maskScale);
  gl.uniform1f(U.spread, state.spread);
  gl.uniform1f(U.edges, state.edges);
  gl.uniform1i(U.curve, state.curve);
  gl.uniform1f(U.seed, state.seed);
  gl.uniform1i(U.mode, state.mode);
  gl.uniform1f(U.rimWidth, state.rimWidth);
  gl.uniform1f(U.rimDark, state.rimDark);
  gl.uniform1f(U.paperAngle, state.paperAngle);
  gl.uniform1f(U.paperAniso, state.paperAniso);
  gl.uniform1f(U.paperGranulation, state.paperGranulation);
  gl.uniform1i(U.bloomCount, state.bloomCount);
  gl.uniform1f(U.bloomRim, state.bloomRim);
  gl.uniform1f(U.bloomRate, state.bloomRate);
  gl.uniform1f(U.diffStrength, state.diffStrength);
  gl.uniform1f(U.diffRadius, state.diffRadius);
  gl.uniform1f(U.sedBands, state.sedBands);
  gl.uniform1f(U.sedSoftness, state.sedSoftness);
  gl.uniform1i(U.sedDirection, state.sedDirection);
  gl.uniform1i(U.sedSource, state.sedSource);
  gl.uniform1f(U.saltDensity, state.saltDensity);
  gl.uniform1f(U.saltContrast, state.saltContrast);
  gl.uniform1f(U.saltSparkle, state.saltSparkle);
  gl.uniform1i(U.saltSource, state.saltSource);
  gl.uniform1f(U.saltBias, state.saltBias);
  gl.uniform1i(U.saltImage, state.saltImage);
  gl.uniform2f(U.irisFocus, state.irisFocusX, state.irisFocusY);
  gl.uniform1f(U.irisJitter, state.irisJitter);
  gl.uniform1f(U.bleedFinger, state.bleedFinger);
  gl.uniform1f(U.bleedAmount, state.bleedAmount);
  gl.uniform1f(U.bleedHalo, state.bleedHalo);
  gl.uniform1f(U.runGravity, state.runGravity);
  gl.uniform1f(U.runDrip, state.runDrip);
  const bg = hexToRgb(state.bg);
  gl.uniform3f(U.bg, bg[0], bg[1], bg[2]);
  gl.uniform1i(U.validA, state.imgA ? 1 : 0);
  gl.uniform1i(U.validB, state.imgB ? 1 : 0);
}
requestAnimationFrame(render);

// ============================================================================
// UI — image slots (small custom HTML) + Tweakpane for everything else
// ============================================================================

const filepicker = document.getElementById('filepicker');
const dropHint   = document.getElementById('drop-hint');

let pickingSlot = null;
document.querySelectorAll('.slot').forEach(s => {
  s.addEventListener('click', () => { pickingSlot = s.dataset.slot; filepicker.click(); });
});
filepicker.addEventListener('change', e => {
  const f = e.target.files[0];
  if (f) loadFile(f, pickingSlot);
  e.target.value = '';
});

// IndexedDB persistence — remembers the last A and B blobs across sessions.
const IDB_NAME = 'transition-tool';
const IDB_STORE = 'images';
function idbOpen() {
  return new Promise((resolve, reject) => {
    const r = indexedDB.open(IDB_NAME, 1);
    r.onupgradeneeded = () => r.result.createObjectStore(IDB_STORE);
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

function loadFile(file, slot) {
  if (!file.type.startsWith('image/')) return;
  idbPut(slot === 'A' ? 'imageA' : 'imageB', file);
  loadFromUrl(URL.createObjectURL(file), slot);
}

function loadFromUrl(url, slot) {
  const img = new Image();
  img.crossOrigin = 'anonymous';
  img.onload = () => {
    if (slot === 'A') { state.imgA = img; uploadImage(texA, img); updateSlotPreview('A', url); }
    else              { state.imgB = img; uploadImage(texB, img); updateSlotPreview('B', url); }
    canvas.classList.remove('empty');
    dropHint.classList.add('hidden');
    resizeCanvas();
    advec.needsReset = true;
    recomputeSeedPositions();
    if (typeof maybeAutoplay === 'function') maybeAutoplay();
  };
  img.src = url;
}

// === image-aware seed positions (for seed-point injection mode) ============
// Downsample the chosen image to a small grid, score each pixel by the chosen
// property, greedy-pick top-N positions with a min-distance constraint so the
// seeds don't cluster. Positions are stored in UV space and uploaded to the
// sim shader as a vec2[16] uniform.
const MAX_SEEDS = 16;
let seedPositions = new Float32Array(MAX_SEEDS * 2);

function recomputeSeedPositions() {
  // 0=random, 1=light, 2=dark, 3=coloured, 4=edges
  if (state.advecSeedSource === 0) return;
  // Choose source image(s)
  let imgs;
  if (state.advecSeedImage === 0)      imgs = state.imgA ? [state.imgA] : [];
  else if (state.advecSeedImage === 1) imgs = state.imgB ? [state.imgB] : [];
  else                                  imgs = [state.imgA, state.imgB].filter(Boolean);
  if (imgs.length === 0) return;

  const ref = imgs[0];
  const w = 96, h = Math.max(8, Math.round(96 * ref.naturalHeight / ref.naturalWidth));
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true });

  // Accumulate score across selected images.
  const scoreGrid = new Float32Array(w * h);
  for (const im of imgs) {
    ctx.clearRect(0, 0, w, h);
    ctx.drawImage(im, 0, 0, w, h);
    const data = ctx.getImageData(0, 0, w, h).data;
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const i = (y * w + x) * 4;
        const r = data[i] / 255, g = data[i + 1] / 255, b = data[i + 2] / 255;
        const lum = 0.299 * r + 0.587 * g + 0.114 * b;
        let s;
        if (state.advecSeedSource === 1) {        // light
          s = lum;
        } else if (state.advecSeedSource === 2) { // dark
          s = 1 - lum;
        } else if (state.advecSeedSource === 3) { // coloured (HSV chroma/max)
          const mx = Math.max(r, g, b), mn = Math.min(r, g, b);
          s = mx > 1e-4 ? (mx - mn) / mx : 0;
        } else {                                   // edges (gradient magnitude)
          if (x === 0 || x === w - 1 || y === 0 || y === h - 1) { s = 0; }
          else {
            const lN = (data[((y - 1) * w + x) * 4] / 255) * 0.299 + (data[((y - 1) * w + x) * 4 + 1] / 255) * 0.587 + (data[((y - 1) * w + x) * 4 + 2] / 255) * 0.114;
            const lS = (data[((y + 1) * w + x) * 4] / 255) * 0.299 + (data[((y + 1) * w + x) * 4 + 1] / 255) * 0.587 + (data[((y + 1) * w + x) * 4 + 2] / 255) * 0.114;
            const lW = (data[(y * w + x - 1) * 4] / 255) * 0.299 + (data[(y * w + x - 1) * 4 + 1] / 255) * 0.587 + (data[(y * w + x - 1) * 4 + 2] / 255) * 0.114;
            const lE = (data[(y * w + x + 1) * 4] / 255) * 0.299 + (data[(y * w + x + 1) * 4 + 1] / 255) * 0.587 + (data[(y * w + x + 1) * 4 + 2] / 255) * 0.114;
            s = Math.hypot(lE - lW, lS - lN);
          }
        }
        scoreGrid[y * w + x] += s;
      }
    }
  }
  // Build score+position list
  const list = [];
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      list.push([scoreGrid[y * w + x] / imgs.length, (x + 0.5) / w, 1 - (y + 0.5) / h]);
    }
  }
  list.sort((a, b) => b[0] - a[0]);

  // Greedy non-clustering selection
  const minDist = 0.12;
  const picked = [];
  for (const item of list) {
    if (picked.length >= MAX_SEEDS) break;
    if (item[0] <= 0) break;
    let ok = true;
    for (const p of picked) {
      const dx = item[1] - p[1], dy = item[2] - p[2];
      if (dx * dx + dy * dy < minDist * minDist) { ok = false; break; }
    }
    if (ok) picked.push(item);
  }
  // Fill any unused slots with the lowest-scoring picked position so the
  // distance field still resolves; sim respects u_advSeedCount as the cutoff.
  while (picked.length < MAX_SEEDS) picked.push(picked[picked.length - 1] || [0, 0.5, 0.5]);
  for (let i = 0; i < MAX_SEEDS; i++) {
    seedPositions[i * 2]     = picked[i][1];
    seedPositions[i * 2 + 1] = picked[i][2];
  }
  advec.needsReset = true;
}

function updateSlotPreview(slot, url) {
  const el = document.querySelector(`.slot[data-slot="${slot}"]`);
  el.querySelector('.placeholder')?.remove();
  let img = el.querySelector('img');
  if (!img) { img = document.createElement('img'); el.appendChild(img); }
  img.src = url;
}

// Auto-load bundled defaults; once both have loaded, kick off playback.
let _autoplayStarted = false;
function maybeAutoplay() {
  if (_autoplayStarted) return;
  if (!state.imgA || !state.imgB) return;
  _autoplayStarted = true;
  state.playing = true;
  state.t = 0;
  state.startTime = performance.now();
  // btnPlay may not exist yet at the moment images finish — guard.
  if (typeof btnPlay !== 'undefined') btnPlay.title = 'Pause';
}
(async () => {
  const [blobA, blobB] = await Promise.all([idbGet('imageA'), idbGet('imageB')]);
  loadFromUrl(blobA ? URL.createObjectURL(blobA) : './defaults/lofoten_A.jpg', 'A');
  loadFromUrl(blobB ? URL.createObjectURL(blobB) : './defaults/lofoten_B.jpg', 'B');
})();

document.getElementById('swap').addEventListener('click', () => {
  [state.imgA, state.imgB] = [state.imgB, state.imgA];
  if (state.imgA) uploadImage(texA, state.imgA);
  if (state.imgB) uploadImage(texB, state.imgB);
  advec.needsReset = true;
  recomputeSeedPositions();
  const sA = document.querySelector('.slot[data-slot="A"] img');
  const sB = document.querySelector('.slot[data-slot="B"] img');
  if (sA && sB)        { const tmp = sA.src; sA.src = sB.src; sB.src = tmp; }
  else if (sA && !sB)  { const url = sA.src; sA.remove(); updateSlotPreview('B', url); }
  else if (sB && !sA)  { const url = sB.src; sB.remove(); updateSlotPreview('A', url); }
});

document.getElementById('clear').addEventListener('click', async () => {
  state.imgA = null; state.imgB = null;
  document.querySelectorAll('.slot').forEach(s => {
    s.querySelector('img')?.remove();
    if (!s.querySelector('.placeholder')) {
      const ph = document.createElement('span');
      ph.className = 'placeholder';
      ph.textContent = 'click / drop';
      s.appendChild(ph);
    }
  });
  canvas.classList.add('empty');
  dropHint.classList.remove('hidden');
  await idbClearAll();
});

document.getElementById('reset').addEventListener('click', async () => {
  await idbClearAll();
  loadFromUrl('./defaults/lofoten_A.jpg', 'A');
  loadFromUrl('./defaults/lofoten_B.jpg', 'B');
});

// drag-and-drop — only the A and B slots are drop targets. Dragging an image
// elsewhere does nothing. We still need to prevent the browser's default file
// open behaviour at the window level, otherwise dropping outside a slot would
// navigate away from the page.
document.querySelectorAll('.slot').forEach(s => {
  s.addEventListener('dragenter', e => { e.preventDefault(); e.stopPropagation(); s.classList.add('drop-target'); });
  s.addEventListener('dragover',  e => { e.preventDefault(); e.stopPropagation(); e.dataTransfer.dropEffect = 'copy'; });
  s.addEventListener('dragleave', e => { e.preventDefault(); e.stopPropagation(); s.classList.remove('drop-target'); });
  s.addEventListener('drop', e => {
    e.preventDefault(); e.stopPropagation();
    s.classList.remove('drop-target');
    const files = [...e.dataTransfer.files].filter(f => f.type.startsWith('image/'));
    if (files.length === 0) return;
    loadFile(files[0], s.dataset.slot);
  });
});
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop',     e => e.preventDefault());

window.addEventListener('resize', resizeCanvas);

// ----- panel minimize -----
const togglePanelBtn = document.getElementById('toggle-panel');
function setMinimized(min) {
  document.body.classList.toggle('minimized', min);
  togglePanelBtn.textContent = min ? '›' : '‹';
  togglePanelBtn.title = min ? 'Show controls (Tab)' : 'Hide controls (Tab)';
  // recompute canvas display size for the new layout once the CSS transition starts
  requestAnimationFrame(() => requestAnimationFrame(resizeCanvas));
  setTimeout(resizeCanvas, 250);
}
togglePanelBtn.addEventListener('click', () => {
  setMinimized(!document.body.classList.contains('minimized'));
});

// ----- Tweakpane -----
const pane = new Pane({ container: document.getElementById('tp-host') });

// Adds the `title` attribute (native hover tooltip) to a binding's row element.
function tip(binding, text) { if (binding && binding.element) binding.element.title = text; return binding; }

// PLAYBACK ------------------------------------------------------------------
const fPlay = pane.addFolder({ title: 'Playback', expanded: true });
const bT = fPlay.addBinding(state, 't', { min: 0, max: 1, step: 0.001, label: 'progress' });
tip(bT, 'Scrub the transition. Updates as the morph plays back.');
tip(fPlay.addBinding(state, 'duration', { min: 0.5, max: 30, step: 0.1 }),
    'Total duration of the morph in seconds.');

const btnPlay = fPlay.addButton({ title: 'Pause' });
const btnRestart = fPlay.addButton({ title: 'Restart from start' });
const btnLoop = fPlay.addButton({ title: 'Loop: on' });

btnPlay.on('click', () => {
  if (!state.imgA || !state.imgB) return;
  if (state.playing) {
    state.playing = false; btnPlay.title = 'Play';
  } else {
    state.playing = true;
    const consumed = state.reverse ? (1 - state.t) : state.t;
    state.startTime = performance.now() - consumed * state.duration * 1000;
    btnPlay.title = 'Pause';
  }
});
btnRestart.on('click', () => {
  state.t = 0;
  state.reverse = false;
  state.playing = true;
  state.startTime = performance.now();
  btnPlay.title = 'Pause';
  advec.needsReset = true; // will warm up the sim from A
});
btnLoop.on('click', () => {
  state.loop = !state.loop;
  btnLoop.title = 'Loop: ' + (state.loop ? 'on' : 'off');
});

// WATERCOLOR MODE -----------------------------------------------------------
const fWater = pane.addFolder({ title: 'Watercolor', expanded: true });
tip(
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
      'wet bleed':              8,
      'pigment run':            9,
      'wet advection':          10,
      'gravity advection':      11,
      'curl-noise eddies':      12,
      'brush-channel advection': 13,
      'seed-point injection':   14,
    },
  }).on('change', () => { updateModeFolders(); advec.needsReset = true; }),
  'Watercolor behaviour layered on top of the dissolve.'
);

// Per-mode default values for the "Reset defaults" button.
const MODE_DEFAULTS = {
  1:  { rimWidth: 0.12, rimDark: 0.6 },
  2:  { paperAngle: 0, paperAniso: 4, paperGranulation: 0.5 },
  3:  { bloomCount: 8, bloomRim: 0.6, bloomRate: 0.55 },
  4:  { diffStrength: 0.55, diffRadius: 0.45 },
  5:  { sedBands: 6, sedSoftness: 0.35, sedDirection: 0, sedSource: 0 },
  6:  { saltDensity: 0.0, saltContrast: 0.55, saltSource: 1, saltBias: 0.6, saltImage: 2 },
  7:  { irisFocusX: 0.5, irisFocusY: 0.5, irisJitter: 0.35 },
  8:  { bleedFinger: 0.5, bleedAmount: 0.45, bleedHalo: 0.5 },
  9:  { runGravity: 0.5, runDrip: 0.35 },
  10: { advecVisc: 0.55, advecRate: 0.18, advecSteps: 3 },
  11: { advecGravity: 0.6, advecGravBias: 0.5, advecGravAngle: 0, advecGravStreak: 0.4, advecGravLateral: 0.3 },
  12: { advecCurlStr: 0.5, advecCurlScale: 2.5 },
  13: { advecBrushFollow: 0.7 },
  14: { advecSeedCount: 5, advecSeedRadius: 0.45 },
};
function resetModeDefaults(modeId) {
  const d = MODE_DEFAULTS[modeId];
  if (!d) return;
  for (const [k, v] of Object.entries(d)) state[k] = v;
  if (modeId >= 10 && modeId <= 14) advec.needsReset = true;
  pane.refresh();
}
function addResetBtn(folder, modeId) {
  folder.addButton({ title: 'Reset defaults' }).on('click', () => resetModeDefaults(modeId));
}

const fRim    = fWater.addFolder({ title: 'Pigment rim',    expanded: true });
const fPaper  = fWater.addFolder({ title: 'Paper grain',    expanded: true });
const fBlooms = fWater.addFolder({ title: 'Backrun blooms', expanded: true });
const fDiff   = fWater.addFolder({ title: 'Wet diffusion',  expanded: true });
const fSed    = fWater.addFolder({ title: 'Tonal sediment', expanded: true });
const fSalt   = fWater.addFolder({ title: 'Salt',           expanded: true });
const fIris   = fWater.addFolder({ title: 'Iris',           expanded: true });
const fBleed  = fWater.addFolder({ title: 'Wet bleed',      expanded: true });
const fRun    = fWater.addFolder({ title: 'Pigment run',    expanded: true });
const fAdvec   = fWater.addFolder({ title: 'Wet advection',          expanded: true });
const fAdvecG  = fWater.addFolder({ title: 'Gravity advection',      expanded: true });
const fAdvecC  = fWater.addFolder({ title: 'Curl-noise eddies',      expanded: true });
const fAdvecB  = fWater.addFolder({ title: 'Brush-channel advection', expanded: true });
const fAdvecS  = fWater.addFolder({ title: 'Seed-point injection',   expanded: true });

tip(fRim.addBinding(state, 'rimWidth', { min: 0, max: 0.4, step: 0.005, label: 'rim width' }),
    'Thickness of the dark settling band at the wet front.');
tip(fRim.addBinding(state, 'rimDark', { min: 0, max: 1, step: 0.01, label: 'rim dark' }),
    'How dark the settled pigment band is.');
addResetBtn(fRim, 1);

tip(fPaper.addBinding(state, 'paperAngle', { min: 0, max: 1, step: 0.005, label: 'fiber angle' }),
    'Paper-fiber direction: 0 = horizontal grain, 0.5 = vertical.');
tip(fPaper.addBinding(state, 'paperAniso', { min: 1, max: 10, step: 0.1, label: 'anisotropy' }),
    'How elongated the fibers are. Higher = longer streaks along the grain.');
tip(fPaper.addBinding(state, 'paperGranulation', { min: 0, max: 1, step: 0.01, label: 'granulation' }),
    'Fine speckle of pigment grains settling in the paper tooth.');
addResetBtn(fPaper, 2);

tip(fBlooms.addBinding(state, 'bloomCount', { min: 1, max: 24, step: 1, label: 'count' }),
    'Number of bloom seed points growing across the canvas.');
tip(fBlooms.addBinding(state, 'bloomRate', { min: 0.1, max: 2, step: 0.01, label: 'growth rate' }),
    'How fast each bloom expands. Lower = slower, larger fronts overlap less.');
tip(fBlooms.addBinding(state, 'bloomRim', { min: 0, max: 1, step: 0.01, label: 'rim dark' }),
    'Darkness of the settling rim at each bloom front.');
addResetBtn(fBlooms, 3);

tip(fDiff.addBinding(state, 'diffStrength', { min: 0, max: 1, step: 0.01, label: 'strength' }),
    'How strongly B\'s pigment bleeds outward into A\'s still-dry region.');
tip(fDiff.addBinding(state, 'diffRadius', { min: 0, max: 1, step: 0.01, label: 'radius' }),
    'How far the wet pigment reaches when bleeding.');
addResetBtn(fDiff, 4);

tip(fSed.addBinding(state, 'sedSource', {
  label: 'decompose by',
  options: {
    'luminance':    0,
    'saturation':   1,
    'hue':          2,
    'detail':       3,
    'temperature':  4,
  },
}), 'Which image property is split into bands. Each band reveals at a distinct point in t.');
tip(fSed.addBinding(state, 'sedBands', { min: 1, max: 16, step: 1, label: 'bands' }),
    'Number of bands the property is split into. Fewer = more obvious stepping.');
tip(fSed.addBinding(state, 'sedSoftness', { min: 0, max: 1, step: 0.01, label: 'softness' }),
    'Softness between bands. 0 = crisp steps, 1 = continuous gradient.');
tip(fSed.addBinding(state, 'sedDirection', {
  label: 'order',
  options: { 'low → high': 0, 'high → low': 1 },
}), 'Reveal direction. e.g. for luminance: low → high means shadows first.');
addResetBtn(fSed, 5);

tip(fSalt.addBinding(state, 'saltDensity', { min: 0, max: 1, step: 0.01, label: 'grain' }),
    'Grain size of the salt texture. Low = few large blobs, high = fine speckle. Controls the look of the dissolve regardless of whether the image is driving it.');
tip(fSalt.addBinding(state, 'saltContrast', { min: 0, max: 1, step: 0.01, label: 'contrast' }),
    'How sharply each crystal stands out against the surrounding paper.');
tip(fSalt.addBinding(state, 'saltSource', {
  label: 'reveal from',
  options: {
    'random (none)':   0,
    'light areas':     1,
    'dark areas':      2,
    'coloured areas':  3,
    'edge detail':     4,
  },
}), 'Bias which pixels open earliest based on this property of the source image.');
tip(fSalt.addBinding(state, 'saltImage', {
  label: 'sample',
  options: { 'A': 0, 'B': 1, 'both': 2 },
}), 'Which painting the bias reads from.');
tip(fSalt.addBinding(state, 'saltBias', { min: 0, max: 1, step: 0.01, label: 'bias amount' }),
    'How strongly the image steers the reveal. 0 = pure random salt; 1 = image fully drives timing while grain still controls the visible texture.');
addResetBtn(fSalt, 6);

tip(fIris.addBinding(state, 'irisFocusX', { min: 0, max: 1, step: 0.005, label: 'focus x' }),
    'Horizontal position of the iris focal point (0 = left, 1 = right).');
tip(fIris.addBinding(state, 'irisFocusY', { min: 0, max: 1, step: 0.005, label: 'focus y' }),
    'Vertical position of the iris focal point (0 = bottom, 1 = top).');
tip(fIris.addBinding(state, 'irisJitter', { min: 0, max: 1, step: 0.01, label: 'jitter' }),
    'Irregularity of the iris front. Higher = breathes more instead of expanding as a perfect circle.');
addResetBtn(fIris, 7);

tip(fBleed.addBinding(state, 'bleedFinger', { min: 0, max: 1, step: 0.01, label: 'finger' }),
    'Anisotropy / frequency of the fingering protrusions. Low = chunky fronts, high = fine threads.');
tip(fBleed.addBinding(state, 'bleedAmount', { min: 0, max: 1, step: 0.01, label: 'amount' }),
    'How strongly the wet front breaks into finger-like protrusions.');
tip(fBleed.addBinding(state, 'bleedHalo', { min: 0, max: 1, step: 0.01, label: 'wet halo' }),
    'Saturated wet halo just inside the front — the sheen of recently-soaked pigment.');
addResetBtn(fBleed, 8);

tip(fRun.addBinding(state, 'runGravity', { min: 0, max: 1, step: 0.01, label: 'gravity' }),
    '0 = A\'s shadows alone determine reveal order; 1 = pure top-to-bottom run (pigment flows downhill).');
tip(fRun.addBinding(state, 'runDrip', { min: 0, max: 1, step: 0.01, label: 'drip' }),
    'Vertical trail of B\'s pigment carried down from just above the front — like wet paint running.');
addResetBtn(fRun, 9);

tip(fAdvec.addBinding(state, 'advecVisc', { min: 0, max: 1, step: 0.01, label: 'viscosity' }),
    'How much each pixel averages with its neighbours per step. Higher = more spread, less crisp.');
tip(fAdvec.addBinding(state, 'advecRate', { min: 0, max: 1, step: 0.01, label: 'mixing rate' }),
    'Per-step strength of pulling the state toward B in the revealed region. Lower = slower, more painterly accumulation.');
tip(fAdvec.addBinding(state, 'advecSteps', { min: 1, max: 8, step: 1, label: 'steps / frame' }),
    'Diffusion iterations per rendered frame. More steps = smoother bleed but more GPU cost.');
addResetBtn(fAdvec, 10);
fAdvec.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

// Gravity variant
tip(fAdvecG.addBinding(state, 'advecGravAngle', { min: 0, max: 1, step: 0.005, label: 'flow angle' }),
    'Direction pigment falls. 0 = down, 0.25 = right, 0.5 = up, 0.75 = left.');
tip(fAdvecG.addBinding(state, 'advecGravity', { min: 0, max: 1, step: 0.01, label: 'gravity' }),
    'How strongly the upstream neighbour contributes vs the downstream one. Higher = harder fall.');
tip(fAdvecG.addBinding(state, 'advecGravStreak', { min: 0, max: 1, step: 0.01, label: 'streak' }),
    'Anisotropy along the flow axis. Higher = longer trailing streaks of pigment.');
tip(fAdvecG.addBinding(state, 'advecGravLateral', { min: 0, max: 1, step: 0.01, label: 'lateral spread' }),
    'Perpendicular bleed. Low = narrow streams, high = wider gentle washes.');
tip(fAdvecG.addBinding(state, 'advecGravBias', { min: 0, max: 1, step: 0.01, label: 'shadow ↔ flow' }),
    '0 = A\'s shadows attract pigment first; 1 = pure spatial flow regardless of image content.');
addResetBtn(fAdvecG, 11);
fAdvecG.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

// Curl variant
tip(fAdvecC.addBinding(state, 'advecCurlStr', { min: 0, max: 1, step: 0.01, label: 'eddy strength' }),
    'How strongly pigment is carried along the curl velocity field. Higher = more swirling.');
tip(fAdvecC.addBinding(state, 'advecCurlScale', { min: 0.5, max: 8, step: 0.1, label: 'eddy scale' }),
    'Scale of the eddies. Low = a few large swirls, high = many small turbulent ones.');
addResetBtn(fAdvecC, 12);
fAdvecC.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

// Brush-channel variant
tip(fAdvecB.addBinding(state, 'advecBrushFollow', { min: 0, max: 1, step: 0.01, label: 'follow strokes' }),
    'How strongly diffusion is aligned with A\'s local brush direction. 0 = isotropic, 1 = pigment travels almost exclusively along strokes.');
addResetBtn(fAdvecB, 13);
fAdvecB.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

// Seed-point variant
tip(fAdvecS.addBinding(state, 'advecSeedSource', {
  label: 'place at',
  options: {
    'random':         0,
    'light areas':    1,
    'dark areas':     2,
    'coloured areas': 3,
    'edge detail':    4,
  },
}), 'Where seeds are placed: random, or biased to a property of the source images so the bleed grows out of the painting\'s own structure.')
  .on('change', () => recomputeSeedPositions());
tip(fAdvecS.addBinding(state, 'advecSeedImage', {
  label: 'sample',
  options: { 'A': 0, 'B': 1, 'both': 2 },
}), 'Which painting the seed placement reads from.')
  .on('change', () => recomputeSeedPositions());
tip(fAdvecS.addBinding(state, 'advecSeedCount', { min: 1, max: 16, step: 1, label: 'seed count' }),
    'How many drop-points B\'s pigment seeps from.');
tip(fAdvecS.addBinding(state, 'advecSeedRadius', { min: 0.1, max: 1, step: 0.01, label: 'reach' }),
    'How far each seed\'s wash spreads through the canvas (relative to image size).');
addResetBtn(fAdvecS, 14);
fAdvecS.addButton({ title: 'Reset simulation' }).on('click', () => { advec.needsReset = true; });

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
}
updateModeFolders();

// DISSOLVE ------------------------------------------------------------------
const fDis = pane.addFolder({ title: 'Dissolve', expanded: true });
tip(fDis.addBinding(state, 'organic',   { min: 0, max: 1, step: 0.01 }),
    'Mask source: 0 = pure low-freq noise, 1 = the painting\'s own luminance.');
tip(fDis.addBinding(state, 'edges',     { min: -1, max: 1, step: 0.01 }),
    'Brush-stroke detail leads (+) or trails (−) the reveal.');
tip(fDis.addBinding(state, 'spread',    { min: 0, max: 1, step: 0.01 }),
    'Per-pixel reveal softness. Low = crisp dissolve, high = wide gentle fade.');
tip(fDis.addBinding(state, 'maskScale', { min: 0.3, max: 4, step: 0.05, label: 'mask scale' }),
    'Scale of the noise mask. Small = big blobs, large = fine grain.');
fDis.addBinding(state, 'curve', {
  options: { 'ease-in-out': 1, 'linear': 0, 'ease-in': 2, 'ease-out': 3 },
});
tip(fDis.addBinding(state, 'seed', { min: 0, max: 999, step: 1 }),
    'Random seed — shifts the mask and bloom positions.');

// ATMOSPHERE (collapsed by default since defaults are all zero) --------------
const fAtm = pane.addFolder({ title: 'Atmosphere', expanded: false });
tip(fAtm.addBinding(state, 'softness', { min: 0, max: 1, step: 0.01 }),
    'Defocus blur radius at the midpoint of the transition.');
tip(fAtm.addBinding(state, 'glow',     { min: 0, max: 1, step: 0.01 }),
    'Local exposure lift on bright pixels at peak.');
tip(fAtm.addBinding(state, 'bloom',    { min: 0, max: 1, step: 0.01 }),
    'Wide soft halo from bright strokes, screen-blended at peak.');
tip(fAtm.addBinding(state, 'warmth',   { min: -1, max: 1, step: 0.01 }),
    'Colour temperature shift at peak: warm (+) ↔ cool (−).');
tip(fAtm.addBinding(state, 'vignette', { min: 0, max: 1, step: 0.01 }),
    'Radial darken that breathes in at peak.');

// IMAGE FRAMING (zoom + pan per image, collapsed by default) ----------------
const fImgA = pane.addFolder({ title: 'Image A — framing', expanded: false });
tip(fImgA.addBinding(state, 'zoomA', { min: 0.5, max: 4, step: 0.01, label: 'zoom' }),
    'Scale image A in/out of the canvas. >1 crops to the centre; <1 lets it shrink with margins.');
tip(fImgA.addBinding(state, 'panAx', { min: -1, max: 1, step: 0.005, label: 'pan x' }),
    'Horizontal pan of A. Negative = look left of centre; positive = right.');
tip(fImgA.addBinding(state, 'panAy', { min: -1, max: 1, step: 0.005, label: 'pan y' }),
    'Vertical pan of A. Negative = look down; positive = up.');
fImgA.addButton({ title: 'Reset A framing' }).on('click', () => {
  state.zoomA = 1; state.panAx = 0; state.panAy = 0; pane.refresh();
});

const fImgB = pane.addFolder({ title: 'Image B — framing', expanded: false });
tip(fImgB.addBinding(state, 'zoomB', { min: 0.5, max: 4, step: 0.01, label: 'zoom' }),
    'Scale image B in/out of the canvas. >1 crops to the centre; <1 lets it shrink with margins.');
tip(fImgB.addBinding(state, 'panBx', { min: -1, max: 1, step: 0.005, label: 'pan x' }),
    'Horizontal pan of B. Negative = look left of centre; positive = right.');
tip(fImgB.addBinding(state, 'panBy', { min: -1, max: 1, step: 0.005, label: 'pan y' }),
    'Vertical pan of B. Negative = look down; positive = up.');
fImgB.addButton({ title: 'Reset B framing' }).on('click', () => {
  state.zoomB = 1; state.panBx = 0; state.panBy = 0; pane.refresh();
});

// STYLE (collapsed) ---------------------------------------------------------
const fStyle = pane.addFolder({ title: 'Style', expanded: false });
fStyle.addBinding(state, 'fit', {
  options: { 'cover (crop)': 'cover', 'contain (letterbox)': 'contain', 'stretch': 'stretch' },
});
fStyle.addBinding(state, 'bg', { view: 'color' });

// EXPORT --------------------------------------------------------------------
const fExp = pane.addFolder({ title: 'Export', expanded: true });
fExp.addBinding(state, 'exportFps', {
  label: 'fps',
  options: { '24 fps': 24, '30 fps': 30, '60 fps': 60 },
});
fExp.addBinding(state, 'exportSizeMode', {
  label: 'size',
  options: {
    'source (full res)': 'src',
    '5120 wide':         '5120',
    '3840 wide':         '3840',
    '2560 wide':         '2560',
    '1920 wide':         '1920',
    '1280 wide':         '1280',
    '960 wide':          '960',
  },
});
const padPresetsV1 = { _v: 0 };
const bPadPresetV1 = fExp.addBinding(padPresetsV1, '_v', {
  label: 'pad preset',
  options: {
    'none':                  0,
    'half (0.5)':            0.5,
    'full (1.0)':            1.0,
    'Elverket panorama':     1.416,
    'double (2.0)':          2.0,
  },
});
bPadPresetV1.on('change', e => {
  state.exportPadBottom = e.value;
  pane.refresh();
});
fExp.addBinding(state, 'exportPadBottom', { min: 0, max: 3, step: 0.001, label: 'pad below (× h)' });
const btnRecord = fExp.addButton({ title: 'Record video' });
btnRecord.on('click', () => startRecording());
const btnBatch = fExp.addButton({ title: 'Batch record presets (15 s each)' });
btnBatch.on('click', batchRecordPresets);

// PRESETS -------------------------------------------------------------------
// Snapshot only the fields that describe the look; skip transient ones (t, imgA/B, etc).
const PRESET_KEYS = [
  'duration', 'mode', 'curve', 'seed',
  'rimWidth', 'rimDark',
  'paperAngle', 'paperAniso', 'paperGranulation',
  'bloomCount', 'bloomRim', 'bloomRate',
  'diffStrength', 'diffRadius',
  'sedBands', 'sedSoftness', 'sedDirection', 'sedSource',
  'saltDensity', 'saltContrast', 'saltSource', 'saltBias', 'saltImage',
  'irisFocusX', 'irisFocusY', 'irisJitter',
  'bleedFinger', 'bleedAmount', 'bleedHalo',
  'runGravity', 'runDrip',
  'advecVisc', 'advecRate', 'advecSteps',
  'advecGravity', 'advecGravBias', 'advecGravAngle', 'advecGravStreak', 'advecGravLateral',
  'advecCurlStr', 'advecCurlScale',
  'advecBrushFollow',
  'advecSeedCount', 'advecSeedRadius', 'advecSeedSource', 'advecSeedImage',
  'organic', 'edges', 'spread', 'maskScale',
  'softness', 'glow', 'bloom', 'warmth', 'vignette',
  'zoomA', 'panAx', 'panAy', 'zoomB', 'panBx', 'panBy',
];

const FACTORY_PRESETS = {
  'Smooth dreamy': {
    duration: 5, mode: 0, curve: 1, seed: 42,
    organic: 0.65, edges: 0.25, spread: 0.55, maskScale: 0.9,
    softness: 0.0, glow: 0.45, bloom: 0.4, warmth: 0.1, vignette: 0.2,
  },
  'Paper grain — cold press': {
    duration: 7, mode: 2, curve: 1, seed: 42,
    paperAngle: 0, paperAniso: 6, paperGranulation: 0.7,
    organic: 0.3, edges: 0.15, spread: 0.35, maskScale: 0.9,
    softness: 0.25, glow: 0.35, bloom: 0.3, warmth: 0.08, vignette: 0.15,
  },
  'Paper grain — vertical wash': {
    duration: 7, mode: 2, curve: 0, seed: 88,
    paperAngle: 0.5, paperAniso: 7, paperGranulation: 0.5,
    organic: 0.4, edges: 0.1, spread: 0.5, maskScale: 1.1,
    softness: 0.3, glow: 0.4, bloom: 0.35, warmth: 0.0, vignette: 0.18,
  },
  'Backruns — dramatic': {
    duration: 12, mode: 3, curve: 2, seed: 12,
    bloomCount: 4, bloomRate: 0.45, bloomRim: 0.75,
    organic: 0.5, edges: 0.2, spread: 0.5, maskScale: 0.9,
    softness: 0.35, glow: 0.5, bloom: 0.45, warmth: 0.05, vignette: 0.25,
  },
  'Backruns — gentle blossoms': {
    duration: 9, mode: 3, curve: 1, seed: 7,
    bloomCount: 14, bloomRate: 0.9, bloomRim: 0.35,
    organic: 0.4, edges: 0.1, spread: 0.6, maskScale: 0.8,
    softness: 0.3, glow: 0.4, bloom: 0.5, warmth: 0.1, vignette: 0.15,
  },
  'Wet front rim': {
    duration: 5, mode: 1, curve: 1, seed: 42,
    rimWidth: 0.15, rimDark: 0.7,
    organic: 0.6, edges: 0.35, spread: 0.45, maskScale: 1.0,
    softness: 0.3, glow: 0.45, bloom: 0.35, warmth: 0.08, vignette: 0.2,
  },
  'Wet bleed (diffusion)': {
    duration: 8, mode: 4, curve: 1, seed: 23,
    diffStrength: 0.8, diffRadius: 0.6,
    organic: 0.55, edges: 0.2, spread: 0.55, maskScale: 0.9,
    softness: 0.35, glow: 0.45, bloom: 0.45, warmth: 0.06, vignette: 0.2,
  },
};

const LS_KEY = 'transition-tool:presets';
function loadUserPresets()      { try { return JSON.parse(localStorage.getItem(LS_KEY)) || {}; } catch { return {}; } }
function saveUserPresetsToLS(o) { localStorage.setItem(LS_KEY, JSON.stringify(o)); }

const presetUI = { current: '', newName: '' };
const fPresets = pane.addFolder({ title: 'Presets', expanded: true });
let presetSelector;
let presetNameInput;

function buildOptions() {
  const opts = { '— select —': '' };
  for (const k of Object.keys(FACTORY_PRESETS)) opts['★ ' + k] = 'factory:' + k;
  const user = loadUserPresets();
  for (const k of Object.keys(user))            opts['user · ' + k] = 'user:' + k;
  return opts;
}

function rebuildPresetsFolder() {
  while (fPresets.children.length) fPresets.children[0].dispose();
  presetUI.current = '';
  presetSelector = fPresets.addBinding(presetUI, 'current', { label: 'load', options: buildOptions() });
  tip(presetSelector, 'Pick a preset to apply it to the current settings.');
  presetSelector.on('change', e => { if (e.value) applyPreset(e.value); });
  presetNameInput = fPresets.addBinding(presetUI, 'newName', { label: 'name' });
  tip(presetNameInput, 'Name for the new preset to save.');
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
  pane.refresh();
  updateModeFolders();
}

// Keyboard shortcuts (skip when typing in pane inputs)
window.addEventListener('keydown', e => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT' || e.target.tagName === 'TEXTAREA') return;
  if (e.key === ' ')           { e.preventDefault(); btnPlay.element.querySelector('button').click(); }
  if (e.key === 'Tab')         { e.preventDefault(); setMinimized(!document.body.classList.contains('minimized')); }
  if (e.key === 'ArrowLeft')   { state.t = Math.max(0, state.t - 0.02); pane.refresh(); }
  if (e.key === 'ArrowRight')  { state.t = Math.min(1, state.t + 0.02); pane.refresh(); }
});

// ============================================================================
// Recording
// ============================================================================

const MODE_NAMES = {
  0: 'off', 1: 'rim', 2: 'paper', 3: 'blooms', 4: 'diffusion',
  5: 'sediment', 6: 'salt', 7: 'iris',
  8: 'wet-bleed', 9: 'pigment-run',
  10: 'advec', 11: 'advec-gravity', 12: 'advec-curl',
  13: 'advec-brush', 14: 'advec-seed',
};
const SED_SOURCE_NAMES = ['luma', 'sat', 'hue', 'detail', 'temp'];
const SALT_SOURCE_NAMES = ['random', 'light', 'dark', 'col', 'edge'];

function fixed(v, n = 2) { return (Math.round(v * Math.pow(10, n)) / Math.pow(10, n)).toString(); }

// Build a filename that embeds the current mode and its most relevant params.
function makeFilename() {
  const m = state.mode;
  const parts = [MODE_NAMES[m] || `mode${m}`];
  if      (m === 1)  parts.push(`rimW=${fixed(state.rimWidth)}`, `dark=${fixed(state.rimDark)}`);
  else if (m === 2)  parts.push(`ang=${fixed(state.paperAngle)}`, `aniso=${fixed(state.paperAniso, 1)}`, `gran=${fixed(state.paperGranulation)}`);
  else if (m === 3)  parts.push(`n=${state.bloomCount}`, `rate=${fixed(state.bloomRate)}`, `rim=${fixed(state.bloomRim)}`);
  else if (m === 4)  parts.push(`str=${fixed(state.diffStrength)}`, `r=${fixed(state.diffRadius)}`);
  else if (m === 5)  parts.push(`by=${SED_SOURCE_NAMES[state.sedSource] || 'luma'}`, `bands=${state.sedBands}`, `soft=${fixed(state.sedSoftness)}`);
  else if (m === 6)  parts.push(`from=${SALT_SOURCE_NAMES[state.saltSource] || 'random'}`, `grain=${fixed(state.saltDensity)}`, `bias=${fixed(state.saltBias)}`);
  else if (m === 7)  parts.push(`focus=${fixed(state.irisFocusX)}-${fixed(state.irisFocusY)}`, `jit=${fixed(state.irisJitter)}`);
  else if (m === 8)  parts.push(`fing=${fixed(state.bleedFinger)}`, `amt=${fixed(state.bleedAmount)}`, `halo=${fixed(state.bleedHalo)}`);
  else if (m === 9)  parts.push(`grav=${fixed(state.runGravity)}`, `drip=${fixed(state.runDrip)}`);
  else if (m === 10) parts.push(`visc=${fixed(state.advecVisc)}`, `rate=${fixed(state.advecRate)}`);
  else if (m === 11) parts.push(`grav=${fixed(state.advecGravity)}`, `ang=${fixed(state.advecGravAngle)}`, `streak=${fixed(state.advecGravStreak)}`);
  else if (m === 12) parts.push(`curl=${fixed(state.advecCurlStr)}`, `scale=${fixed(state.advecCurlScale, 1)}`);
  else if (m === 13) parts.push(`follow=${fixed(state.advecBrushFollow)}`);
  else if (m === 14) parts.push(`from=${SALT_SOURCE_NAMES[state.advecSeedSource] || 'random'}`, `n=${state.advecSeedCount}`, `r=${fixed(state.advecSeedRadius)}`);
  parts.push(`${Math.round(state.duration)}s`);
  if (state.exportPadBottom > 0) parts.push(`pad=${fixed(state.exportPadBottom)}`);
  return `morph__${parts.join('__')}`;  // extension added by recorder based on codec
}

// Prefer HEVC (H.265) for the wider encoder dimension headroom (~7680 vs
// ~3840 for H.264), then mp4/H.264, then webm. Falls back gracefully.
const RECORDER_MIMES = [
  'video/mp4;codecs=hvc1.1.6.L120.B0',
  'video/mp4;codecs=hev1.1.6.L120.B0',
  'video/mp4;codecs=hvc1',
  'video/mp4;codecs=hev1',
  'video/mp4;codecs=avc1.42E01E',
  'video/mp4;codecs=avc1',
  'video/mp4',
  'video/webm;codecs=vp9',
  'video/webm;codecs=vp8',
  'video/webm',
];
function pickRecorderMime() {
  for (const m of RECORDER_MIMES) {
    if (MediaRecorder.isTypeSupported(m)) return m;
  }
  return 'video/webm';
}
function encoderMaxDim(mime) {
  if (/hev|hvc/.test(mime)) return 7680;
  if (/vp9/.test(mime))     return 7680;
  if (/avc|h264/.test(mime)) return 3840;
  return 3840;
}
function mimeToExt(mime) { return mime.startsWith('video/mp4') ? 'mp4' : 'webm'; }

async function startRecording(opts = {}) {
  if (state.recording) return;
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
  // Hardware encoders fail silently above their max dimension. Cap the largest
  // axis and scale everything proportionally so wide panoramas still record.
  const mime = pickRecorderMime();
  const ENCODER_MAX = encoderMaxDim(mime);
  let scale = 1;
  const maxDim = Math.max(recW, recH + padPx0);
  if (maxDim > ENCODER_MAX) {
    scale = ENCODER_MAX / maxDim;
    recW = Math.round(recW * scale);
    recH = Math.round(recH * scale);
  }
  const padPx = Math.round(recH * state.exportPadBottom);
  const totalH = (recH + padPx) + ((recH + padPx) % 2);
  const offW = recW + (recW % 2);

  const off = document.createElement('canvas');
  off.width = offW; off.height = totalH;
  const offCtx = off.getContext('2d');
  offCtx.fillStyle = '#000';
  offCtx.fillRect(0, 0, off.width, off.height);
  console.log(`[record] codec ${mime}, output ${offW}×${totalH}` + (scale < 1 ? `  (scaled from ${canvas.width}×${canvas.height} for encoder limit)` : ''));

  const stream = off.captureStream(fps);
  const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 12_000_000 });
  const chunks = [];
  rec.ondataavailable = e => { if (e.data.size > 0) chunks.push(e.data); };

  state.recording = true;
  const originalTitle = btnRecord.title;
  btnRecord.title = (opts.statusPrefix ? `${opts.statusPrefix} ` : '')
    + (scale < 1 ? `Scaled to ${offW}×${totalH}. Recording…` : 'Recording…');

  const totalFrames = Math.max(2, Math.round(state.duration * fps));
  rec.start();

  const wasPlaying = state.playing;
  state.playing = false;
  const prevT = state.t;

  for (let i = 0; i < totalFrames; i++) {
    state.t = i / (totalFrames - 1);
    drawOnce();
    offCtx.drawImage(canvas, 0, 0, recW, recH);
    btnRecord.title = `${opts.statusPrefix || ''} frame ${i + 1} / ${totalFrames}`;
    await new Promise(r => setTimeout(r, 1000 / fps));
  }

  rec.stop();
  await new Promise(r => { rec.onstop = r; });

  const blob = new Blob(chunks, { type: mime });
  state.recording = false;
  state.t = prevT;
  state.playing = wasPlaying;

  if (blob.size < 1024) {
    btnRecord.title = 'FAILED — output too large for codec, try smaller size';
    setTimeout(() => { btnRecord.title = originalTitle; }, 4000);
    console.error('[record] empty blob — encoder likely rejected dimensions. Try a smaller export size.');
    return;
  }

  const url = URL.createObjectURL(blob);
  const ext = mimeToExt(mime);
  const base = opts.filename || makeFilename();
  const filename = /\.(mp4|webm)$/i.test(base) ? base : `${base}.${ext}`;
  const a = document.createElement('a');
  a.href = url; a.download = filename; a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);

  btnRecord.title = `saved (${(blob.size / 1024 / 1024).toFixed(1)} MB)`;
  if (!opts.keepTitle) setTimeout(() => { btnRecord.title = originalTitle; }, 2500);
}

// Iterate every factory preset, force 15 s duration, record each to disk with
// a mode-descriptive filename. Restores original state at the end.
async function batchRecordPresets() {
  if (state.recording) return;
  if (!state.imgA || !state.imgB) return;

  const before = snapshotState();
  const beforeDuration = state.duration;
  const names = Object.keys(FACTORY_PRESETS);
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    applyPreset('factory:' + name);
    state.duration = 15;
    pane.refresh();
    advec.needsReset = true;
    await new Promise(r => requestAnimationFrame(r));
    await new Promise(r => requestAnimationFrame(r));
    const safe = name.replace(/[^a-z0-9-]+/gi, '-').toLowerCase();
    const fname = `morph__${safe}__${makeFilename().replace(/^morph__/, '')}`;
    await startRecording({
      filename: fname,
      statusPrefix: `[${i + 1}/${names.length}]`,
      keepTitle: i < names.length - 1,
    });
    await new Promise(r => setTimeout(r, 500));
  }
  // restore
  for (const k of PRESET_KEYS) if (k in before) state[k] = before[k];
  state.duration = beforeDuration;
  pane.refresh();
  advec.needsReset = true;
  btnRecord.title = 'Record video';
}

// Synchronous draw (used by the recorder so we don't depend on rAF cadence)
function drawOnce() {
  gl.useProgram(prog);
  gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, texA);
  gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, texB);
  const cw = canvas.width, ch = canvas.height;
  const fA = composedFit('A', cw, ch);
  const fB = composedFit('B', cw, ch);
  gl.uniform2f(U.aspA, fA.sx, fA.sy); gl.uniform2f(U.offA, fA.ox, fA.oy);
  gl.uniform2f(U.aspB, fB.sx, fB.sy); gl.uniform2f(U.offB, fB.ox, fB.oy);
  pushUniforms();
  gl.drawArrays(gl.TRIANGLES, 0, 6);
}


// Automation hook — small global so headless scripts can drive the tool
// without going through the Tweakpane DOM. Not used by the human UI.
// ----- Auto-persist all settings to localStorage -----
const SESSION_LS_KEY = 'transition-tool:session';
const PERSIST_KEYS = [
  ...PRESET_KEYS,
  'fit', 'bg',
  'exportFps', 'exportSizeMode', 'exportPadBottom',
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
    pane.refresh();
    updateModeFolders();
    if (state.mode >= 10 && state.mode <= 14) advec.needsReset = true;
  } catch {}
}
loadSession();
pane.on('change', () => saveSession());

window.__tool = { state, pane, advec, startRecording, snapshotState, applyPreset, FACTORY_PRESETS, makeFilename, recomputeSeedPositions };

