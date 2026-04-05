// ============================================================
// Lars Lerin — Particle Dissolution (WebGL2 + GLSL)
// ============================================================

const PARTICLE_COUNT = 80000;
const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 4.0;
const BLACK_TIME = 2.0;

// ---- State machine ----
const STATE = { HOLDING: 0, DISSOLVING: 1, BLACK: 2 };
let state = STATE.BLACK;
let stateTimer = 0;
let dissolve = 0;
let currentImageIndex = 0;
let images = [];

// ---- Canvas & GL ----
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
if (!gl) {
  document.body.textContent = 'WebGL2 not supported';
  throw new Error('WebGL2 not supported');
}

// ---- Fullscreen ----
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    canvas.requestFullscreen().catch(() => {});
  } else {
    document.exitFullscreen();
  }
}
canvas.addEventListener('click', toggleFullscreen);
window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') toggleFullscreen();
});

// ---- Resize ----
function resize() {
  canvas.width = window.innerWidth * devicePixelRatio;
  canvas.height = window.innerHeight * devicePixelRatio;
  gl.viewport(0, 0, canvas.width, canvas.height);
}
window.addEventListener('resize', resize);
resize();

// ============================================================
// GLSL Shaders
// ============================================================

const VERT = `#version 300 es
precision highp float;

in vec2 a_position;
in vec3 a_color;
in float a_size;
in vec3 a_noiseOffset;

uniform float u_time;
uniform float u_dissolve;
uniform vec2 u_resolution;

out vec3 v_color;
out float v_alpha;

// ---- Simplex 3D noise (Ashima / Stefan Gustavson) ----
vec3 mod289(vec3 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0/289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+10.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

  vec3 i  = floor(v + dot(v, C.yyy));
  vec3 x0 = v - i + dot(i, C.xxx);

  vec3 g = step(x0.yzx, x0.xyz);
  vec3 l = 1.0 - g;
  vec3 i1 = min(g.xyz, l.zxy);
  vec3 i2 = max(g.xyz, l.zxy);

  vec3 x1 = x0 - i1 + C.xxx;
  vec3 x2 = x0 - i2 + C.yyy;
  vec3 x3 = x0 - D.yyy;

  i = mod289(i);
  vec4 p = permute(permute(permute(
    i.z + vec4(0.0, i1.z, i2.z, 1.0))
  + i.y + vec4(0.0, i1.y, i2.y, 1.0))
  + i.x + vec4(0.0, i1.x, i2.x, 1.0));

  float n_ = 0.142857142857;
  vec3 ns = n_ * D.wyz - D.xzx;

  vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

  vec4 x_ = floor(j * ns.z);
  vec4 y_ = floor(j - 7.0 * x_);

  vec4 x = x_ * ns.x + ns.yyyy;
  vec4 y = y_ * ns.x + ns.yyyy;
  vec4 h = 1.0 - abs(x) - abs(y);

  vec4 b0 = vec4(x.xy, y.xy);
  vec4 b1 = vec4(x.zw, y.zw);

  vec4 s0 = floor(b0)*2.0 + 1.0;
  vec4 s1 = floor(b1)*2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw*sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw*sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m*m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

// ---- Curl-like offset from simplex noise ----
vec3 curlOffset(vec3 p) {
  float e = 0.01;
  float nx = snoise(p + vec3(e,0,0)) - snoise(p - vec3(e,0,0));
  float ny = snoise(p + vec3(0,e,0)) - snoise(p - vec3(0,e,0));
  float nz = snoise(p + vec3(0,0,e)) - snoise(p - vec3(0,0,e));
  return vec3(ny - nz, nz - nx, nx - ny) / (2.0 * e);
}

void main() {
  float d = u_dissolve;

  // Curl noise displacement
  vec3 noiseInput = a_noiseOffset * 2.0 + u_time * 0.3;
  vec3 curl = curlOffset(noiseInput);

  // Scale displacement with dissolve — gentle organic drift
  float driftStrength = d * d * 300.0;
  vec2 offset = curl.xy * driftStrength;

  // Gentle downward gravity pull
  offset.y += d * d * 120.0;

  // Add slight individual wiggle
  float wiggle = snoise(vec3(a_noiseOffset.xy * 5.0, u_time * 0.8));
  offset += vec2(wiggle, wiggle * 0.6) * d * 40.0;

  vec2 pos = a_position + offset;

  // Map to clip space
  vec2 ndc = (pos / u_resolution) * 2.0 - 1.0;
  ndc.y = -ndc.y;
  gl_Position = vec4(ndc, 0.0, 1.0);

  // Size: shrink as dissolve progresses (keep minimum so particles don't vanish abruptly)
  float sizeFade = max(1.0 - pow(d, 1.5), 0.15);
  gl_PointSize = a_size * sizeFade;

  v_color = a_color;

  // Alpha: slow start, then accelerating fade
  float alphaFade = 1.0 - pow(d, 2.0);
  v_alpha = alphaFade;
}
`;

const FRAG = `#version 300 es
precision highp float;

in vec3 v_color;
in float v_alpha;

out vec4 fragColor;

void main() {
  // Soft circle from point sprite
  vec2 crd = gl_PointCoord * 2.0 - 1.0;
  float dist = length(crd);
  float alpha = smoothstep(1.0, 0.4, dist);

  if (alpha < 0.01) discard;

  fragColor = vec4(v_color, alpha * v_alpha);
}
`;

// ============================================================
// Compile shaders & link program
// ============================================================

function compileShader(src, type) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    console.error(gl.getShaderInfoLog(s));
    gl.deleteShader(s);
    return null;
  }
  return s;
}

const vs = compileShader(VERT, gl.VERTEX_SHADER);
const fs = compileShader(FRAG, gl.FRAGMENT_SHADER);
const program = gl.createProgram();
gl.attachShader(program, vs);
gl.attachShader(program, fs);
gl.linkProgram(program);
if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
  console.error(gl.getProgramInfoLog(program));
}
gl.useProgram(program);

// ---- Locations ----
const loc = {
  a_position: gl.getAttribLocation(program, 'a_position'),
  a_color: gl.getAttribLocation(program, 'a_color'),
  a_size: gl.getAttribLocation(program, 'a_size'),
  a_noiseOffset: gl.getAttribLocation(program, 'a_noiseOffset'),
  u_time: gl.getUniformLocation(program, 'u_time'),
  u_dissolve: gl.getUniformLocation(program, 'u_dissolve'),
  u_resolution: gl.getUniformLocation(program, 'u_resolution'),
};

// ---- VAO + Buffers ----
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const posBuffer = gl.createBuffer();
const colorBuffer = gl.createBuffer();
const sizeBuffer = gl.createBuffer();
const noiseBuffer = gl.createBuffer();

function setupAttrib(buffer, location, size) {
  gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
  gl.enableVertexAttribArray(location);
  gl.vertexAttribPointer(location, size, gl.FLOAT, false, 0, 0);
}

setupAttrib(posBuffer, loc.a_position, 2);
setupAttrib(colorBuffer, loc.a_color, 3);
setupAttrib(sizeBuffer, loc.a_size, 1);
setupAttrib(noiseBuffer, loc.a_noiseOffset, 3);

gl.bindVertexArray(null);

// ---- Blending ----
gl.enable(gl.BLEND);
gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

// ============================================================
// Image sampling
// ============================================================

function sampleImage(img, count) {
  const w = img.width || img.naturalWidth;
  const h = img.height || img.naturalHeight;
  const offCanvas = document.createElement('canvas');
  offCanvas.width = w;
  offCanvas.height = h;
  const ctx = offCanvas.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  const data = ctx.getImageData(0, 0, w, h).data;

  // Build brightness-weighted cumulative distribution for rejection sampling
  const totalPixels = w * h;
  const brightness = new Float32Array(totalPixels);
  let brightnessSum = 0;
  for (let i = 0; i < totalPixels; i++) {
    const idx = i * 4;
    const b = (data[idx] * 0.299 + data[idx + 1] * 0.587 + data[idx + 2] * 0.114) / 255;
    // Weight: brighter = more particles, but give some floor so dark areas aren't empty
    const weight = 0.1 + b * 0.9;
    brightness[i] = weight;
    brightnessSum += weight;
  }

  // Normalize to CDF
  const cdf = new Float32Array(totalPixels);
  cdf[0] = brightness[0] / brightnessSum;
  for (let i = 1; i < totalPixels; i++) {
    cdf[i] = cdf[i - 1] + brightness[i] / brightnessSum;
  }

  // Scale image to fit viewport (centered)
  const vw = canvas.width;
  const vh = canvas.height;
  const scale = Math.min(vw / w, vh / h) * 0.9; // 90% of viewport
  const offsetX = (vw - w * scale) / 2;
  const offsetY = (vh - h * scale) / 2;

  const positions = new Float32Array(count * 2);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const noiseOffsets = new Float32Array(count * 3);

  // Compute local density map (for size variation — sparser areas get bigger particles)
  // We'll approximate by sampling and then just using brightness inversely for size
  for (let i = 0; i < count; i++) {
    // Binary search on CDF for weighted random sampling
    const r = Math.random();
    let lo = 0, hi = totalPixels - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < r) lo = mid + 1;
      else hi = mid;
    }

    const pixelIdx = lo;
    const px = pixelIdx % w;
    const py = (pixelIdx / w) | 0;
    const dataIdx = pixelIdx * 4;

    // Position (mapped to viewport)
    // Add small jitter within pixel
    const jx = (Math.random() - 0.5) * scale;
    const jy = (Math.random() - 0.5) * scale;
    positions[i * 2] = offsetX + px * scale + jx;
    positions[i * 2 + 1] = offsetY + py * scale + jy;

    // Color with slight warmth/saturation jitter
    let cr = data[dataIdx] / 255;
    let cg = data[dataIdx + 1] / 255;
    let cb = data[dataIdx + 2] / 255;
    // Warmth jitter
    cr += (Math.random() - 0.3) * 0.04;
    cg += (Math.random() - 0.5) * 0.02;
    cb += (Math.random() - 0.7) * 0.03;
    colors[i * 3] = Math.max(0, Math.min(1, cr));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, cg));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, cb));

    // Size: mix of fine dust and larger brushstroke particles
    // Sparser (darker) areas get slightly larger particles
    const b = brightness[pixelIdx];
    const baseSize = 2 + Math.random() * 4; // 2-6 fine dust
    const largeProbability = 0.15 + (1 - b) * 0.15; // darker areas more likely to get big particles
    const isLarge = Math.random() < largeProbability;
    const size = isLarge ? (8 + Math.random() * 12) : baseSize;
    sizes[i] = size;

    // Noise offset (random per particle for unique motion)
    noiseOffsets[i * 3] = Math.random() * 100;
    noiseOffsets[i * 3 + 1] = Math.random() * 100;
    noiseOffsets[i * 3 + 2] = Math.random() * 100;
  }

  return { positions, colors, sizes, noiseOffsets };
}

// ============================================================
// Upload particle data to GPU
// ============================================================

function uploadParticles(particleData) {
  gl.bindBuffer(gl.ARRAY_BUFFER, posBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, particleData.positions, gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, colorBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, particleData.colors, gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, sizeBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, particleData.sizes, gl.STATIC_DRAW);

  gl.bindBuffer(gl.ARRAY_BUFFER, noiseBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, particleData.noiseOffsets, gl.STATIC_DRAW);
}

// ============================================================
// Fallback watercolor images
// ============================================================

function generateWatercolorImage(seed) {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 768;
  const ctx = c.getContext('2d');

  // Lerin-inspired palettes: moody blues, earth tones, soft greens, warm grays
  const palettes = [
    ['#2b4570', '#5a7fa5', '#8fb3d4', '#c4a882', '#e8dcc8', '#3d6b5e', '#7a9e8e'],
    ['#4a3728', '#7d6b5d', '#b8a592', '#6b8f71', '#a3c4a0', '#d4c5a9', '#8b7355'],
    ['#1a3a4a', '#2d5f73', '#4a8fa8', '#d4a574', '#e8c9a0', '#f0e0c8', '#3a5a3a'],
  ];

  const palette = palettes[seed % palettes.length];

  // Base wash
  ctx.fillStyle = palette[palette.length - 1];
  ctx.fillRect(0, 0, c.width, c.height);

  // Layer overlapping radial gradients to simulate watercolor washes
  const rng = mulberry32(seed * 12345);

  for (let layer = 0; layer < 25; layer++) {
    const cx = rng() * c.width;
    const cy = rng() * c.height;
    const radius = 100 + rng() * 400;
    const color = palette[Math.floor(rng() * palette.length)];

    const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
    grad.addColorStop(0, hexToRgba(color, 0.15 + rng() * 0.25));
    grad.addColorStop(0.5, hexToRgba(color, 0.08 + rng() * 0.12));
    grad.addColorStop(1, hexToRgba(color, 0));

    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, c.width, c.height);
  }

  // Soft edge bleeds
  for (let i = 0; i < 12; i++) {
    const cx = rng() * c.width;
    const cy = rng() * c.height;
    const radius = 50 + rng() * 200;
    const color = palette[Math.floor(rng() * palette.length)];

    ctx.globalCompositeOperation = 'multiply';
    const grad = ctx.createRadialGradient(cx, cy, 0, cx + rng() * 60 - 30, cy + rng() * 60 - 30, radius);
    grad.addColorStop(0, hexToRgba(color, 0.2));
    grad.addColorStop(1, hexToRgba(color, 0));
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.globalCompositeOperation = 'source-over';
  }

  return c;
}

function hexToRgba(hex, alpha) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}

function mulberry32(a) {
  return function () {
    a |= 0; a = a + 0x6D2B79F5 | 0;
    let t = Math.imul(a ^ a >>> 15, 1 | a);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  };
}

// ============================================================
// Image loading
// ============================================================

async function loadImages() {
  const loaded = [];

  // Try loading from shared/images/
  const filenames = [
    'image1.jpg', 'image2.jpg', 'image3.jpg',
    'image1.png', 'image2.png', 'image3.png',
    'lerin1.jpg', 'lerin2.jpg', 'lerin3.jpg',
    'painting1.jpg', 'painting2.jpg', 'painting3.jpg',
  ];

  const tryLoad = (src) =>
    new Promise((resolve) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => resolve(img);
      img.onerror = () => resolve(null);
      img.src = src;
    });

  const results = await Promise.all(
    filenames.map((f) => tryLoad(`../shared/images/${f}`))
  );
  for (const img of results) {
    if (img) loaded.push(img);
  }

  if (loaded.length > 0) return loaded;

  // Fallback: generate 3 watercolor-style canvases
  console.log('No images found in shared/images/ — using generated watercolor fallbacks');
  for (let i = 0; i < 3; i++) {
    loaded.push(generateWatercolorImage(i));
  }
  return loaded;
}

// ============================================================
// Animation loop
// ============================================================

let lastTime = 0;
let globalTime = 0;
let particlesReady = false;

function transitionToNextImage() {
  const img = images[currentImageIndex % images.length];
  const data = sampleImage(img, PARTICLE_COUNT);
  uploadParticles(data);
  particlesReady = true;
  currentImageIndex++;
  state = STATE.HOLDING;
  stateTimer = 0;
  dissolve = 0;
}

function frame(now) {
  requestAnimationFrame(frame);

  const dt = Math.min((now - lastTime) / 1000, 0.1);
  lastTime = now;
  globalTime += dt;
  stateTimer += dt;

  // State machine
  switch (state) {
    case STATE.HOLDING:
      dissolve = 0;
      if (stateTimer >= HOLD_TIME) {
        state = STATE.DISSOLVING;
        stateTimer = 0;
      }
      break;
    case STATE.DISSOLVING:
      dissolve = Math.min(stateTimer / DISSOLVE_TIME, 1.0);
      if (stateTimer >= DISSOLVE_TIME) {
        state = STATE.BLACK;
        stateTimer = 0;
        particlesReady = false;
      }
      break;
    case STATE.BLACK:
      dissolve = 1;
      if (stateTimer >= BLACK_TIME) {
        transitionToNextImage();
      }
      break;
  }

  // Render
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  if (particlesReady) {
    gl.useProgram(program);
    gl.uniform1f(loc.u_time, globalTime);
    gl.uniform1f(loc.u_dissolve, dissolve);
    gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.POINTS, 0, PARTICLE_COUNT);
    gl.bindVertexArray(null);
  }
}

// ============================================================
// Init
// ============================================================

async function init() {
  images = await loadImages();
  console.log(`Loaded ${images.length} image(s)`);
  transitionToNextImage();
  requestAnimationFrame(frame);
}

init();
