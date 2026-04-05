import * as THREE from 'https://unpkg.com/three@0.170.0/build/three.module.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const PARTICLE_COUNT = 80000;
const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 4.0;
const BLACK_TIME = 2.0;

// ---------------------------------------------------------------------------
// Gaussian sprite texture (64x64, white with radial alpha falloff)
// ---------------------------------------------------------------------------
function createGaussianTexture(size = 64) {
  const canvas = document.createElement('canvas');
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext('2d');
  const imageData = ctx.createImageData(size, size);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x / half) - 1.0;
      const dy = (y / half) - 1.0;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const alpha = Math.exp(-dist * dist * 4.0);
      const i = (y * size + x) * 4;
      imageData.data[i]     = 255;
      imageData.data[i + 1] = 255;
      imageData.data[i + 2] = 255;
      imageData.data[i + 3] = Math.floor(alpha * 255);
    }
  }
  ctx.putImageData(imageData, 0, 0);
  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// Fallback watercolor gradient images (Lerin palette)
// ---------------------------------------------------------------------------
function generateFallbackImage(index) {
  const w = 800, h = 600;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);

  const palettes = [
    // Deep blues and muted earth
    [
      { cx: 0.3, cy: 0.4, r: 0.6, colors: ['rgba(30,60,120,0.9)', 'rgba(60,90,140,0.4)', 'rgba(30,60,120,0)'] },
      { cx: 0.7, cy: 0.6, r: 0.5, colors: ['rgba(140,110,70,0.8)', 'rgba(160,130,90,0.3)', 'rgba(140,110,70,0)'] },
      { cx: 0.5, cy: 0.3, r: 0.4, colors: ['rgba(180,200,220,0.7)', 'rgba(150,170,200,0.2)', 'rgba(180,200,220,0)'] },
    ],
    // Winter greens and grey
    [
      { cx: 0.4, cy: 0.5, r: 0.7, colors: ['rgba(50,80,60,0.9)', 'rgba(70,100,80,0.4)', 'rgba(50,80,60,0)'] },
      { cx: 0.6, cy: 0.3, r: 0.5, colors: ['rgba(160,170,165,0.8)', 'rgba(140,150,145,0.3)', 'rgba(160,170,165,0)'] },
      { cx: 0.3, cy: 0.7, r: 0.45, colors: ['rgba(90,120,100,0.7)', 'rgba(70,100,80,0.2)', 'rgba(90,120,100,0)'] },
    ],
    // Warm earth and sky
    [
      { cx: 0.5, cy: 0.6, r: 0.65, colors: ['rgba(120,80,50,0.9)', 'rgba(150,110,70,0.4)', 'rgba(120,80,50,0)'] },
      { cx: 0.3, cy: 0.3, r: 0.5, colors: ['rgba(100,130,170,0.8)', 'rgba(120,150,190,0.3)', 'rgba(100,130,170,0)'] },
      { cx: 0.7, cy: 0.4, r: 0.4, colors: ['rgba(200,180,150,0.7)', 'rgba(180,160,130,0.2)', 'rgba(200,180,150,0)'] },
    ],
  ];

  const spots = palettes[index % palettes.length];
  for (const s of spots) {
    const grad = ctx.createRadialGradient(s.cx * w, s.cy * h, 0, s.cx * w, s.cy * h, s.r * Math.max(w, h));
    grad.addColorStop(0, s.colors[0]);
    grad.addColorStop(0.6, s.colors[1]);
    grad.addColorStop(1, s.colors[2]);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }
  return canvas;
}

// ---------------------------------------------------------------------------
// Image loader — try shared/images/, fallback to generated
// ---------------------------------------------------------------------------
async function loadImages() {
  const extensions = ['jpg', 'jpeg', 'png'];
  const found = [];

  // Try to discover images from shared folder
  for (let i = 1; i <= 10; i++) {
    for (const ext of extensions) {
      const url = `../shared/images/${i}.${ext}`;
      try {
        const resp = await fetch(url, { method: 'HEAD' });
        if (resp.ok) { found.push(url); break; }
      } catch (_) { /* ignore */ }
    }
  }

  // Also try common names
  const names = ['lerin', 'painting', 'watercolor', 'sample'];
  for (const name of names) {
    for (const ext of extensions) {
      const url = `../shared/images/${name}.${ext}`;
      try {
        const resp = await fetch(url, { method: 'HEAD' });
        if (resp.ok && !found.includes(url)) { found.push(url); break; }
      } catch (_) { /* ignore */ }
    }
  }

  if (found.length > 0) return { type: 'url', sources: found };
  // Fallback to generated
  return { type: 'canvas', sources: [0, 1, 2] };
}

function loadImageElement(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}

function getImageData(source, type) {
  const canvas = document.createElement('canvas');
  let w, h;
  if (type === 'canvas') {
    const fallback = generateFallbackImage(source);
    w = fallback.width; h = fallback.height;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(fallback, 0, 0);
  } else {
    w = source.width; h = source.height;
    canvas.width = w; canvas.height = h;
    canvas.getContext('2d').drawImage(source, 0, 0);
  }
  return { data: canvas.getContext('2d').getImageData(0, 0, w, h), width: w, height: h };
}

// ---------------------------------------------------------------------------
// Sample pixels from image (brightness-weighted)
// ---------------------------------------------------------------------------
function sampleParticles(imageInfo, count) {
  const { data, width, height } = imageInfo;
  const pixels = data.data;

  // Build brightness CDF for weighted sampling
  const totalPixels = width * height;
  const brightness = new Float32Array(totalPixels);
  let sum = 0;
  for (let i = 0; i < totalPixels; i++) {
    const r = pixels[i * 4], g = pixels[i * 4 + 1], b = pixels[i * 4 + 2];
    const br = (r * 0.299 + g * 0.587 + b * 0.114) / 255;
    // Boost mid-range, keep some darks
    brightness[i] = 0.05 + br * 0.95;
    sum += brightness[i];
  }

  // Normalized CDF
  const cdf = new Float32Array(totalPixels);
  cdf[0] = brightness[0] / sum;
  for (let i = 1; i < totalPixels; i++) {
    cdf[i] = cdf[i - 1] + brightness[i] / sum;
  }

  // Binary search helper
  function sampleIndex() {
    const r = Math.random();
    let lo = 0, hi = totalPixels - 1;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (cdf[mid] < r) lo = mid + 1; else hi = mid;
    }
    return lo;
  }

  const aspect = width / height;
  const scaleY = 1.0;
  const scaleX = aspect;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const noiseOffsets = new Float32Array(count * 3);
  const dissolveDelays = new Float32Array(count);

  for (let i = 0; i < count; i++) {
    const idx = sampleIndex();
    const px = idx % width;
    const py = Math.floor(idx / width);

    // Map to centered coordinates (-scaleX..scaleX, -scaleY..scaleY)
    const x = ((px / width) - 0.5) * 2 * scaleX;
    const y = ((1 - py / height) - 0.5) * 2 * scaleY; // flip Y

    positions[i * 3]     = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = 0;

    // Color with slight warmth/saturation jitter
    let r = pixels[idx * 4] / 255;
    let g = pixels[idx * 4 + 1] / 255;
    let b = pixels[idx * 4 + 2] / 255;

    // Slight warmth
    r = Math.min(1, r + (Math.random() * 0.04 - 0.01));
    g = Math.min(1, g + (Math.random() * 0.02 - 0.01));
    b = Math.min(1, b + (Math.random() * 0.01 - 0.01));

    // Saturation jitter
    const lum = r * 0.299 + g * 0.587 + b * 0.114;
    const satShift = 1.0 + (Math.random() * 0.2 - 0.1);
    r = lum + (r - lum) * satShift;
    g = lum + (g - lum) * satShift;
    b = lum + (b - lum) * satShift;

    colors[i * 3]     = Math.max(0, Math.min(1, r));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, g));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, b));

    // Size: mix of large brushstroke and fine dust
    const isBrush = Math.random() < 0.3;
    sizes[i] = isBrush ? (8 + Math.random() * 12) : (2 + Math.random() * 4);

    // Noise offset for dissolution direction
    noiseOffsets[i * 3]     = (Math.random() - 0.5) * 2;
    noiseOffsets[i * 3 + 1] = (Math.random() - 0.5) * 2;
    noiseOffsets[i * 3 + 2] = Math.random() * 6.28;

    // Dissolve delay (stagger)
    dissolveDelays[i] = Math.random() * 0.3;
  }

  return { positions, colors, sizes, noiseOffsets, dissolveDelays };
}

// ---------------------------------------------------------------------------
// GLSL Simplex 3D noise (Ashima Arts)
// ---------------------------------------------------------------------------
const SIMPLEX_NOISE_GLSL = `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x*34.0)+1.0)*x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0/6.0, 1.0/3.0);
  const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
  vec3 i = floor(v + dot(v, C.yyy));
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
`;

// ---------------------------------------------------------------------------
// Shaders
// ---------------------------------------------------------------------------
const vertexShader = `
uniform float u_time;
uniform float u_dissolve;
uniform float u_pixelRatio;

attribute vec3 customColor;
attribute float size;
attribute vec3 noiseOffset;
attribute float dissolveDelay;

varying vec3 vColor;
varying float vEffectiveDissolve;

${SIMPLEX_NOISE_GLSL}

void main() {
  vColor = customColor;

  // Staggered dissolve
  float ed = clamp((u_dissolve - dissolveDelay) / (1.0 - dissolveDelay), 0.0, 1.0);
  vEffectiveDissolve = ed;

  vec3 pos = position;

  if (ed > 0.0) {
    // Organic displacement via simplex noise
    float t = u_time * 0.3;
    float nx = snoise(vec3(noiseOffset.xy + t, noiseOffset.z));
    float ny = snoise(vec3(noiseOffset.yz + t, noiseOffset.x + 1.0));

    float strength = ed * ed * 1.5;
    pos.x += nx * strength * (0.5 + noiseOffset.x * 0.5);
    pos.y += ny * strength * (0.5 + noiseOffset.y * 0.5);

    // Gentle downward drift
    pos.y -= ed * ed * 0.4;

    // Slight upward curl for lighter particles
    float lightness = dot(customColor, vec3(0.299, 0.587, 0.114));
    pos.y += ed * lightness * 0.15;
  }

  vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPosition;

  // Size reduction as dissolve progresses
  float s = size * (1.0 - ed * 0.7);
  gl_PointSize = s * u_pixelRatio;
}
`;

const fragmentShader = `
uniform sampler2D u_sprite;

varying vec3 vColor;
varying float vEffectiveDissolve;

void main() {
  vec4 sprite = texture2D(u_sprite, gl_PointCoord);
  float spriteAlpha = sprite.a;

  float alpha = spriteAlpha * (1.0 - pow(vEffectiveDissolve, 2.0));
  if (alpha < 0.01) discard;

  gl_FragColor = vec4(vColor * sprite.rgb, alpha);
}
`;

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 1);
renderer.setSize(window.innerWidth, window.innerHeight);

const aspect = window.innerWidth / window.innerHeight;
const camera = new THREE.OrthographicCamera(
  -aspect, aspect, 1, -1, 0.1, 10
);
camera.position.z = 1;

const scene = new THREE.Scene();

const spriteTexture = createGaussianTexture(64);

const material = new THREE.ShaderMaterial({
  uniforms: {
    u_time: { value: 0 },
    u_dissolve: { value: 0 },
    u_pixelRatio: { value: renderer.getPixelRatio() },
    u_sprite: { value: spriteTexture },
  },
  vertexShader,
  fragmentShader,
  transparent: true,
  blending: THREE.AdditiveBlending,
  depthWrite: false,
});

let geometry = new THREE.BufferGeometry();
let points = new THREE.Points(geometry, material);
scene.add(points);

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
const STATE = { HOLDING: 0, DISSOLVING: 1, BLACK: 2, LOADING: 3 };
let state = STATE.LOADING;
let stateStart = 0;
let dissolveProgress = 0;

let imageLibrary = null;
let currentImageIndex = 0;

// ---------------------------------------------------------------------------
// Build particles from image data
// ---------------------------------------------------------------------------
function buildParticles(imageInfo) {
  const { positions, colors, sizes, noiseOffsets, dissolveDelays } = sampleParticles(imageInfo, PARTICLE_COUNT);

  if (geometry) geometry.dispose();
  geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('customColor', new THREE.BufferAttribute(colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(sizes, 1));
  geometry.setAttribute('noiseOffset', new THREE.BufferAttribute(noiseOffsets, 3));
  geometry.setAttribute('dissolveDelay', new THREE.BufferAttribute(dissolveDelays, 1));

  points.geometry = geometry;
}

async function loadNextImage() {
  const lib = imageLibrary;
  const source = lib.sources[currentImageIndex % lib.sources.length];
  currentImageIndex++;

  let imageInfo;
  if (lib.type === 'url') {
    const img = await loadImageElement(source);
    imageInfo = getImageData(img, 'url');
  } else {
    imageInfo = getImageData(source, 'canvas');
  }

  buildParticles(imageInfo);
}

function setState(s) {
  state = s;
  stateStart = performance.now() / 1000;
}

// ---------------------------------------------------------------------------
// Animation
// ---------------------------------------------------------------------------
const clock = new THREE.Clock();

function animate() {
  requestAnimationFrame(animate);

  const elapsed = clock.getElapsedTime();
  const stateElapsed = (performance.now() / 1000) - stateStart;

  switch (state) {
    case STATE.HOLDING:
      dissolveProgress = 0;
      if (stateElapsed >= HOLD_TIME) setState(STATE.DISSOLVING);
      break;

    case STATE.DISSOLVING:
      dissolveProgress = Math.min(stateElapsed / DISSOLVE_TIME, 1.0);
      if (stateElapsed >= DISSOLVE_TIME) setState(STATE.BLACK);
      break;

    case STATE.BLACK:
      dissolveProgress = 1.0;
      if (stateElapsed >= BLACK_TIME) {
        setState(STATE.LOADING);
        loadNextImage().then(() => setState(STATE.HOLDING));
      }
      break;

    case STATE.LOADING:
      dissolveProgress = 1.0;
      break;
  }

  material.uniforms.u_time.value = elapsed;
  material.uniforms.u_dissolve.value = dissolveProgress;
  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Resize handler
// ---------------------------------------------------------------------------
function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  const a = w / h;
  camera.left = -a;
  camera.right = a;
  camera.top = 1;
  camera.bottom = -1;
  camera.updateProjectionMatrix();
  material.uniforms.u_pixelRatio.value = renderer.getPixelRatio();
}
window.addEventListener('resize', onResize);

// ---------------------------------------------------------------------------
// Fullscreen (F key or click)
// ---------------------------------------------------------------------------
function toggleFullscreen() {
  if (!document.fullscreenElement) {
    canvas.requestFullscreen?.() || canvas.webkitRequestFullscreen?.();
  } else {
    document.exitFullscreen?.() || document.webkitExitFullscreen?.();
  }
}
window.addEventListener('keydown', (e) => { if (e.key === 'f' || e.key === 'F') toggleFullscreen(); });
canvas.addEventListener('click', toggleFullscreen);

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
(async () => {
  imageLibrary = await loadImages();
  await loadNextImage();
  setState(STATE.HOLDING);
  animate();
})();
