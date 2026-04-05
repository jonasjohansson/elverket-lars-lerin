// ============================================================
// Lars Lerin — Pixel Displacement Dissolution (WebGL2 + GLSL)
// Fullscreen image rendered as texture, dissolved via fragment shader
// ============================================================

const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 6.0;
const BLACK_TIME = 2.0;

const STATE = { HOLDING: 0, DISSOLVING: 1, BLACK: 2 };
let state = STATE.BLACK;
let stateTimer = 0;
let dissolve = 0;
let currentImageIndex = 0;
let images = [];

// ---- Canvas & GL ----
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { alpha: false, antialias: false, premultipliedAlpha: false });
if (!gl) {
  document.body.textContent = 'WebGL2 not supported';
  throw new Error('WebGL2 not supported');
}

// ---- Fullscreen (F key only) ----
window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) {
      canvas.requestFullscreen().catch(() => {});
    } else {
      document.exitFullscreen();
    }
  }
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
out vec2 v_uv;

void main() {
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAG = `#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;
uniform float u_dissolve;
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_imageSize;

// ============================================================
// Simplex 3D Noise — Ashima Arts (Stefan Gustavson, Ian McEwan)
// github.com/ashima/webgl-noise
// ============================================================

vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
  const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
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

  vec4 s0 = floor(b0) * 2.0 + 1.0;
  vec4 s1 = floor(b1) * 2.0 + 1.0;
  vec4 sh = -step(h, vec4(0.0));

  vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
  vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

  vec3 p0 = vec3(a0.xy, h.x);
  vec3 p1 = vec3(a0.zw, h.y);
  vec3 p2 = vec3(a1.xy, h.z);
  vec3 p3 = vec3(a1.zw, h.w);

  vec4 norm = taylorInvSqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
  p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

  vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
  m = m * m;
  return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}

// Fractional Brownian Motion for richer noise
float fbm(vec3 p) {
  float val = 0.0;
  float amp = 0.5;
  float freq = 1.0;
  for (int i = 0; i < 4; i++) {
    val += amp * snoise(p * freq);
    freq *= 2.0;
    amp *= 0.5;
  }
  return val;
}

// ============================================================
// Image fitting — cover viewport, centered
// ============================================================

vec2 fitUV(vec2 uv) {
  float viewAspect = u_resolution.x / u_resolution.y;
  float imgAspect = u_imageSize.x / u_imageSize.y;

  vec2 scale = vec2(1.0);
  if (imgAspect > viewAspect) {
    // Image is wider — fit height, crop sides
    scale.x = viewAspect / imgAspect;
  } else {
    // Image is taller — fit width, crop top/bottom
    scale.y = imgAspect / viewAspect;
  }

  return (uv - 0.5) / scale + 0.5;
}

// ============================================================
// Main fragment shader
// ============================================================

void main() {
  vec2 uv = v_uv;

  // Fit image to viewport (contain, not cover — show full painting)
  float viewAspect = u_resolution.x / u_resolution.y;
  float imgAspect = u_imageSize.x / u_imageSize.y;

  vec2 scale = vec2(1.0);
  if (imgAspect > viewAspect) {
    scale.y = viewAspect / imgAspect;
  } else {
    scale.x = imgAspect / viewAspect;
  }

  vec2 imgUV = (uv - 0.5) / scale + 0.5;

  // Check if we're outside the image area
  bool outsideImage = imgUV.x < 0.0 || imgUV.x > 1.0 || imgUV.y < 0.0 || imgUV.y > 1.0;

  if (outsideImage && u_dissolve < 0.001) {
    fragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // Directional dissolve: left side goes first
  float localDelay = imgUV.x * 0.6;
  float d = clamp((u_dissolve - localDelay) / (1.0 - localDelay + 0.001), 0.0, 1.0);

  // Smooth easing for organic feel
  float dEased = d * d * (3.0 - 2.0 * d);

  if (dEased > 0.0) {
    float noiseScale = 3.0;
    float t = u_time;

    // Multi-octave noise for organic displacement
    float n1 = fbm(vec3(imgUV * noiseScale, t * 0.2));
    float n2 = fbm(vec3(imgUV * noiseScale + 100.0, t * 0.15));
    float n3 = snoise(vec3(imgUV * 6.0, t * 0.1));

    // Primary leftward displacement — accelerates with dissolve
    float displacement = dEased * dEased * 0.4;
    imgUV.x -= displacement;

    // Noise-based horizontal scatter — like wind gusts
    imgUV.x -= n1 * dEased * 0.08;

    // Subtle vertical wobble — paint drifting
    imgUV.y += n2 * dEased * 0.03;

    // Fine-grain turbulence for strand-like detail
    imgUV.x -= n3 * dEased * 0.02;
    imgUV.y += snoise(vec3(imgUV * 10.0, t * 0.25)) * dEased * 0.01;
  }

  // Sample the displaced image
  vec4 color;
  if (imgUV.x < 0.0 || imgUV.x > 1.0 || imgUV.y < 0.0 || imgUV.y > 1.0) {
    color = vec4(0.0);
  } else {
    color = texture(u_image, imgUV);
  }

  // Fade: pixels that have been displaced far enough become transparent
  float alpha = 1.0 - smoothstep(0.25, 0.95, dEased);

  // Edge wisps: near the dissolve front, add organic transparency
  float edgeNoise = snoise(vec3(v_uv * 8.0, u_time * 0.3));
  float wispFactor = dEased * 0.35 * (0.5 + edgeNoise * 0.5);
  alpha *= 1.0 - wispFactor;

  // Fine wispy tendrils at the leading edge of dissolution
  float edgeD = smoothstep(0.0, 0.15, dEased) * (1.0 - smoothstep(0.15, 0.5, dEased));
  float tendrilNoise = snoise(vec3(v_uv.x * 20.0, v_uv.y * 12.0, u_time * 0.2));
  alpha -= edgeD * max(tendrilNoise, 0.0) * 0.4;
  alpha = max(alpha, 0.0);

  // Final compositing over black
  fragColor = vec4(color.rgb * alpha, 1.0);
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

// ---- Uniform locations ----
const loc = {
  u_image: gl.getUniformLocation(program, 'u_image'),
  u_dissolve: gl.getUniformLocation(program, 'u_dissolve'),
  u_time: gl.getUniformLocation(program, 'u_time'),
  u_resolution: gl.getUniformLocation(program, 'u_resolution'),
  u_imageSize: gl.getUniformLocation(program, 'u_imageSize'),
};

// ---- Fullscreen quad ----
const quadVerts = new Float32Array([
  -1, -1,
   1, -1,
  -1,  1,
   1,  1,
]);

const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const quadBuffer = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuffer);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);

const aPos = gl.getAttribLocation(program, 'a_position');
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);

gl.bindVertexArray(null);

// ---- Texture ----
const texture = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, texture);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);

let currentImageWidth = 1;
let currentImageHeight = 1;

function uploadImage(img) {
  const w = img.width || img.naturalWidth;
  const h = img.height || img.naturalHeight;
  currentImageWidth = w;
  currentImageHeight = h;

  gl.bindTexture(gl.TEXTURE_2D, texture);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
}

// ============================================================
// Fallback watercolor images
// ============================================================

function generateWatercolorImage(seed) {
  const c = document.createElement('canvas');
  c.width = 1024;
  c.height = 768;
  const ctx = c.getContext('2d');

  const palettes = [
    ['#2b4570', '#5a7fa5', '#8fb3d4', '#c4a882', '#e8dcc8', '#3d6b5e', '#7a9e8e'],
    ['#4a3728', '#7d6b5d', '#b8a592', '#6b8f71', '#a3c4a0', '#d4c5a9', '#8b7355'],
    ['#1a3a4a', '#2d5f73', '#4a8fa8', '#d4a574', '#e8c9a0', '#f0e0c8', '#3a5a3a'],
  ];

  const palette = palettes[seed % palettes.length];

  ctx.fillStyle = palette[palette.length - 1];
  ctx.fillRect(0, 0, c.width, c.height);

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
let imageReady = false;

function transitionToNextImage() {
  const img = images[currentImageIndex % images.length];
  uploadImage(img);
  imageReady = true;
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
        imageReady = false;
      }
      break;
    case STATE.BLACK:
      dissolve = 1;
      if (stateTimer >= BLACK_TIME) {
        transitionToNextImage();
      }
      break;
  }

  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);

  if (imageReady) {
    gl.useProgram(program);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.uniform1i(loc.u_image, 0);
    gl.uniform1f(loc.u_dissolve, dissolve);
    gl.uniform1f(loc.u_time, globalTime);
    gl.uniform2f(loc.u_resolution, canvas.width, canvas.height);
    gl.uniform2f(loc.u_imageSize, currentImageWidth, currentImageHeight);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
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
