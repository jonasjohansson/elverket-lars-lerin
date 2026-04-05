import * as THREE from 'https://unpkg.com/three@0.170.0/build/three.module.js';

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------
const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 6.0;
const BLACK_TIME = 2.0;
const DELAY_SPREAD = 0.6; // left-to-right delay range

// ---------------------------------------------------------------------------
// Fallback watercolor gradient (Lerin palette)
// ---------------------------------------------------------------------------
function generateFallbackTexture(index) {
  const w = 1024, h = 768;
  const canvas = document.createElement('canvas');
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  ctx.fillStyle = '#000';
  ctx.fillRect(0, 0, w, h);

  const palettes = [
    [
      { cx: 0.3, cy: 0.4, r: 0.6, colors: ['rgba(30,60,120,0.9)', 'rgba(60,90,140,0.4)', 'rgba(30,60,120,0)'] },
      { cx: 0.7, cy: 0.6, r: 0.5, colors: ['rgba(140,110,70,0.8)', 'rgba(160,130,90,0.3)', 'rgba(140,110,70,0)'] },
      { cx: 0.5, cy: 0.3, r: 0.4, colors: ['rgba(180,200,220,0.7)', 'rgba(150,170,200,0.2)', 'rgba(180,200,220,0)'] },
    ],
    [
      { cx: 0.4, cy: 0.5, r: 0.7, colors: ['rgba(50,80,60,0.9)', 'rgba(70,100,80,0.4)', 'rgba(50,80,60,0)'] },
      { cx: 0.6, cy: 0.3, r: 0.5, colors: ['rgba(160,170,165,0.8)', 'rgba(140,150,145,0.3)', 'rgba(160,170,165,0)'] },
      { cx: 0.3, cy: 0.7, r: 0.45, colors: ['rgba(90,120,100,0.7)', 'rgba(70,100,80,0.2)', 'rgba(90,120,100,0)'] },
    ],
    [
      { cx: 0.5, cy: 0.6, r: 0.65, colors: ['rgba(120,80,50,0.9)', 'rgba(150,110,70,0.4)', 'rgba(120,80,50,0)'] },
      { cx: 0.3, cy: 0.3, r: 0.5, colors: ['rgba(100,130,170,0.8)', 'rgba(120,150,190,0.3)', 'rgba(100,130,170,0)'] },
      { cx: 0.7, cy: 0.4, r: 0.4, colors: ['rgba(200,180,150,0.7)', 'rgba(180,160,130,0.2)', 'rgba(200,180,150,0)'] },
    ],
  ];

  const spots = palettes[index % palettes.length];
  for (const s of spots) {
    const grad = ctx.createRadialGradient(
      s.cx * w, s.cy * h, 0,
      s.cx * w, s.cy * h, s.r * Math.max(w, h)
    );
    grad.addColorStop(0, s.colors[0]);
    grad.addColorStop(0.6, s.colors[1]);
    grad.addColorStop(1, s.colors[2]);
    ctx.fillStyle = grad;
    ctx.fillRect(0, 0, w, h);
  }

  const tex = new THREE.CanvasTexture(canvas);
  tex.needsUpdate = true;
  return tex;
}

// ---------------------------------------------------------------------------
// Image loader — try shared/images/, fallback to generated
// ---------------------------------------------------------------------------
async function loadImages() {
  const loader = new THREE.TextureLoader();
  const extensions = ['jpg', 'png'];
  const found = [];

  const names = ['lerin1', 'lerin2', 'lerin3', 'lerin', 'painting', 'watercolor', 'sample'];
  for (let i = 1; i <= 10; i++) names.push(String(i));

  for (const name of names) {
    for (const ext of extensions) {
      const url = `../shared/images/${name}.${ext}`;
      try {
        const resp = await fetch(url, { method: 'HEAD' });
        if (resp.ok && !found.includes(url)) { found.push(url); break; }
      } catch (_) { /* ignore */ }
    }
  }

  if (found.length > 0) {
    const textures = [];
    for (const url of found) {
      try {
        const tex = await loader.loadAsync(url);
        tex.minFilter = THREE.LinearFilter;
        tex.magFilter = THREE.LinearFilter;
        textures.push(tex);
      } catch (_) { /* skip broken */ }
    }
    if (textures.length > 0) return textures;
  }

  // Fallback
  return [generateFallbackTexture(0), generateFallbackTexture(1), generateFallbackTexture(2)];
}

// ---------------------------------------------------------------------------
// Ashima Arts Simplex 3D Noise (full)
// ---------------------------------------------------------------------------
const SIMPLEX_NOISE_GLSL = /* glsl */ `
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 1.0) * x); }
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
  vec3  ns = n_ * D.wyz - D.xzx;

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
`;

// ---------------------------------------------------------------------------
// Vertex shader — simple fullscreen quad
// ---------------------------------------------------------------------------
const vertexShader = /* glsl */ `
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}
`;

// ---------------------------------------------------------------------------
// Fragment shader — strand/fiber dissolution
// ---------------------------------------------------------------------------
const fragmentShader = /* glsl */ `
precision highp float;

uniform sampler2D u_image;
uniform float u_time;
uniform float u_dissolve;       // 0 = solid, 1 = fully dissolved
uniform float u_delaySpread;    // how much left-to-right stagger
uniform vec2 u_imageSize;       // width, height of the image in pixels
uniform vec2 u_resolution;      // viewport width, height

varying vec2 vUv;

${SIMPLEX_NOISE_GLSL}

void main() {
  // --- Aspect-correct UV mapping (contain) ---
  float viewAspect = u_resolution.x / u_resolution.y;
  float imgAspect  = u_imageSize.x / u_imageSize.y;

  vec2 scale;
  if (viewAspect > imgAspect) {
    // Viewport wider than image — fit height
    scale = vec2(imgAspect / viewAspect, 1.0);
  } else {
    // Viewport taller than image — fit width
    scale = vec2(1.0, viewAspect / imgAspect);
  }

  vec2 uv = (vUv - 0.5) / scale + 0.5;

  // Outside image bounds — black
  if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
    gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);
    return;
  }

  // --- Directional dissolve: left-to-right with delay spread ---
  float localDelay = uv.x * u_delaySpread;
  float d = clamp((u_dissolve - localDelay) / (1.0 - localDelay + 0.001), 0.0, 1.0);

  // When fully holding (d == 0), show crisp image
  if (d <= 0.0) {
    gl_FragColor = texture2D(u_image, uv);
    return;
  }

  // --- Strand noise: stretched horizontally for fiber look ---
  // Low x-frequency, high y-frequency = horizontal strands
  float t = u_time * 0.1;

  // Primary strand pattern — large fibers
  float n1 = snoise(vec3(uv.x * 2.0, uv.y * 20.0, t));
  // Finer detail — thinner sub-strands
  float n2 = snoise(vec3(uv.x * 5.0, uv.y * 40.0, t * 1.5));
  // Very fine hair-like detail
  float n3 = snoise(vec3(uv.x * 8.0, uv.y * 80.0, t * 0.7));

  float strandNoise = (n1 + n2 * 0.5 + n3 * 0.25) / 1.75;

  // Threshold rises with dissolve — more strands peel away
  // Maps d (0..1) to threshold (-1..1)
  float threshold = d * 2.0 - 1.0;

  float alpha;
  vec2 sampleUv = uv;

  if (strandNoise > threshold) {
    // --- This strand is peeling: displace and fade ---
    float peelAmount = (strandNoise - threshold) / (1.0 - threshold + 0.001);
    peelAmount = clamp(peelAmount, 0.0, 1.0);

    // Smooth the peel for organic feel
    float smoothPeel = peelAmount * peelAmount * (3.0 - 2.0 * peelAmount);

    // Leftward displacement — strands blow away to the left
    sampleUv.x -= smoothPeel * d * 0.3;

    // Slight vertical drift — organic wavering
    float drift = snoise(vec3(uv * 3.0, u_time * 0.2));
    sampleUv.y += drift * smoothPeel * 0.02;

    // Slight curl — strands curve as they detach
    float curl = snoise(vec3(uv.x * 4.0, uv.y * 10.0, u_time * 0.15));
    sampleUv.y += curl * smoothPeel * d * 0.015;

    // Fade out peeling strands
    alpha = 1.0 - smoothPeel * d;
    alpha = max(alpha, 0.0);

    // Additional fade for strands that have moved far
    float displacement = abs(sampleUv.x - uv.x);
    alpha *= 1.0 - smoothstep(0.0, 0.25, displacement);

  } else {
    // --- Solid: still part of the image ---
    alpha = 1.0;
  }

  // Clamp sample UVs to avoid wrapping artifacts
  sampleUv = clamp(sampleUv, vec2(0.0), vec2(1.0));

  vec4 color = texture2D(u_image, sampleUv);

  // Final fadeout at the very end of the dissolve
  float globalFade = 1.0 - smoothstep(0.85, 1.0, d);
  alpha *= globalFade;

  gl_FragColor = vec4(color.rgb, 1.0) * alpha;
}
`;

// ---------------------------------------------------------------------------
// Renderer, camera, scene
// ---------------------------------------------------------------------------
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
renderer.setClearColor(0x000000, 1);
renderer.setSize(window.innerWidth, window.innerHeight);

const camera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
const scene = new THREE.Scene();

// Fullscreen quad
const geometry = new THREE.PlaneGeometry(2, 2);
const material = new THREE.ShaderMaterial({
  uniforms: {
    u_image:      { value: null },
    u_time:       { value: 0 },
    u_dissolve:   { value: 0 },
    u_delaySpread:{ value: DELAY_SPREAD },
    u_imageSize:  { value: new THREE.Vector2(1, 1) },
    u_resolution: { value: new THREE.Vector2(window.innerWidth, window.innerHeight) },
  },
  vertexShader,
  fragmentShader,
  transparent: true,
  depthWrite: false,
});

const quad = new THREE.Mesh(geometry, material);
scene.add(quad);

// ---------------------------------------------------------------------------
// State machine
// ---------------------------------------------------------------------------
const STATE = { HOLDING: 0, DISSOLVING: 1, BLACK: 2 };
let state = STATE.HOLDING;
let stateStart = 0;
let dissolveProgress = 0;

let textures = [];
let currentIndex = 0;

function setState(s) {
  state = s;
  stateStart = performance.now() / 1000;
}

function setTexture(tex) {
  material.uniforms.u_image.value = tex;
  if (tex.image) {
    material.uniforms.u_imageSize.value.set(tex.image.width, tex.image.height);
  } else {
    material.uniforms.u_imageSize.value.set(1024, 768);
  }
}

function nextImage() {
  currentIndex = (currentIndex + 1) % textures.length;
  setTexture(textures[currentIndex]);
}

// ---------------------------------------------------------------------------
// Animation loop
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
      // Smooth ease-in-out for the dissolve
      const raw = Math.min(stateElapsed / DISSOLVE_TIME, 1.0);
      dissolveProgress = raw < 0.5
        ? 2 * raw * raw
        : 1 - Math.pow(-2 * raw + 2, 2) / 2;
      if (stateElapsed >= DISSOLVE_TIME) setState(STATE.BLACK);
      break;

    case STATE.BLACK:
      dissolveProgress = 1.0;
      if (stateElapsed >= BLACK_TIME) {
        nextImage();
        setState(STATE.HOLDING);
      }
      break;
  }

  material.uniforms.u_time.value = elapsed;
  material.uniforms.u_dissolve.value = dissolveProgress;

  renderer.render(scene, camera);
}

// ---------------------------------------------------------------------------
// Resize
// ---------------------------------------------------------------------------
function onResize() {
  const w = window.innerWidth;
  const h = window.innerHeight;
  renderer.setSize(w, h);
  renderer.setPixelRatio(window.devicePixelRatio);
  material.uniforms.u_resolution.value.set(w, h);
}
window.addEventListener('resize', onResize);

// ---------------------------------------------------------------------------
// Fullscreen — F key only
// ---------------------------------------------------------------------------
window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) {
      canvas.requestFullscreen?.() || canvas.webkitRequestFullscreen?.();
    } else {
      document.exitFullscreen?.() || document.webkitExitFullscreen?.();
    }
  }
});

// ---------------------------------------------------------------------------
// Start
// ---------------------------------------------------------------------------
(async () => {
  textures = await loadImages();
  setTexture(textures[0]);
  setState(STATE.HOLDING);
  animate();
})();
