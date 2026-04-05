import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ─── Settings ───
const settings = {
  particleCount: 2000000,
  // Splat sizes
  splatSize: 0.8,
  splatVariation: 0.4,
  sizeAttenuation: 5.0,
  // Depth mapping
  depthNear: 0.4,
  depthFar: -1.8,
  depthExponent: 1.2, // gamma curve on depth
  // Color
  colorJitter: 0.04,
  saturationJitter: 0.12,
  opacity: 1.0,
  // Rendering
  backgroundColor: '#0a0a0a',
  sortParticles: false,
  // Dissolution (optional, toggle on/off)
  dissolveEnabled: false,
  dissolveProgress: 0.0,
  windStrength: 2.0,
  activationSpread: 0.5,
  gravity: 0.15,
  turbulence: 0.3,
};

// ─── Renderer ───
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(settings.backgroundColor);

// ─── Scene & Camera ───
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 3.5);

// ─── OrbitControls ───
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
controls.rotateSpeed = 0.5;
controls.zoomSpeed = 0.8;
controls.panSpeed = 0.5;
controls.target.set(0, 0, -0.5);
controls.update();

// ─── Gaussian Splat Sprite ───
function createSplatTexture() {
  const size = 128;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  const id = ctx.createImageData(size, size);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x / half) - 1.0;
      const dy = (y / half) - 1.0;
      const dist2 = dx * dx + dy * dy;
      // Tight Gaussian — dense particles need sharp falloff
      const alpha = Math.exp(-dist2 * 5.0);
      const i = (y * size + x) * 4;
      id.data[i] = 255; id.data[i + 1] = 255; id.data[i + 2] = 255;
      id.data[i + 3] = Math.floor(alpha * 255);
    }
  }
  ctx.putImageData(id, 0, 0);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}

// ─── Image Loading ───
const COLOR_IMG = '../shared/images/lerin1.jpg';
const DEPTH_IMG = '../shared/images/lerin1_depth.png';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed: ${src}`));
    img.src = src;
  });
}

function getPixels(img, maxWidth = 1600) {
  // Downsample large images for faster processing
  let w = img.width, h = img.height;
  if (w > maxWidth) {
    const scale = maxWidth / w;
    w = Math.round(w * scale);
    h = Math.round(h * scale);
  }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// ─── Build Point Cloud ───
async function buildPointCloud() {
  let colorData, depthData, imgWidth, imgHeight;
  try {
    const [colorImg, depthImg] = await Promise.all([loadImage(COLOR_IMG), loadImage(DEPTH_IMG)]);
    imgWidth = colorImg.width;
    imgHeight = colorImg.height;
    colorData = getPixels(colorImg);
    depthData = getPixels(depthImg);
  } catch (e) {
    console.error('Failed to load images:', e);
    return null;
  }

  const count = settings.particleCount;
  const aspect = imgWidth / imgHeight;
  const sceneH = 2.0;
  const sceneW = sceneH * aspect;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const seeds = new Float32Array(count * 3);
  const activations = new Float32Array(count);

  const cPx = colorData.data;
  const dPx = depthData.data;
  const total = imgWidth * imgHeight;

  // Brightness CDF for weighted sampling
  const brightness = new Float32Array(total);
  let bSum = 0;
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    const b = (cPx[idx] * 0.299 + cPx[idx + 1] * 0.587 + cPx[idx + 2] * 0.114) / 255;
    brightness[i] = 0.02 + b * 0.98;
    bSum += brightness[i];
  }
  const cdf = new Float32Array(total);
  cdf[0] = brightness[0] / bSum;
  for (let i = 1; i < total; i++) cdf[i] = cdf[i - 1] + brightness[i] / bSum;

  for (let i = 0; i < count; i++) {
    // Weighted random pixel via CDF
    const r = Math.random();
    let lo = 0, hi = total - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
    const pixIdx = lo;
    const px = pixIdx % imgWidth;
    const py = Math.floor(pixIdx / imgWidth);
    const rgba = pixIdx * 4;

    const nx = px / imgWidth;
    const ny = py / imgHeight;

    // Position: centered, depth from depth map
    const x = (nx - 0.5) * sceneW;
    const y = (0.5 - ny) * sceneH;
    const rawDepth = dPx[rgba] / 255;
    const depth = Math.pow(rawDepth, settings.depthExponent);
    const z = depth * (settings.depthNear - settings.depthFar) + settings.depthFar;

    // Sub-pixel jitter to avoid grid artifacts
    const jitterScale = sceneW / imgWidth;
    positions[i * 3] = x + (Math.random() - 0.5) * jitterScale;
    positions[i * 3 + 1] = y + (Math.random() - 0.5) * jitterScale;
    positions[i * 3 + 2] = z;

    // Painterly color variation
    let cr = cPx[rgba] / 255;
    let cg = cPx[rgba + 1] / 255;
    let cb = cPx[rgba + 2] / 255;

    const warmth = (Math.random() - 0.4) * settings.colorJitter;
    cr = Math.max(0, Math.min(1, cr + warmth));
    cg = Math.max(0, Math.min(1, cg + warmth * 0.4));
    cb = Math.max(0, Math.min(1, cb - warmth * 0.3));

    const lum = cr * 0.299 + cg * 0.587 + cb * 0.114;
    const satShift = 1.0 + (Math.random() - 0.5) * settings.saturationJitter;
    cr = Math.max(0, Math.min(1, lum + (cr - lum) * satShift));
    cg = Math.max(0, Math.min(1, lum + (cg - lum) * satShift));
    cb = Math.max(0, Math.min(1, lum + (cb - lum) * satShift));

    colors[i * 3] = cr;
    colors[i * 3 + 1] = cg;
    colors[i * 3 + 2] = cb;

    // Size with variation
    sizes[i] = settings.splatSize + (Math.random() - 0.5) * settings.splatVariation;

    // Dissolution data
    activations[i] = nx * settings.activationSpread;
    seeds[i * 3] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 1] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 2] = (Math.random() - 0.5) * 2;
  }

  return { positions, colors, sizes, seeds, activations };
}

// ─── Shaders ───
const vertexShader = /* glsl */ `
attribute vec3 customColor;
attribute float splatSize;
attribute vec3 seed;
attribute float activation;

uniform float u_pixelRatio;
uniform float u_sizeAttenuation;
uniform float u_opacity;
uniform float u_time;

// Dissolution uniforms
uniform bool u_dissolveEnabled;
uniform float u_dissolveProgress;
uniform float u_windStrength;
uniform float u_gravity;
uniform float u_turbulence;

varying vec3 vColor;
varying float vAlpha;

void main() {
  vec3 pos = position;
  float alpha = u_opacity;

  // Dissolution
  if (u_dissolveEnabled && u_dissolveProgress > 0.0) {
    float d = clamp((u_dissolveProgress - activation) / (1.0 - activation + 0.001), 0.0, 1.0);

    if (d > 0.0) {
      float t = d * d;
      // Leftward wind
      pos.x -= t * u_windStrength;
      // Turbulence
      pos.x += sin(u_time * 1.7 + seed.x * 6.28) * u_turbulence * t;
      pos.y += sin(u_time * 2.3 + seed.y * 6.28) * u_turbulence * 0.7 * t;
      pos.z += sin(u_time * 1.1 + seed.z * 6.28) * u_turbulence * 0.3 * t;
      // Gravity
      pos.y -= t * u_gravity;
      // Fade
      float disp = length(pos - position);
      alpha *= 1.0 - smoothstep(0.2, 2.5, disp);
    }
  }

  vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPosition;

  // Size with perspective attenuation
  gl_PointSize = splatSize * u_pixelRatio * (u_sizeAttenuation / -mvPosition.z);
  gl_PointSize = clamp(gl_PointSize, 0.5, 64.0);

  vColor = customColor;
  vAlpha = alpha;
}
`;

const fragmentShader = /* glsl */ `
uniform sampler2D u_sprite;

varying vec3 vColor;
varying float vAlpha;

void main() {
  float spriteAlpha = texture2D(u_sprite, gl_PointCoord).a;
  float alpha = spriteAlpha * vAlpha;
  if (alpha < 0.003) discard;
  gl_FragColor = vec4(vColor, alpha);
}
`;

// ─── Init ───
async function init() {
  console.time('buildPointCloud');
  const data = await buildPointCloud();
  console.timeEnd('buildPointCloud');
  if (!data) return;
  console.log(`${settings.particleCount.toLocaleString()} particles loaded`);

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('customColor', new THREE.BufferAttribute(data.colors, 3));
  geometry.setAttribute('splatSize', new THREE.BufferAttribute(data.sizes, 1));
  geometry.setAttribute('seed', new THREE.BufferAttribute(data.seeds, 3));
  geometry.setAttribute('activation', new THREE.BufferAttribute(data.activations, 1));

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      u_pixelRatio: { value: renderer.getPixelRatio() },
      u_sizeAttenuation: { value: settings.sizeAttenuation },
      u_opacity: { value: settings.opacity },
      u_sprite: { value: createSplatTexture() },
      u_time: { value: 0 },
      u_dissolveEnabled: { value: settings.dissolveEnabled },
      u_dissolveProgress: { value: 0 },
      u_windStrength: { value: settings.windStrength },
      u_gravity: { value: settings.gravity },
      u_turbulence: { value: settings.turbulence },
    },
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
    depthTest: true,
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // ─── GUI ───
  const gui = new GUI({ title: '3D Point Cloud' });

  const fRender = gui.addFolder('Rendering');
  fRender.add(settings, 'sizeAttenuation', 0.5, 20, 0.25).name('Splat Scale').onChange(v => material.uniforms.u_sizeAttenuation.value = v);
  fRender.add(settings, 'opacity', 0, 1, 0.01).name('Opacity').onChange(v => material.uniforms.u_opacity.value = v);
  fRender.addColor(settings, 'backgroundColor').name('Background').onChange(v => renderer.setClearColor(v));

  const fDepth = gui.addFolder('Depth');
  fDepth.add(settings, 'depthNear', -2, 3, 0.1).name('Near (white)');
  fDepth.add(settings, 'depthFar', -8, 1, 0.1).name('Far (black)');
  fDepth.add(settings, 'depthExponent', 0.2, 3, 0.1).name('Depth Curve');

  const fColor = gui.addFolder('Painterly');
  fColor.add(settings, 'colorJitter', 0, 0.2, 0.01).name('Color Jitter');
  fColor.add(settings, 'saturationJitter', 0, 0.5, 0.01).name('Saturation');

  const fDissolve = gui.addFolder('Dissolution');
  fDissolve.add(settings, 'dissolveEnabled').name('Enable').onChange(v => material.uniforms.u_dissolveEnabled.value = v);
  fDissolve.add(settings, 'dissolveProgress', 0, 1, 0.001).name('Progress').listen().onChange(v => material.uniforms.u_dissolveProgress.value = v);
  fDissolve.add(settings, 'windStrength', 0, 8, 0.1).name('Wind').onChange(v => material.uniforms.u_windStrength.value = v);
  fDissolve.add(settings, 'gravity', 0, 1, 0.01).name('Gravity').onChange(v => material.uniforms.u_gravity.value = v);
  fDissolve.add(settings, 'turbulence', 0, 1, 0.01).name('Turbulence').onChange(v => material.uniforms.u_turbulence.value = v);
  fDissolve.add(settings, 'activationSpread', 0, 1, 0.05).name('L-R Spread');

  const actions = {
    resetCamera: () => {
      camera.position.set(0, 0, 3.5);
      controls.target.set(0, 0, -0.5);
      controls.update();
    },
    autoDissolve: () => {
      settings.dissolveEnabled = true;
      material.uniforms.u_dissolveEnabled.value = true;
      settings.dissolveProgress = 0;
      const start = performance.now();
      const duration = 8000;
      function tick() {
        const p = Math.min(1, (performance.now() - start) / duration);
        settings.dissolveProgress = p;
        material.uniforms.u_dissolveProgress.value = p;
        if (p < 1) requestAnimationFrame(tick);
      }
      tick();
    },
    resetDissolve: () => {
      settings.dissolveProgress = 0;
      material.uniforms.u_dissolveProgress.value = 0;
    },
  };

  gui.add(actions, 'resetCamera').name('Reset Camera');
  gui.add(actions, 'autoDissolve').name('▶ Play Dissolve');
  gui.add(actions, 'resetDissolve').name('↺ Reset Dissolve');

  // ─── Animation ───
  const clock = new THREE.Clock();

  function animate() {
    requestAnimationFrame(animate);
    controls.update();
    material.uniforms.u_time.value = clock.getElapsedTime();
    renderer.render(scene, camera);
  }

  animate();
}

// ─── Resize ───
window.addEventListener('resize', () => {
  const w = window.innerWidth, h = window.innerHeight;
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
});

// ─── Fullscreen (F key) ───
window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  }
});

init();
