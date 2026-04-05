import * as THREE from 'https://unpkg.com/three@0.170.0/build/three.module.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ─── Settings (tweakable via GUI) ───
const settings = {
  // Particles
  particleCount: 600000,
  sizeMin: 0.8,
  sizeMax: 1.8,
  sizeLargeMin: 2.5,
  sizeLargeMax: 6.0,
  largeProbability: 0.2,
  sizeAttenuation: 12.0,
  colorJitter: 0.06,
  saturationJitter: 0.15,

  // Depth
  depthNear: 0.3,
  depthFar: -1.2,

  // Timing
  holdDuration: 4.0,
  dissolveDuration: 8.0,
  blackDuration: 2.0,

  // Dissolution
  activationSpread: 0.5,  // how much left-right delay spread
  windForceMin: 0.3,      // wind on far particles
  windForceMax: 1.0,      // wind on near particles
  windStrength: 2.0,      // overall wind multiplier
  turbulenceX: 0.3,
  turbulenceY: 0.2,
  turbulenceZ: 0.1,
  turbulenceSpeed: 1.7,
  gravity: 0.15,
  fadeStart: 0.2,
  fadeEnd: 2.0,

  // Camera
  cameraDriftX: 0.06,
  cameraDriftY: 0.04,
  cameraDriftZ: 0.03,
  cameraZ: 3.0,
  dissolveCameraDrift: 0.1,

  // Rendering
  backgroundColor: '#000000',
};

// ─── Canvas & Renderer ───
const canvas = document.getElementById('c');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: false });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(settings.backgroundColor);

// ─── Scene & Camera ───
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(50, window.innerWidth / window.innerHeight, 0.1, 100);
camera.position.set(0, 0, settings.cameraZ);
camera.lookAt(0, 0, 0);

// ─── Gaussian Sprite Texture ───
function createSpriteTexture() {
  const size = 64;
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  const id = ctx.createImageData(size, size);
  const half = size / 2;
  for (let y = 0; y < size; y++) {
    for (let x = 0; x < size; x++) {
      const dx = (x / half) - 1.0;
      const dy = (y / half) - 1.0;
      const dist = Math.sqrt(dx * dx + dy * dy);
      const alpha = Math.exp(-dist * dist * 2.5); // wider, softer falloff for painterly look
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

function getPixels(img) {
  const c = document.createElement('canvas');
  c.width = img.width; c.height = img.height;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0);
  return ctx.getImageData(0, 0, img.width, img.height);
}

// ─── Particle Sampling ───
async function loadParticleData() {
  let colorData, depthData, imgWidth, imgHeight;
  try {
    const [colorImg, depthImg] = await Promise.all([loadImage(COLOR_IMG), loadImage(DEPTH_IMG)]);
    imgWidth = colorImg.width;
    imgHeight = colorImg.height;
    colorData = getPixels(colorImg);
    depthData = getPixels(depthImg);
  } catch (e) {
    console.warn('Image load failed:', e.message);
    return generateFallback();
  }

  const count = settings.particleCount;
  const aspect = imgWidth / imgHeight;
  const sceneH = 2.0;
  const sceneW = sceneH * aspect;

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const depths = new Float32Array(count);
  const activations = new Float32Array(count);
  const seeds = new Float32Array(count * 3);

  const cPx = colorData.data;
  const dPx = depthData.data;
  const total = imgWidth * imgHeight;

  // Brightness CDF
  const brightness = new Float32Array(total);
  let bSum = 0;
  for (let i = 0; i < total; i++) {
    const idx = i * 4;
    const b = (cPx[idx] * 0.299 + cPx[idx + 1] * 0.587 + cPx[idx + 2] * 0.114) / 255;
    brightness[i] = 0.05 + b * 0.95;
    bSum += brightness[i];
  }
  const cdf = new Float32Array(total);
  cdf[0] = brightness[0] / bSum;
  for (let i = 1; i < total; i++) cdf[i] = cdf[i - 1] + brightness[i] / bSum;

  for (let i = 0; i < count; i++) {
    const r = Math.random();
    let lo = 0, hi = total - 1;
    while (lo < hi) { const mid = (lo + hi) >> 1; if (cdf[mid] < r) lo = mid + 1; else hi = mid; }
    const pixIdx = lo;
    const px = pixIdx % imgWidth;
    const py = Math.floor(pixIdx / imgWidth);
    const rgba = pixIdx * 4;
    const nx = px / imgWidth;
    const ny = py / imgHeight;

    const x = (nx - 0.5) * sceneW;
    const y = (0.5 - ny) * sceneH;
    const depthVal = dPx[rgba] / 255;
    const z = depthVal * (settings.depthNear - settings.depthFar) + settings.depthFar;

    positions[i * 3] = x;
    positions[i * 3 + 1] = y;
    positions[i * 3 + 2] = z;

    // Painterly color: warmth jitter + saturation variation
    let cr = cPx[rgba] / 255;
    let cg = cPx[rgba + 1] / 255;
    let cb = cPx[rgba + 2] / 255;
    // Warmth shift (slightly random warm/cool per particle)
    const warmth = (Math.random() - 0.4) * settings.colorJitter;
    cr = Math.max(0, Math.min(1, cr + warmth));
    cg = Math.max(0, Math.min(1, cg + warmth * 0.4));
    cb = Math.max(0, Math.min(1, cb - warmth * 0.3));
    // Saturation jitter (some particles more vivid, some more muted)
    const lum = cr * 0.299 + cg * 0.587 + cb * 0.114;
    const satShift = 1.0 + (Math.random() - 0.5) * settings.saturationJitter;
    cr = Math.max(0, Math.min(1, lum + (cr - lum) * satShift));
    cg = Math.max(0, Math.min(1, lum + (cg - lum) * satShift));
    cb = Math.max(0, Math.min(1, lum + (cb - lum) * satShift));
    colors[i * 3] = cr;
    colors[i * 3 + 1] = cg;
    colors[i * 3 + 2] = cb;

    const isLarge = Math.random() >= (1 - settings.largeProbability);
    sizes[i] = isLarge
      ? settings.sizeLargeMin + Math.random() * (settings.sizeLargeMax - settings.sizeLargeMin)
      : settings.sizeMin + Math.random() * (settings.sizeMax - settings.sizeMin);

    depths[i] = depthVal;
    activations[i] = nx * settings.activationSpread;
    seeds[i * 3] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 1] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 2] = (Math.random() - 0.5) * 2;
  }

  return { positions, colors, sizes, depths, activations, seeds };
}

function generateFallback() {
  const count = settings.particleCount;
  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const sizes = new Float32Array(count);
  const depths = new Float32Array(count);
  const activations = new Float32Array(count);
  const seeds = new Float32Array(count * 3);
  for (let i = 0; i < count; i++) {
    const nx = Math.random(), ny = Math.random();
    const dv = 0.3 + nx * 0.3;
    positions[i * 3] = (nx - 0.5) * 3;
    positions[i * 3 + 1] = (0.5 - ny) * 2;
    positions[i * 3 + 2] = dv * (settings.depthNear - settings.depthFar) + settings.depthFar;
    const g = 0.3 + ny * 0.5;
    colors[i * 3] = g * 0.8; colors[i * 3 + 1] = g * 0.85; colors[i * 3 + 2] = g;
    sizes[i] = settings.sizeMin + Math.random() * (settings.sizeMax - settings.sizeMin);
    depths[i] = dv;
    activations[i] = nx * settings.activationSpread;
    seeds[i * 3] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 1] = (Math.random() - 0.5) * 2;
    seeds[i * 3 + 2] = (Math.random() - 0.5) * 2;
  }
  return { positions, colors, sizes, depths, activations, seeds };
}

// ─── Shaders ───
const vertexShader = /* glsl */ `
attribute vec3 customColor;
attribute float size;
attribute float depth;
attribute float activation;
attribute vec3 seed;

uniform float u_time;
uniform float u_pixelRatio;
uniform float u_dissolve;
uniform float u_sizeAttenuation;
uniform float u_windForceMin;
uniform float u_windForceMax;
uniform float u_windStrength;
uniform float u_turbulenceX;
uniform float u_turbulenceY;
uniform float u_turbulenceZ;
uniform float u_turbulenceSpeed;
uniform float u_gravity;
uniform float u_fadeStart;
uniform float u_fadeEnd;

varying vec3 vColor;
varying float vAlpha;

void main() {
  vec3 pos = position;
  float alpha = 1.0;

  float d = clamp((u_dissolve - activation) / (1.0 - activation + 0.001), 0.0, 1.0);

  if (d > 0.0) {
    float t = d * d;
    float force = mix(u_windForceMin, u_windForceMax, depth);

    // Leftward wind
    pos.x -= t * force * u_windStrength;

    // Turbulence
    pos.x += sin(u_time * u_turbulenceSpeed + seed.x * 6.28) * u_turbulenceX * t;
    pos.y += sin(u_time * (u_turbulenceSpeed * 1.35) + seed.y * 6.28) * u_turbulenceY * t;
    pos.z += sin(u_time * (u_turbulenceSpeed * 0.65) + seed.z * 6.28) * u_turbulenceZ * t;

    // Gravity
    pos.y -= t * u_gravity;

    // Fade based on displacement
    float disp = length(pos - position);
    alpha = 1.0 - smoothstep(u_fadeStart, u_fadeEnd, disp);
  }

  vec4 mvPosition = modelViewMatrix * vec4(pos, 1.0);
  gl_Position = projectionMatrix * mvPosition;

  gl_PointSize = size * u_pixelRatio * (u_sizeAttenuation / -mvPosition.z);
  gl_PointSize = max(gl_PointSize, 0.5);

  vColor = customColor;
  vAlpha = alpha;
}
`;

const fragmentShader = /* glsl */ `
uniform sampler2D u_sprite;
varying vec3 vColor;
varying float vAlpha;

void main() {
  vec4 sprite = texture2D(u_sprite, gl_PointCoord);
  float alpha = sprite.a * vAlpha;
  if (alpha < 0.005) discard;
  gl_FragColor = vec4(vColor, alpha);
}
`;

// ─── Init ───
async function init() {
  const data = await loadParticleData();

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('customColor', new THREE.BufferAttribute(data.colors, 3));
  geometry.setAttribute('size', new THREE.BufferAttribute(data.sizes, 1));
  geometry.setAttribute('depth', new THREE.BufferAttribute(data.depths, 1));
  geometry.setAttribute('activation', new THREE.BufferAttribute(data.activations, 1));
  geometry.setAttribute('seed', new THREE.BufferAttribute(data.seeds, 3));

  const material = new THREE.ShaderMaterial({
    vertexShader,
    fragmentShader,
    uniforms: {
      u_time: { value: 0 },
      u_pixelRatio: { value: renderer.getPixelRatio() },
      u_dissolve: { value: 0 },
      u_sprite: { value: createSpriteTexture() },
      u_sizeAttenuation: { value: settings.sizeAttenuation },
      u_windForceMin: { value: settings.windForceMin },
      u_windForceMax: { value: settings.windForceMax },
      u_windStrength: { value: settings.windStrength },
      u_turbulenceX: { value: settings.turbulenceX },
      u_turbulenceY: { value: settings.turbulenceY },
      u_turbulenceZ: { value: settings.turbulenceZ },
      u_turbulenceSpeed: { value: settings.turbulenceSpeed },
      u_gravity: { value: settings.gravity },
      u_fadeStart: { value: settings.fadeStart },
      u_fadeEnd: { value: settings.fadeEnd },
    },
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
    depthTest: true,
  });

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // ─── GUI ───
  const gui = new GUI({ title: 'Dissolution Settings' });

  const fTiming = gui.addFolder('Timing');
  fTiming.add(settings, 'holdDuration', 0, 20, 0.5).name('Hold (s)');
  fTiming.add(settings, 'dissolveDuration', 1, 30, 0.5).name('Dissolve (s)');
  fTiming.add(settings, 'blackDuration', 0, 10, 0.5).name('Black (s)');

  const fWind = gui.addFolder('Wind Force');
  fWind.add(settings, 'windStrength', 0, 10, 0.1).name('Strength').onChange(v => material.uniforms.u_windStrength.value = v);
  fWind.add(settings, 'windForceMin', 0, 2, 0.05).name('Force Min (far)').onChange(v => material.uniforms.u_windForceMin.value = v);
  fWind.add(settings, 'windForceMax', 0, 3, 0.05).name('Force Max (near)').onChange(v => material.uniforms.u_windForceMax.value = v);
  fWind.add(settings, 'activationSpread', 0, 1, 0.05).name('L-R Spread');
  fWind.add(settings, 'gravity', 0, 1, 0.01).name('Gravity').onChange(v => material.uniforms.u_gravity.value = v);

  const fTurb = gui.addFolder('Turbulence');
  fTurb.add(settings, 'turbulenceX', 0, 1, 0.01).name('X Amount').onChange(v => material.uniforms.u_turbulenceX.value = v);
  fTurb.add(settings, 'turbulenceY', 0, 1, 0.01).name('Y Amount').onChange(v => material.uniforms.u_turbulenceY.value = v);
  fTurb.add(settings, 'turbulenceZ', 0, 1, 0.01).name('Z Amount').onChange(v => material.uniforms.u_turbulenceZ.value = v);
  fTurb.add(settings, 'turbulenceSpeed', 0, 5, 0.1).name('Speed').onChange(v => material.uniforms.u_turbulenceSpeed.value = v);

  const fFade = gui.addFolder('Fade');
  fFade.add(settings, 'fadeStart', 0, 2, 0.05).name('Fade Start').onChange(v => material.uniforms.u_fadeStart.value = v);
  fFade.add(settings, 'fadeEnd', 0.5, 5, 0.1).name('Fade End').onChange(v => material.uniforms.u_fadeEnd.value = v);

  const fSize = gui.addFolder('Particle Size');
  fSize.add(settings, 'sizeAttenuation', 1, 50, 1).name('Attenuation').onChange(v => material.uniforms.u_sizeAttenuation.value = v);

  const fColor = gui.addFolder('Painterly');
  fColor.add(settings, 'colorJitter', 0, 0.2, 0.01).name('Color Jitter');
  fColor.add(settings, 'saturationJitter', 0, 0.5, 0.01).name('Saturation Jitter');

  const fDepth = gui.addFolder('Depth');
  fDepth.add(settings, 'depthNear', -2, 3, 0.1).name('Near (white)');
  fDepth.add(settings, 'depthFar', -5, 1, 0.1).name('Far (black)');

  const fCamera = gui.addFolder('Camera');
  fCamera.add(settings, 'cameraZ', 1, 10, 0.1).name('Distance');
  fCamera.add(settings, 'cameraDriftX', 0, 0.3, 0.01).name('Drift X');
  fCamera.add(settings, 'cameraDriftY', 0, 0.3, 0.01).name('Drift Y');
  fCamera.add(settings, 'cameraDriftZ', 0, 0.3, 0.01).name('Drift Z');
  fCamera.add(settings, 'dissolveCameraDrift', 0, 0.5, 0.01).name('Dissolve Drift');

  fCamera.addColor(settings, 'backgroundColor').name('Background').onChange(v => renderer.setClearColor(v));

  // ─── Animation ───
  const clock = new THREE.Clock();
  let cycleStart = 0;

  function animate() {
    requestAnimationFrame(animate);

    const elapsed = clock.getElapsedTime();
    const cycleDuration = settings.holdDuration + settings.dissolveDuration + settings.blackDuration;
    const cycleTime = elapsed - cycleStart;

    let dissolve = 0;
    if (cycleTime < settings.holdDuration) {
      dissolve = 0;
    } else if (cycleTime < settings.holdDuration + settings.dissolveDuration) {
      dissolve = (cycleTime - settings.holdDuration) / settings.dissolveDuration;
    } else if (cycleTime < cycleDuration) {
      dissolve = 1.0;
    } else {
      cycleStart = elapsed;
      dissolve = 0;
    }

    material.uniforms.u_time.value = elapsed;
    material.uniforms.u_dissolve.value = dissolve;

    // Camera drift
    camera.position.x = Math.sin(elapsed * 0.25) * settings.cameraDriftX;
    camera.position.y = Math.cos(elapsed * 0.2) * settings.cameraDriftY;
    camera.position.z = settings.cameraZ + Math.sin(elapsed * 0.15) * settings.cameraDriftZ;

    if (dissolve > 0 && dissolve < 1) {
      camera.position.x += dissolve * settings.dissolveCameraDrift;
    }

    camera.lookAt(0, 0, 0);
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
