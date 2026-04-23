import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, uniform, float, vec2, vec3, vec4, Fn, mix, sin, cos, smoothstep, clamp } from 'three/tsl';

const canvas = document.getElementById('c');
const info = document.getElementById('info');

const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor('#000');
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 3.2);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.04;
controls.target.set(0, 0, 0);

const PARTICLE_COUNT = 800_000;
const PAINTING_HEIGHT = 2.0;
const ALPHA_THRESHOLD = 0.5 * 255;

const PAINTING_A = '../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_0012-trim.png';
const PAINTING_B = '../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_0029-trim.png';

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed: ${src}`));
    img.src = src;
  });
}

function getPixels(img, maxW = 1600) {
  let w = img.width, h = img.height;
  if (w > maxW) { const s = maxW / w; w = Math.round(w * s); h = Math.round(h * s); }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// Sample N particles from opaque pixels. Returns { pos, col, worldW, worldH }.
function samplePainting(pixels, count, worldH) {
  const { data, width: iw, height: ih } = pixels;
  const aspect = iw / ih;
  const worldW = worldH * aspect;

  const opaque = [];
  for (let y = 0; y < ih; y++) {
    for (let x = 0; x < iw; x++) {
      const i = (y * iw + x) * 4;
      if (data[i + 3] > ALPHA_THRESHOLD) opaque.push(y * iw + x);
    }
  }
  if (opaque.length === 0) throw new Error('No opaque pixels in painting');

  const pos = new Float32Array(count * 2);
  const col = new Float32Array(count * 3);
  const jX = worldW / iw;
  const jY = worldH / ih;

  for (let p = 0; p < count; p++) {
    const pixelIdx = opaque[(Math.random() * opaque.length) | 0];
    const px = pixelIdx % iw;
    const py = (pixelIdx / iw) | 0;
    const rgba = pixelIdx * 4;
    const nx = px / iw, ny = py / ih;
    pos[p * 2]     = (nx - 0.5) * worldW + (Math.random() - 0.5) * jX;
    pos[p * 2 + 1] = (0.5 - ny) * worldH + (Math.random() - 0.5) * jY;
    col[p * 3]     = data[rgba]     / 255;
    col[p * 3 + 1] = data[rgba + 1] / 255;
    col[p * 3 + 2] = data[rgba + 2] / 255;
  }
  return { pos, col, worldW, worldH };
}

async function init() {
  await renderer.init();

  info.textContent = 'Loading paintings...';
  const [imgA, imgB] = await Promise.all([loadImage(PAINTING_A), loadImage(PAINTING_B)]);
  const pxA = getPixels(imgA);
  const pxB = getPixels(imgB);

  info.textContent = `Sampling ${(PARTICLE_COUNT / 1e3) | 0}k particles × 2...`;
  await new Promise(r => requestAnimationFrame(r));
  const sA = samplePainting(pxA, PARTICLE_COUNT, PAINTING_HEIGHT);
  const sB = samplePainting(pxB, PARTICLE_COUNT, PAINTING_HEIGHT);
  console.log(`A ${sA.worldW.toFixed(2)}×${sA.worldH.toFixed(2)}, B ${sB.worldW.toFixed(2)}×${sB.worldH.toFixed(2)}`);

  info.textContent = `${(PARTICLE_COUNT / 1e6).toFixed(2)}M particles sampled`;

  const positions = new Float32Array(PARTICLE_COUNT * 3);  // placeholder
  const aPosA = new Float32Array(PARTICLE_COUNT * 2);
  const aColA = new Float32Array(PARTICLE_COUNT * 3);
  const aPosB = new Float32Array(PARTICLE_COUNT * 2);
  const aColB = new Float32Array(PARTICLE_COUNT * 3);
  const aZ    = new Float32Array(PARTICLE_COUNT);
  const aSeed = new Float32Array(PARTICLE_COUNT * 4);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    aPosA[i * 2]     = sA.pos[i * 2];
    aPosA[i * 2 + 1] = sA.pos[i * 2 + 1];
    aColA[i * 3]     = sA.col[i * 3];
    aColA[i * 3 + 1] = sA.col[i * 3 + 1];
    aColA[i * 3 + 2] = sA.col[i * 3 + 2];

    aPosB[i * 2]     = sB.pos[i * 2];
    aPosB[i * 2 + 1] = sB.pos[i * 2 + 1];
    aColB[i * 3]     = sB.col[i * 3];
    aColB[i * 3 + 1] = sB.col[i * 3 + 1];
    aColB[i * 3 + 2] = sB.col[i * 3 + 2];

    aZ[i] = 0.0;  // will become scatter in Task 4

    aSeed[i * 4]     = (Math.random() - 0.5) * 2;
    aSeed[i * 4 + 1] = (Math.random() - 0.5) * 2;
    aSeed[i * 4 + 2] = Math.random();
    aSeed[i * 4 + 3] = 0.3 + Math.random() * 0.7;
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aPosA', new THREE.BufferAttribute(aPosA, 2));
  geometry.setAttribute('aColA', new THREE.BufferAttribute(aColA, 3));
  geometry.setAttribute('aPosB', new THREE.BufferAttribute(aPosB, 2));
  geometry.setAttribute('aColB', new THREE.BufferAttribute(aColB, 3));
  geometry.setAttribute('aZ',    new THREE.BufferAttribute(aZ, 1));
  geometry.setAttribute('aSeed', new THREE.BufferAttribute(aSeed, 4));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);

  const uSize = uniform(4.0);

  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
  });

  material.positionNode = Fn(() => {
    const pA = attribute('aPosA');
    return vec3(pA.x, pA.y, float(0.0));
  })();

  material.colorNode = Fn(() => {
    const cA = attribute('aColA');
    return vec4(cA.x, cA.y, cA.z, 1.0);
  })();

  material.sizeNode = uSize;
  material.sizeAttenuation = true;

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  info.textContent = 'painting A';

  renderer.setAnimationLoop(() => {
    controls.update();
    renderer.render(scene, camera);
  });
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init().catch(err => { info.textContent = 'Error: ' + err.message; console.error(err); });
