import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

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
controls.enabled = false;
controls.target.set(0, 0, 0);

const PARTICLE_COUNT = 800_000;
const PAINTING_HEIGHT = 2.0;
const ALPHA_THRESHOLD = 0.5 * 255;

const PAINTING_IDS = ['0012', '0013', '0018', '0020', '0022', '0029', '0031', '0032', '0038'];
const PAINTINGS = PAINTING_IDS.map(n => `../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_${n}-trim.png`);
const NUM_PAINTINGS = PAINTINGS.length;

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

// Sample N particles from opaque pixels of this painting.
// Returns { pos: Float32Array(N*2), col: Float32Array(N*3), worldW, worldH }.
function samplePainting(pixels, count) {
  const { data, width: iw, height: ih } = pixels;
  const aspect = iw / ih;
  const worldH = PAINTING_HEIGHT;
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

  for (let p = 0; p < count; p++) {
    const pixelIdx = opaque[(Math.random() * opaque.length) | 0];
    const px = pixelIdx % iw;
    const py = (pixelIdx / iw) | 0;
    const rgba = pixelIdx * 4;

    const nx = px / iw, ny = py / ih;
    const jx = (Math.random() - 0.5) * (worldW / iw);
    const jy = (Math.random() - 0.5) * (worldH / ih);
    pos[p * 2]     = (nx - 0.5) * worldW + jx;
    pos[p * 2 + 1] = (0.5 - ny) * worldH + jy;

    col[p * 3]     = data[rgba]     / 255;
    col[p * 3 + 1] = data[rgba + 1] / 255;
    col[p * 3 + 2] = data[rgba + 2] / 255;
  }

  return { pos, col, worldW, worldH };
}

// Pack per-painting address arrays into (9 × N) DataTextures.
// Layout: column = painting index, row = particle index.
function packAddressTextures(samples) {
  const N = PARTICLE_COUNT;
  const cols = samples.length;

  const posData = new Float32Array(cols * N * 2);
  const colData = new Float32Array(cols * N * 3);

  for (let p = 0; p < cols; p++) {
    const s = samples[p];
    for (let i = 0; i < N; i++) {
      // texel (col=p, row=i) → data index = (i * cols + p)
      const texelIdx = i * cols + p;
      posData[texelIdx * 2]     = s.pos[i * 2];
      posData[texelIdx * 2 + 1] = s.pos[i * 2 + 1];
      colData[texelIdx * 3]     = s.col[i * 3];
      colData[texelIdx * 3 + 1] = s.col[i * 3 + 1];
      colData[texelIdx * 3 + 2] = s.col[i * 3 + 2];
    }
  }

  const posTex = new THREE.DataTexture(posData, cols, N, THREE.RGFormat, THREE.FloatType);
  posTex.minFilter = THREE.NearestFilter;
  posTex.magFilter = THREE.NearestFilter;
  posTex.wrapS = THREE.ClampToEdgeWrapping;
  posTex.wrapT = THREE.ClampToEdgeWrapping;
  posTex.needsUpdate = true;

  const colTex = new THREE.DataTexture(colData, cols, N, THREE.RGBFormat, THREE.FloatType);
  colTex.minFilter = THREE.NearestFilter;
  colTex.magFilter = THREE.NearestFilter;
  colTex.wrapS = THREE.ClampToEdgeWrapping;
  colTex.wrapT = THREE.ClampToEdgeWrapping;
  colTex.needsUpdate = true;

  return { posTex, colTex };
}

async function init() {
  await renderer.init();
  info.textContent = `Loading ${NUM_PAINTINGS} paintings...`;

  const images = await Promise.all(PAINTINGS.map(loadImage));
  const pixels = images.map(img => getPixels(img));

  info.textContent = `Sampling ${PARTICLE_COUNT / 1000}k × ${NUM_PAINTINGS} addresses...`;
  await new Promise(r => requestAnimationFrame(r));

  const samples = pixels.map(px => samplePainting(px, PARTICLE_COUNT));
  const { posTex, colTex } = packAddressTextures(samples);
  console.log('packed:', posTex.image.width, '×', posTex.image.height, 'pos +', colTex.image.width, '×', colTex.image.height, 'col');

  info.textContent = `${NUM_PAINTINGS} paintings · ${(PARTICLE_COUNT / 1e6).toFixed(2)}M particles · WebGPU`;
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
