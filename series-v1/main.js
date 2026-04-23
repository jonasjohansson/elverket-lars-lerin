import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, vec4, float, Fn } from 'three/tsl';

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
camera.position.set(0, 0, 3.8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false; // free-look toggled with Space later
controls.target.set(0, 0, -0.3);

const PARTICLES_PER_PAINTING = 250_000;
const PAINTING_HEIGHT = 2.0;
const ALPHA_THRESHOLD = 0.5 * 255;

const PAINTINGS = [
  '0012', '0013', '0018', '0020', '0022', '0029', '0031', '0032', '0038',
].map(n => `../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_${n}-trim.png`);

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

// Build per-painting particle arrays. Returns { positions, colors, seeds, edgeDist, worldW, worldH }.
// worldW/worldH are world-space dimensions of the painting (aspect-preserved, height = PAINTING_HEIGHT).
function buildPaintingParticles(pixels, count) {
  const { data, width: iw, height: ih } = pixels;
  const aspect = iw / ih;
  const worldH = PAINTING_HEIGHT;
  const worldW = worldH * aspect;

  // Collect all opaque pixel indices first (for uniform sampling)
  const opaque = [];
  for (let y = 0; y < ih; y++) {
    for (let x = 0; x < iw; x++) {
      const i = (y * iw + x) * 4;
      if (data[i + 3] > ALPHA_THRESHOLD) opaque.push(y * iw + x);
    }
  }
  if (opaque.length === 0) throw new Error('No opaque pixels in painting');

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const seeds = new Float32Array(count * 4);
  // edgeDist computed in a later task; fill with 1.0 (all "interior") for now
  const edgeDist = new Float32Array(count).fill(1.0);

  for (let p = 0; p < count; p++) {
    const pixelIdx = opaque[Math.floor(Math.random() * opaque.length)];
    const px = pixelIdx % iw;
    const py = Math.floor(pixelIdx / iw);
    const rgba = pixelIdx * 4;

    const nx = px / iw, ny = py / ih;
    // sub-pixel jitter so particles don't grid-align
    const jx = (Math.random() - 0.5) * (worldW / iw);
    const jy = (Math.random() - 0.5) * (worldH / ih);
    positions[p * 3]     = (nx - 0.5) * worldW + jx;
    positions[p * 3 + 1] = (0.5 - ny) * worldH + jy;
    positions[p * 3 + 2] = (Math.random() - 0.5) * 0.16;

    colors[p * 3]     = data[rgba]     / 255;
    colors[p * 3 + 1] = data[rgba + 1] / 255;
    colors[p * 3 + 2] = data[rgba + 2] / 255;

    seeds[p * 4]     = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 1] = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 2] = Math.random();
    seeds[p * 4 + 3] = 0.3 + Math.random() * 0.7;
  }

  return { positions, colors, seeds, edgeDist, worldW, worldH };
}

async function init() {
  await renderer.init();

  info.textContent = `Loading ${PAINTINGS.length} paintings...`;
  const images = await Promise.all(PAINTINGS.map(loadImage));
  const pixels = images.map(img => getPixels(img));
  const builds = pixels.map(px => buildPaintingParticles(px, PARTICLES_PER_PAINTING));

  const GAP_FACTOR = 0.6;
  const avgW = builds.reduce((s, b) => s + b.worldW, 0) / builds.length;
  const gap = avgW * GAP_FACTOR;
  let runningX = 0;
  const paintingXs = [];
  for (let i = 0; i < builds.length; i++) {
    const halfW = builds[i].worldW / 2;
    if (i > 0) runningX += builds[i - 1].worldW / 2 + gap + halfW;
    paintingXs.push(runningX);
  }

  const mat = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: true,
  });
  mat.colorNode = Fn(() => vec4(attribute('aColor'), 1.0))();
  mat.sizeNode = float(2.5);
  mat.sizeAttenuation = true;

  const meshes = [];
  for (let i = 0; i < builds.length; i++) {
    const b = builds[i];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b.positions, 3));
    g.setAttribute('aColor',   new THREE.BufferAttribute(b.colors, 3));
    g.setAttribute('aSeed',    new THREE.BufferAttribute(b.seeds, 4));
    g.setAttribute('aEdge',    new THREE.BufferAttribute(b.edgeDist, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    const m = new THREE.Points(g, mat);
    m.position.x = paintingXs[i];
    scene.add(m);
    meshes.push(m);
  }

  info.textContent = `${PAINTINGS.length} paintings · ${((PARTICLES_PER_PAINTING*PAINTINGS.length)/1e6).toFixed(2)}M · WebGPU`;

  camera.position.x = paintingXs[0];
  controls.target.x = paintingXs[0];

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
