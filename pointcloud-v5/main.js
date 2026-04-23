import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn, sin, cos } from 'three/tsl';

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
const PARTICLES_PER_ROW = 1024;
const ROWS_PER_PAINTING = Math.ceil(PARTICLE_COUNT / PARTICLES_PER_ROW);

const PAINTING_IDS = ['0012', '0013', '0018', '0020', '0022', '0029', '0031', '0032', '0038'];
const PAINTINGS = PAINTING_IDS.map(n => `../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_${n}-trim.png`);
const NUM_PAINTINGS = PAINTINGS.length;

const params = {
  splatScale: 5.0,
  opacity: 0.9,
  bgColor: '#000000',
  waveAmplitude: 0.015,
  waveSpeed: 0.25,
  waveFrequency: 2.0,
  wave2Amplitude: 0.008,
  wave2Speed: 0.18,
  wave2Frequency: 3.5,
  microDrift: 0.0015,
};

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

// Pack per-painting address arrays into a 1024-wide atlas texture.
// Layout: 1024 particles per row, 9 paintings stacked vertically.
// For particle i in painting p:
//   x = i % PARTICLES_PER_ROW
//   y = p * ROWS_PER_PAINTING + floor(i / PARTICLES_PER_ROW)
// posTex is RG float; colTex is RGBA float (alpha=1, unused but RGB-float is deprecated).
function packAddressTextures(samples) {
  const N = PARTICLE_COUNT;
  const cols = samples.length;
  const texW = PARTICLES_PER_ROW;
  const texH = cols * ROWS_PER_PAINTING;

  const posData = new Float32Array(texW * texH * 2);
  const colData = new Float32Array(texW * texH * 4);

  for (let p = 0; p < cols; p++) {
    const s = samples[p];
    const yOffset = p * ROWS_PER_PAINTING;
    for (let i = 0; i < N; i++) {
      const x = i % texW;
      const y = yOffset + ((i / texW) | 0);
      const texelIdx = y * texW + x;
      posData[texelIdx * 2]     = s.pos[i * 2];
      posData[texelIdx * 2 + 1] = s.pos[i * 2 + 1];
      colData[texelIdx * 4]     = s.col[i * 3];
      colData[texelIdx * 4 + 1] = s.col[i * 3 + 1];
      colData[texelIdx * 4 + 2] = s.col[i * 3 + 2];
      colData[texelIdx * 4 + 3] = 1.0;
    }
  }

  const posTex = new THREE.DataTexture(posData, texW, texH, THREE.RGFormat, THREE.FloatType);
  posTex.minFilter = THREE.NearestFilter;
  posTex.magFilter = THREE.NearestFilter;
  posTex.wrapS = THREE.ClampToEdgeWrapping;
  posTex.wrapT = THREE.ClampToEdgeWrapping;
  posTex.needsUpdate = true;

  const colTex = new THREE.DataTexture(colData, texW, texH, THREE.RGBAFormat, THREE.FloatType);
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
  console.log(`packed: ${posTex.image.width}×${posTex.image.height} atlas · ${PARTICLES_PER_ROW}/row · ${ROWS_PER_PAINTING}rows/painting × ${NUM_PAINTINGS}`);

  // Geometry: one vertex per particle. position attr is a placeholder (zeroes);
  // real position is fetched from posTex in the vertex node.
  const positions = new Float32Array(PARTICLE_COUNT * 3);
  const indices = new Float32Array(PARTICLE_COUNT);
  const seeds = new Float32Array(PARTICLE_COUNT * 4);
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    indices[i] = i;
    seeds[i * 4]     = (Math.random() - 0.5) * 2;
    seeds[i * 4 + 1] = (Math.random() - 0.5) * 2;
    seeds[i * 4 + 2] = Math.random();              // delay 0..1
    seeds[i * 4 + 3] = 0.3 + Math.random() * 0.7;  // size jitter
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute('aIndex',   new THREE.BufferAttribute(indices, 1));
  geometry.setAttribute('aSeed',    new THREE.BufferAttribute(seeds, 4));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);

  // Uniforms. Only from-index is wired up in this task.
  const uFromIdx      = uniform(0.0);
  const uSize         = uniform(params.splatScale);
  const uAtlasW       = uniform(PARTICLES_PER_ROW);                        // 1024
  const uAtlasH       = uniform(NUM_PAINTINGS * ROWS_PER_PAINTING);        // 7038
  const uRowsPerPaint = uniform(ROWS_PER_PAINTING);                        // 782
  const uTime         = uniform(0.0);
  const uWaveAmp      = uniform(params.waveAmplitude);
  const uWaveSpd      = uniform(params.waveSpeed);
  const uWaveFreq     = uniform(params.waveFrequency);
  const uWave2Amp     = uniform(params.wave2Amplitude);
  const uWave2Spd     = uniform(params.wave2Speed);
  const uWave2Freq    = uniform(params.wave2Frequency);
  const uMicroDrift   = uniform(params.microDrift);

  // Returns vec2 uv into the 1024-wide atlas for (paintingIdx, particleIdx).
  // x = particleIdx % ATLAS_W
  // y = paintingIdx * ROWS_PER_PAINTING + floor(particleIdx / ATLAS_W)
  // uv = ((x + 0.5) / ATLAS_W, (y + 0.5) / ATLAS_H)
  function atlasUV(paintingIdx, particleIdx) {
    const row = particleIdx.div(uAtlasW).floor();
    const col = particleIdx.sub(row.mul(uAtlasW));      // = particleIdx mod ATLAS_W
    const y   = paintingIdx.mul(uRowsPerPaint).add(row);
    return vec2(col.add(0.5).div(uAtlasW), y.add(0.5).div(uAtlasH));
  }

  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
  });

  material.positionNode = Fn(() => {
    const idx  = attribute('aIndex');
    const seed = attribute('aSeed');
    const seedX = seed.x, seedY = seed.y;
    const t = uTime;

    const uvFrom = atlasUV(uFromIdx, idx);
    const posFrom = texture(posTex, uvFrom).xy;
    const pos = vec3(posFrom.x, posFrom.y, float(0.0));

    const wave1Phase = pos.x.mul(uWaveFreq).add(pos.y.mul(1.5)).add(t.mul(uWaveSpd));
    const w1x = sin(wave1Phase).mul(uWaveAmp);
    const w1y = cos(wave1Phase.mul(0.7)).mul(uWaveAmp.mul(0.8));
    const w1z = sin(wave1Phase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.6));

    const wave2Phase = pos.y.mul(uWave2Freq).add(pos.z.mul(2.0)).add(t.mul(uWave2Spd));
    const w2x = cos(wave2Phase).mul(uWave2Amp.mul(0.5));
    const w2y = sin(wave2Phase).mul(uWave2Amp);
    const w2z = cos(wave2Phase.mul(1.3)).mul(uWave2Amp.mul(0.7));

    const dx = sin(t.mul(0.13).add(seedX.mul(7.0))).mul(uMicroDrift);
    const dy = cos(t.mul(0.11).add(seedY.mul(5.0))).mul(uMicroDrift);

    return pos.add(vec3(
      w1x.add(w2x).add(dx),
      w1y.add(w2y).add(dy),
      w1z.add(w2z),
    ));
  })();

  material.colorNode = Fn(() => {
    const idx = attribute('aIndex');
    const uvFrom = atlasUV(uFromIdx, idx);
    const colFrom = texture(colTex, uvFrom).xyz;
    return vec4(colFrom, 1.0);
  })();

  material.sizeNode = uSize;
  material.sizeAttenuation = true;

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // Debug handle for manually changing the rendered painting (removed in Task 6).
  window.__u = { uFromIdx };

  info.textContent = `${NUM_PAINTINGS} paintings · ${(PARTICLE_COUNT / 1e6).toFixed(2)}M particles · WebGPU`;
  renderer.setAnimationLoop(() => {
    uTime.value = performance.now() / 1000;
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
