import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, vec3, vec4, float, Fn, sin, cos, uniform, smoothstep, mix, pass } from 'three/tsl';
import { afterImage } from 'three/addons/tsl/display/AfterImageNode.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

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
const EDGE_BAND_PX = 40;

const PAINTINGS = [
  '0012', '0013', '0018', '0020', '0022', '0029', '0031', '0032', '0038',
].map(n => `../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_${n}-trim.png`);

const params = {
  waveAmplitude: 0.015,
  waveSpeed: 0.25,
  waveFrequency: 2.0,
  wave2Amplitude: 0.008,
  wave2Speed: 0.18,
  wave2Frequency: 3.5,
  microDrift: 0.0015,
  particleSize: 2.5,
  // Flow-field transition knobs (Anadol-style)
  curlAmp: 0.28,
  curlFreq: 1.8,
  curlSpeed: 0.4,
  coreFlow: 0.3,
  trailStrength: 0.88,
};

const uTime     = uniform(0.0);
const uWaveAmp  = uniform(params.waveAmplitude);
const uWaveSpd  = uniform(params.waveSpeed);
const uWaveFreq = uniform(params.waveFrequency);
const uWave2Amp  = uniform(params.wave2Amplitude);
const uWave2Spd  = uniform(params.wave2Speed);
const uWave2Freq = uniform(params.wave2Frequency);
const uMicroDrift = uniform(params.microDrift);
const uSize = uniform(params.particleSize);

const uWindDir  = uniform(0.0);   // +1 / -1 while transitioning
const uWindGust = uniform(0.0);   // 0..1 envelope over transition
const uStride   = uniform(1.0);   // set from avgW + gap inside init()
const uCurlAmp   = uniform(params.curlAmp);
const uCurlFreq  = uniform(params.curlFreq);
const uCurlSpeed = uniform(params.curlSpeed);
const uCoreFlow  = uniform(params.coreFlow);       // 0 = edges only, 1 = whole painting
const uTrailStrength = uniform(params.trailStrength);
let uPaintingGusts = [];

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

// Chamfer distance transform on alpha. Returns Float32Array length iw*ih:
// 0 at transparent pixels, larger = deeper inside. Not true Euclidean; close enough.
function alphaDistanceField(pixels) {
  const { data, width: iw, height: ih } = pixels;
  const INF = 1e9;
  const d = new Float32Array(iw * ih);
  for (let i = 0; i < d.length; i++) {
    d[i] = data[i * 4 + 3] > ALPHA_THRESHOLD ? INF : 0;
  }
  // Forward pass
  for (let y = 0; y < ih; y++) {
    for (let x = 0; x < iw; x++) {
      const i = y * iw + x;
      if (d[i] === 0) continue;
      if (x > 0)             d[i] = Math.min(d[i], d[i - 1] + 1);
      if (y > 0)             d[i] = Math.min(d[i], d[i - iw] + 1);
      if (x > 0 && y > 0)    d[i] = Math.min(d[i], d[i - iw - 1] + 1.41421);
      if (x < iw - 1 && y>0) d[i] = Math.min(d[i], d[i - iw + 1] + 1.41421);
    }
  }
  // Backward pass
  for (let y = ih - 1; y >= 0; y--) {
    for (let x = iw - 1; x >= 0; x--) {
      const i = y * iw + x;
      if (d[i] === 0) continue;
      if (x < iw - 1)             d[i] = Math.min(d[i], d[i + 1] + 1);
      if (y < ih - 1)             d[i] = Math.min(d[i], d[i + iw] + 1);
      if (x < iw-1 && y < ih-1)   d[i] = Math.min(d[i], d[i + iw + 1] + 1.41421);
      if (x > 0 && y < ih - 1)    d[i] = Math.min(d[i], d[i + iw - 1] + 1.41421);
    }
  }
  return d;
}

// Build per-painting particle arrays. Returns { positions, colors, seeds, edgeDist, worldW, worldH }.
// worldW/worldH are world-space dimensions of the painting (aspect-preserved, height = PAINTING_HEIGHT).
function buildPaintingParticles(pixels, count) {
  const { data, width: iw, height: ih } = pixels;
  const aspect = iw / ih;
  const worldH = PAINTING_HEIGHT;
  const worldW = worldH * aspect;

  const dist = alphaDistanceField(pixels);

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
  const edgeDist = new Float32Array(count);

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

    edgeDist[p] = Math.min(1.0, dist[pixelIdx] / EDGE_BAND_PX);
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
  uStride.value = avgW + gap;
  let runningX = 0;
  const paintingXs = [];
  for (let i = 0; i < builds.length; i++) {
    const halfW = builds[i].worldW / 2;
    if (i > 0) runningX += builds[i - 1].worldW / 2 + gap + halfW;
    paintingXs.push(runningX);
  }

  uPaintingGusts = builds.map(() => uniform(0.0));

  function makeMaterial(uPaintingGust) {
    const m = new THREE.PointsNodeMaterial({
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: true,
    });
    // Edge weight: 1 at silhouette, 0 at core. With uCoreFlow > 0, the core still
    // participates at a reduced weight so the whole painting breathes into the storm.
    const computeDetach = (edge) => {
      const edgeWeight = float(1.0).sub(smoothstep(float(0.0), float(0.6), edge));
      const combined = mix(uCoreFlow, float(1.0), edgeWeight);
      return combined.mul(uWindGust).mul(uPaintingGust);
    };

    m.colorNode = Fn(() => {
      const col = attribute('aColor');
      const detach = computeDetach(attribute('aEdge'));
      const alpha = float(1.0).sub(detach.mul(0.3));
      return vec4(col, alpha);
    })();
    m.sizeNode = uSize;
    m.sizeAttenuation = true;
    m.positionNode = Fn(() => {
      const pos = attribute('position');
      const seed = attribute('aSeed');
      const seedX = seed.x, seedY = seed.y;
      const time = uTime;

      // Ambient waves (unchanged from Task 7)
      const wave1Phase = pos.x.mul(uWaveFreq).add(pos.y.mul(1.5)).add(time.mul(uWaveSpd));
      const w1x = sin(wave1Phase).mul(uWaveAmp);
      const w1y = cos(wave1Phase.mul(0.7)).mul(uWaveAmp.mul(0.8));
      const w1z = sin(wave1Phase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.6));
      const wave2Phase = pos.y.mul(uWave2Freq).add(pos.z.mul(2.0)).add(time.mul(uWave2Spd));
      const w2x = cos(wave2Phase).mul(uWave2Amp.mul(0.5));
      const w2y = sin(wave2Phase).mul(uWave2Amp);
      const w2z = cos(wave2Phase.mul(1.3)).mul(uWave2Amp.mul(0.7));
      const dx = sin(time.mul(0.13).add(seedX.mul(7.0))).mul(uMicroDrift);
      const dy = cos(time.mul(0.11).add(seedY.mul(5.0))).mul(uMicroDrift);

      const detach = computeDetach(attribute('aEdge'));

      // Wind X: outgoing/incoming both drift in +windDir. Travel amplitude = half a stride at peak gust.
      const windX = uWindDir.mul(uStride).mul(detach).mul(0.5);

      // Cross-wind Y and swirl Z from v2's transition scatter pattern
      const tw = time.mul(0.25);
      const windY = sin(tw.add(seedX.mul(3.0))).mul(detach).mul(0.1);
      const windZ = cos(tw.mul(0.7).add(seedX.add(seedY).mul(2.0))).mul(detach).mul(0.08);

      // Pseudo-curl noise: three lobes with swapped axes → divergence-free-ish tumble.
      // Sampled at world position so neighboring particles swirl coherently.
      const cf = uCurlFreq;
      const ct = time.mul(uCurlSpeed);
      const curlX = sin(pos.y.mul(cf).add(ct)).mul(cos(pos.z.mul(cf).add(ct.mul(1.3))));
      const curlY = sin(pos.z.mul(cf).add(ct.mul(1.7))).mul(cos(pos.x.mul(cf).add(ct)));
      const curlZ = sin(pos.x.mul(cf).add(ct.mul(0.7))).mul(cos(pos.y.mul(cf).add(ct.mul(1.3))));
      const curlMag = uCurlAmp.mul(detach);

      return pos.add(vec3(
        w1x.add(w2x).add(dx).add(windX).add(curlX.mul(curlMag)),
        w1y.add(w2y).add(dy).add(windY).add(curlY.mul(curlMag)),
        w1z.add(w2z).add(windZ).add(curlZ.mul(curlMag)),
      ));
    })();
    return m;
  }

  const meshes = [];
  for (let i = 0; i < builds.length; i++) {
    const b = builds[i];
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b.positions, 3));
    g.setAttribute('aColor',   new THREE.BufferAttribute(b.colors, 3));
    g.setAttribute('aSeed',    new THREE.BufferAttribute(b.seeds, 4));
    g.setAttribute('aEdge',    new THREE.BufferAttribute(b.edgeDist, 1));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    const m = new THREE.Points(g, makeMaterial(uPaintingGusts[i]));
    m.position.x = paintingXs[i];
    scene.add(m);
    meshes.push(m);
  }

  info.textContent = `${PAINTINGS.length} paintings · ${((PARTICLES_PER_PAINTING*PAINTINGS.length)/1e6).toFixed(2)}M · WebGPU`;

  camera.position.x = paintingXs[0];
  controls.target.x = paintingXs[0];

  // Post-processing: afterimage feedback for flowing streak trails
  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  postProcessing.outputNode = afterImage(sceneColor, uTrailStrength);

  const TRANSITION_DURATION_ref = { value: 2.5 };  // seconds (mutable for GUI later)
  let currentIndex = 0;
  let targetIndex = 0;
  let transitionStart = 0;
  let transitioning = false;

  function easeInOut(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

  function snapTo(i) {
    if (transitioning) return;
    const clamped = Math.max(0, Math.min(PAINTINGS.length - 1, i));
    if (clamped === currentIndex) return;
    targetIndex = clamped;
    transitionStart = performance.now();
    transitioning = true;
    uWindDir.value = Math.sign(targetIndex - currentIndex);
    uPaintingGusts[currentIndex].value = 1.0;
    uPaintingGusts[targetIndex].value = 1.0;
  }

  window.addEventListener('keydown', (e) => {
    if (e.key === 'ArrowRight') snapTo(currentIndex + 1);
    if (e.key === 'ArrowLeft')  snapTo(currentIndex - 1);
    if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen().catch(() => {});
    }
    if (e.code === 'Space') { e.preventDefault(); controls.enabled = !controls.enabled; }
  });

  const gui = new GUI({ title: 'Series Ribbon' });

  const fR = gui.addFolder('Rendering');
  fR.add(params, 'particleSize', 0.5, 8, 0.1).name('Particle Size').onChange(v => uSize.value = v);

  const fW = gui.addFolder('Ambient');
  fW.add(params, 'waveAmplitude', 0, 0.1, 0.002).name('Wave Amp').onChange(v => uWaveAmp.value = v);
  fW.add(params, 'waveSpeed', 0, 1, 0.01).name('Wave Speed').onChange(v => uWaveSpd.value = v);
  fW.add(params, 'waveFrequency', 0, 8, 0.1).name('Wave Freq').onChange(v => uWaveFreq.value = v);
  fW.add(params, 'wave2Amplitude', 0, 0.05, 0.002).name('Wave 2 Amp').onChange(v => uWave2Amp.value = v);
  fW.add(params, 'wave2Speed', 0, 1, 0.01).name('Wave 2 Speed').onChange(v => uWave2Spd.value = v);
  fW.add(params, 'wave2Frequency', 0, 8, 0.1).name('Wave 2 Freq').onChange(v => uWave2Freq.value = v);
  fW.add(params, 'microDrift', 0, 0.01, 0.0005).name('Micro Drift').onChange(v => uMicroDrift.value = v);

  const fT = gui.addFolder('Transition');
  const transitionParams = { duration: TRANSITION_DURATION_ref.value, stride: avgW + gap };
  fT.add(transitionParams, 'duration', 0.5, 8, 0.1).name('Duration (s)').onChange(v => TRANSITION_DURATION_ref.value = v);
  fT.add(transitionParams, 'stride', 1, 8, 0.1).name('Stride').onChange(v => uStride.value = v);

  const fF = gui.addFolder('Flow');
  fF.add(params, 'curlAmp', 0, 1.5, 0.01).name('Curl Amp').onChange(v => uCurlAmp.value = v);
  fF.add(params, 'curlFreq', 0.1, 8, 0.1).name('Curl Freq').onChange(v => uCurlFreq.value = v);
  fF.add(params, 'curlSpeed', 0, 2, 0.05).name('Curl Speed').onChange(v => uCurlSpeed.value = v);
  fF.add(params, 'coreFlow', 0, 1, 0.02).name('Core Flow').onChange(v => uCoreFlow.value = v);
  fF.add(params, 'trailStrength', 0, 0.98, 0.01).name('Trails').onChange(v => uTrailStrength.value = v);

  const actions = {
    next: () => snapTo(currentIndex + 1),
    prev: () => snapTo(currentIndex - 1),
  };
  gui.add(actions, 'prev').name('← Prev');
  gui.add(actions, 'next').name('→ Next');

  renderer.setAnimationLoop(() => {
    uTime.value = performance.now() / 1000;
    if (transitioning) {
      const progress = Math.min(1, (performance.now() - transitionStart) / (TRANSITION_DURATION_ref.value * 1000));
      uWindGust.value = Math.sin(progress * Math.PI); // 0 → 1 → 0 over the transition
      const eased = easeInOut(progress);
      const fromX = paintingXs[currentIndex];
      const toX = paintingXs[targetIndex];
      const x = fromX + (toX - fromX) * eased;
      camera.position.x = x;
      controls.target.x = x;
      if (progress >= 1) {
        uWindGust.value = 0;
        uPaintingGusts[currentIndex].value = 0;
        uPaintingGusts[targetIndex].value = 0;
        currentIndex = targetIndex;
        transitioning = false;
      }
    }
    controls.update();
    postProcessing.render();
  });
}

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init().catch(err => { info.textContent = 'Error: ' + err.message; console.error(err); });
