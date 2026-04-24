import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, uniform, float, vec2, vec3, vec4, Fn, mix, sin, cos, smoothstep, clamp, uv } from 'three/tsl';

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

const params = {
  holdDuration: 4.0,
  transitionDuration: 6.0,
  autoLoop: true,
};

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

    aZ[i] = (Math.random() - 0.5) * 0.4;  // [-0.2, +0.2] default scatter

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
  const uPhase = uniform(0.0);
  const uPatchScale = uniform(3.0);
  const uFadeWindow = uniform(0.08);
  const uTime = uniform(0.0);
  const uDispersalAmp = uniform(0.35);
  const uGravity = uniform(0.25);
  const uChaosAmp = uniform(0.18);
  const uDarken = uniform(0.7);
  const uOpacity = uniform(0.9);
  const uShrink = uniform(0.85);
  const uWaveAmp = uniform(0.015);
  const uWaveSpeed = uniform(0.25);
  const uWaveFreq = uniform(2.0);

  // Clustered death moment dM ∈ [0,1] from position — neighbors share similar dM.
  const dieMoment = (posXY, delay) => {
    const pf = uPatchScale;
    const a = sin(posXY.x.mul(pf)).mul(sin(posXY.y.mul(pf.mul(0.87)).add(1.3)));
    const b = sin(posXY.x.mul(pf.mul(2.1)).add(2.7)).mul(cos(posXY.y.mul(pf.mul(1.7)).add(0.9)));
    const patch = a.mul(0.55).add(b.mul(0.35)).mul(0.5).add(0.5); // ~[0.05, 0.95]
    return clamp(patch.add(delay.mul(0.15)), 0.0, 1.0);
  };

  // Per-particle transition progress: 0 = fully A, 1 = fully B.
  const dieProgress = (posXY, delay) => {
    const dM = dieMoment(posXY, delay);
    return smoothstep(dM.sub(uFadeWindow), dM.add(uFadeWindow), uPhase);
  };

  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: false,
  });

  material.positionNode = Fn(() => {
    const pA = attribute('aPosA');
    const pB = attribute('aPosB');
    const z  = attribute('aZ');
    const seed = attribute('aSeed');
    const delay = seed.z;

    const d = dieProgress(pA, delay);
    const xy = mix(pA, pB, d);

    // crumble peaks at d=0.5, zero at both ends
    const crumble = float(1.0).sub(smoothstep(0.0, 1.0, d.sub(0.5).abs().mul(2.0)));

    // per-particle dispersion direction
    const dirX = seed.x.mul(1.2).add(sin(seed.y.mul(9.0)).mul(0.4));
    const dirY = seed.y.mul(0.4).sub(float(0.35));
    const dirZ = cos(seed.x.mul(7.0).add(seed.y.mul(5.0))).mul(0.6).add(seed.y.mul(0.3));
    const disp = uDispersalAmp.mul(crumble);

    // chaos: turbulent jitter during crumble
    const t = uTime;
    const chX = sin(t.mul(1.4).add(seed.x.mul(7.0))).mul(uChaosAmp).mul(crumble);
    const chY = cos(t.mul(1.1).add(seed.y.mul(6.0))).mul(uChaosAmp.mul(0.7)).mul(crumble);
    const chZ = sin(t.mul(0.9).add(seed.x.add(seed.y).mul(5.0))).mul(uChaosAmp.mul(0.5)).mul(crumble);

    const dx = dirX.mul(disp).add(chX);
    const dy = dirY.mul(disp).sub(crumble.mul(uGravity)).add(chY);
    const dz = dirZ.mul(disp).add(chZ);

    // ambient wave — always on, gentle breathing
    const wavePhase = pA.x.mul(uWaveFreq).add(pA.y.mul(1.5)).add(t.mul(uWaveSpeed));
    const wX = sin(wavePhase).mul(uWaveAmp);
    const wY = cos(wavePhase.mul(0.7)).mul(uWaveAmp.mul(0.8));
    const wZ = sin(wavePhase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.6));

    return vec3(
      xy.x.add(dx).add(wX),
      xy.y.add(dy).add(wY),
      z.add(dz).add(wZ)
    );
  })();

  material.colorNode = Fn(() => {
    const cA = attribute('aColA');
    const cB = attribute('aColB');
    const pA = attribute('aPosA');
    const seed = attribute('aSeed');
    const delay = seed.z;

    const d = dieProgress(pA, delay);
    const c = mix(cA, cB, d);
    const crumble = float(1.0).sub(smoothstep(0.0, 1.0, d.sub(0.5).abs().mul(2.0)));

    const dark = float(1.0).sub(crumble.mul(uDarken));
    const litCol = c.mul(dark);
    const alpha = uOpacity.mul(float(1.0).sub(crumble.mul(0.6)));

    // soft-splat: radial gaussian from point-sprite center
    const pUV = uv();                             // [0,1] over the point sprite
    const r = pUV.sub(0.5).length();              // [0, ~0.7]
    const softMask = smoothstep(0.5, 0.0, r);     // 1 at center, 0 at edges
    const finalAlpha = alpha.mul(softMask);
    return vec4(litCol.x, litCol.y, litCol.z, finalAlpha);
  })();

  material.sizeNode = Fn(() => {
    const pA = attribute('aPosA');
    const seed = attribute('aSeed');
    const delay = seed.z;
    const d = dieProgress(pA, delay);
    const crumble = float(1.0).sub(smoothstep(0.0, 1.0, d.sub(0.5).abs().mul(2.0)));
    const shrink = float(1.0).sub(crumble.mul(uShrink));
    return uSize.mul(seed.w).mul(shrink);
  })();
  material.sizeAttenuation = true;

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  info.textContent = 'painting A';

  window.__u = { uPhase, uPatchScale, uFadeWindow };

  let stateStart = performance.now() / 1000;
  let stateName = 'holdA';  // holdA | AtoB | holdB | BtoA

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') {
      e.preventDefault();
      params.autoLoop = !params.autoLoop;
      if (params.autoLoop) stateStart = performance.now() / 1000;
    } else if (e.code === 'ArrowRight') {
      params.autoLoop = false;
      stateName = 'holdB';
      uPhase.value = 1;
    } else if (e.code === 'ArrowLeft') {
      params.autoLoop = false;
      stateName = 'holdA';
      uPhase.value = 0;
    } else if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen().catch(() => {});
    }
  });

  const clock = new THREE.Clock();
  renderer.setAnimationLoop(() => {
    const t = clock.getElapsedTime();
    uTime.value = t;

    if (params.autoLoop) {
      const now = performance.now() / 1000;
      const dt = now - stateStart;
      const { holdDuration: H, transitionDuration: T } = params;

      if (stateName === 'holdA') {
        uPhase.value = 0;
        if (dt >= H) { stateName = 'AtoB'; stateStart = now; }
      } else if (stateName === 'AtoB') {
        uPhase.value = Math.min(1, dt / T);
        if (dt >= T) { stateName = 'holdB'; stateStart = now; uPhase.value = 1; }
      } else if (stateName === 'holdB') {
        uPhase.value = 1;
        if (dt >= H) { stateName = 'BtoA'; stateStart = now; }
      } else if (stateName === 'BtoA') {
        uPhase.value = Math.max(0, 1 - dt / T);
        if (dt >= T) { stateName = 'holdA'; stateStart = now; uPhase.value = 0; }
      }
    }

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
