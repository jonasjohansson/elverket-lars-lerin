# pointcloud-v5 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build `pointcloud-v5/` — a single stationary soft-splat particle cloud that reads as each of the 9 Serie_1 paintings and sandstorms slowly between them on arrow-key input.

**Architecture:** One persistent pool of ~800k particles, each carrying 9 pre-sampled "addresses" (position + color) packed into two WebGPU data textures. A transition is driven by three uniforms (`uFromIdx`, `uToIdx`, `uT`); the vertex node lerps each particle's address from column `uFromIdx` to column `uToIdx` of the data textures, while a peaked gust envelope (`sin²(πt)`) adds curl-noise displacement and desaturates particle color toward a warm dust tone. Ambient wave drift always runs underneath.

**Tech Stack:** three.js 0.170 WebGPU + TSL, lil-gui, ES module import map (same pattern as `pointcloud-v4/` and `series-v1/`).

**Design doc:** `docs/plans/2026-04-24-pointcloud-v5-design.md`

**Testing note:** This repo's prototypes are visual WebGPU pieces with no automated tests. Each task below ends with a **visual verification step** — open the page in a browser and confirm the described state before committing. Bad TSL math is silent; don't skip verification.

---

## Task 1: Scaffold pointcloud-v5 with empty scene

**Files:**
- Create: `pointcloud-v5/index.html`
- Create: `pointcloud-v5/main.js`

**Step 1: Create `pointcloud-v5/index.html`**

Copy `pointcloud-v4/index.html` verbatim to `pointcloud-v5/index.html`. No changes needed — same `<title>`, canvas, `#info`, and three.js 0.170 import map.

**Step 2: Create `pointcloud-v5/main.js` with a minimal WebGPU scene**

```js
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

async function init() {
  await renderer.init();
  info.textContent = 'pointcloud-v5 · empty · WebGPU';
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
```

**Step 3: Start a local server if one isn't running**

```bash
python3 -m http.server 8000
```

Visit `http://localhost:8000/pointcloud-v5/`. Expect a black page with `pointcloud-v5 · empty · WebGPU` in the top-left. No errors in devtools console.

**Step 4: Commit**

```bash
git add pointcloud-v5/index.html pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): scaffold empty WebGPU scene"
```

---

## Task 2: Load 9 paintings and sample per-painting addresses

Not yet rendered. Goal: for each painting, produce an `{ pos: Float32Array(N*2), col: Float32Array(N*3), worldW, worldH }` object. Particle count is same across all 9 (required so a single geometry can re-address them).

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add constants and image helpers at the top of `main.js`**

Before `async function init()`:

```js
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
```

**Step 2: Load all 9 paintings in `init()` and log**

Replace the body of `init()` before `setAnimationLoop` with:

```js
await renderer.init();
info.textContent = `Loading ${NUM_PAINTINGS} paintings...`;

const images = await Promise.all(PAINTINGS.map(loadImage));
const pixels = images.map(img => getPixels(img));

info.textContent = `Sampling ${PARTICLE_COUNT / 1000}k × ${NUM_PAINTINGS} addresses...`;
await new Promise(r => requestAnimationFrame(r));

const samples = pixels.map(px => samplePainting(px, PARTICLE_COUNT));

console.log('painting dims:', samples.map(s => `${s.worldW.toFixed(2)}×${s.worldH.toFixed(2)}`));
console.log('total address bytes:',
  samples.length * (samples[0].pos.byteLength + samples[0].col.byteLength));

info.textContent = `${NUM_PAINTINGS} paintings · ${(PARTICLE_COUNT / 1e6).toFixed(2)}M particles · WebGPU`;
```

**Step 3: Visual verification**

Reload. Console should show:
- An array of 9 dimension strings like `["1.43×2.00", "1.98×2.00", ...]`
- `total address bytes: ~144000000` (≈144 MB CPU-side; will be halved when packed to half-float textures in Task 3, but fine in RAM for now)
- No errors

The `#info` line should read `9 paintings · 0.80M particles · WebGPU`.

**Step 4: Remove the two `console.log` lines.** Keep everything else.

**Step 5: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): load 9 trim paintings and sample per-painting addresses"
```

---

## Task 3: Pack addresses into data textures

Build two `DataTexture`s with width=9 (one column per painting) and height=PARTICLE_COUNT (one row per particle): `posTex` (RG float) and `colTex` (RGB float).

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add a packing helper**

Before `init()`:

```js
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
```

Note: `RGBFormat` has been deprecated in newer three.js; at 0.170 it still works. If you hit a warning, swap to `RGBAFormat` and write a 4th (alpha=1) channel.

**Step 2: Call it in `init()`**

After the `samples = pixels.map(...)` line:

```js
const { posTex, colTex } = packAddressTextures(samples);
console.log('packed:', posTex.image.width, '×', posTex.image.height, 'pos +', colTex.image.width, '×', colTex.image.height, 'col');
```

**Step 3: Visual verification**

Reload. Console should log `packed: 9 × 800000 pos + 9 × 800000 col`. No errors.

**Step 4: Remove the `console.log`.** Keep everything else.

**Step 5: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): pack addresses into 9×N data textures"
```

---

## Task 4: Render painting 0 (static, read from data textures)

No transition yet. Just confirm the vertex node can fetch a position and color from the data textures, given a particle row index.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add TSL imports at the top**

```js
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn } from 'three/tsl';
```

**Step 2: Build geometry and material after `packAddressTextures` call**

```js
// Geometry: one vertex per particle. Positions are a placeholder (0); the real
// position is fetched from posTex in the vertex node.
const positions = new Float32Array(PARTICLE_COUNT * 3); // all zeros
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

// Uniforms
const uFromIdx = uniform(0.0);
const uToIdx   = uniform(0.0);
const uT       = uniform(0.0);
const uNumPaintings = uniform(NUM_PAINTINGS);
const uNumParticles = uniform(PARTICLE_COUNT);
const uSize    = uniform(5.0);

// TSL helper: texel UV for (col, row) on a data texture of size cols × rows
// with NearestFilter. We center the sample on the texel.
function texelUV(col, row, cols, rows) {
  return vec2(
    col.add(0.5).div(cols),
    row.add(0.5).div(rows),
  );
}

const material = new THREE.PointsNodeMaterial({
  transparent: true,
  blending: THREE.NormalBlending,
  depthWrite: false,
});

material.positionNode = Fn(() => {
  const idx = attribute('aIndex');
  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const posFrom = texture(posTex, fromUV).xy;
  return vec3(posFrom.x, posFrom.y, float(0.0));
})();

material.colorNode = Fn(() => {
  const idx = attribute('aIndex');
  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const colFrom = texture(colTex, fromUV).xyz;
  return vec4(colFrom, 1.0);
})();

material.sizeNode = uSize;
material.sizeAttenuation = true;

const points = new THREE.Points(geometry, material);
scene.add(points);
```

**Step 3: Visual verification**

Reload. Painting 0 (`17_RESAN_OCH_ORIENTEN_LL_0012-trim.png`) should render as a 0.8M-particle cloud, centered, roughly filling the viewport. Colors should match the painting.

If the points are too small or too big, tweak `uSize` default (range 3–10). If you see only a handful of points, the data texture isn't being sampled correctly — check console for shader errors and verify `texelUV` math.

**Step 4: Sanity-check painting switch via console**

Open devtools and run:

```js
// Directly poke the uniform — should change the rendered painting
// Find via: uFromIdx is captured in init's closure. Temporarily expose it:
```

Temporarily add `window.__u = { uFromIdx };` right after the uniform is defined. Reload. In console:

```js
__u.uFromIdx.value = 1;   // should show painting 0013
__u.uFromIdx.value = 4;   // should show painting 0022
__u.uFromIdx.value = 0;   // back to 0012
```

Each should render the correct painting cleanly.

**Step 5: Remove the `window.__u = …` line.**

**Step 6: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): render painting from data-texture address lookup"
```

---

## Task 5: Ambient wave drift

Add the v4/series-v1 two-wave drift so the still image gently breathes.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add wave uniforms and the params object**

Above the other uniforms:

```js
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

const uTime        = uniform(0.0);
const uWaveAmp     = uniform(params.waveAmplitude);
const uWaveSpd     = uniform(params.waveSpeed);
const uWaveFreq    = uniform(params.waveFrequency);
const uWave2Amp    = uniform(params.wave2Amplitude);
const uWave2Spd    = uniform(params.wave2Speed);
const uWave2Freq   = uniform(params.wave2Frequency);
const uMicroDrift  = uniform(params.microDrift);
```

Set the initial `uSize` from `params.splatScale` so they stay in sync:

```js
const uSize = uniform(params.splatScale);
```

**Step 2: Add `sin, cos` to the TSL import**

```js
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn, sin, cos } from 'three/tsl';
```

**Step 3: Extend `positionNode` with waves**

Replace the existing `positionNode` body:

```js
material.positionNode = Fn(() => {
  const idx  = attribute('aIndex');
  const seed = attribute('aSeed');
  const seedX = seed.x, seedY = seed.y;
  const t = uTime;

  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const posFrom = texture(posTex, fromUV).xy;
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
```

**Step 4: Advance `uTime` in the animation loop**

Replace `setAnimationLoop(() => { ... })` with:

```js
renderer.setAnimationLoop(() => {
  uTime.value = performance.now() / 1000;
  controls.update();
  renderer.render(scene, camera);
});
```

**Step 5: Visual verification**

Reload. Painting 0 is still fully readable but now shimmers gently — particles drift ~1% of the painting's width. Completely subtle. If you can't see any motion, bump `params.waveAmplitude` to 0.04 temporarily, confirm the wave, then revert to 0.015.

**Step 6: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): ambient two-wave drift"
```

---

## Task 6: Target lerp between from and to addresses

Add a `uT` slider that linearly morphs each particle from its `uFromIdx` address to its `uToIdx` address. No scatter or color dwell yet — just geometric interpolation.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add `mix, smoothstep` to the TSL import**

```js
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn, sin, cos, mix, smoothstep } from 'three/tsl';
```

**Step 2: Extend `positionNode` to lerp from/to**

Replace the `fromUV`/`posFrom` chunk with a from+to lookup and a lerp:

```js
const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
const toUV   = texelUV(uToIdx,   idx, uNumPaintings, uNumParticles);
const posFrom = texture(posTex, fromUV).xy;
const posTo   = texture(posTex, toUV).xy;

// Ease the lerp so ends are calm, middle is fastest
const tEased = smoothstep(float(0.0), float(1.0), uT);
const posXY = mix(posFrom, posTo, tEased);
const pos = vec3(posXY.x, posXY.y, float(0.0));
```

**Step 3: Extend `colorNode` similarly (linear crossfade for now)**

```js
material.colorNode = Fn(() => {
  const idx = attribute('aIndex');
  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const toUV   = texelUV(uToIdx,   idx, uNumPaintings, uNumParticles);
  const colFrom = texture(colTex, fromUV).xyz;
  const colTo   = texture(colTex, toUV).xyz;
  const col = mix(colFrom, colTo, uT);
  return vec4(col, 1.0);
})();
```

**Step 4: Add a temporary debug slider via a minimal GUI harness**

Add `lil-gui` import at the top:

```js
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';
```

At the end of `init()` before `setAnimationLoop`:

```js
const debugParams = { fromIdx: 0, toIdx: 1, t: 0 };
const gui = new GUI({ title: 'pointcloud-v5 (debug)' });
gui.add(debugParams, 'fromIdx', 0, NUM_PAINTINGS - 1, 1).name('From').onChange(v => uFromIdx.value = v);
gui.add(debugParams, 'toIdx',   0, NUM_PAINTINGS - 1, 1).name('To').onChange(v => uToIdx.value = v);
gui.add(debugParams, 't', 0, 1, 0.001).name('t').onChange(v => uT.value = v);
```

**Step 5: Visual verification**

Reload. Default: painting 0 ↔ painting 1, t = 0 shows painting 0.

- Drag `t` from 0 to 1 slowly: each particle should travel smoothly from its address in painting 0 to its address in painting 1. Colors crossfade linearly. At t=0.5 you should see an uncanny hybrid — limbs of painting 0 morphing into painting 1.
- Change `To` to 5, set `t = 1` — painting 5 appears cleanly. Set `t = 0.5`: hybrid of 0 and 5.
- Change `From` to 3, `To` to 8, `t=0.5`: hybrid of 3 and 8.

This is the geometric backbone of the transition. It should feel mechanical and direct — no storm yet.

**Step 6: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): from→to target lerp driven by uT"
```

---

## Task 7: Gust scatter — curl noise with peaked envelope

Layer a mid-transition wind displacement on top of the lerp. Amplitude peaks at t=0.5 and is zero at both ends.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add gust uniforms**

Near the other params/uniforms:

```js
// In params:
  gustAmplitude: 0.15,
  gustChaos: 0.04,
  gustDirectionX: -1.0,  // leftward wind
  gustDirectionZ: 0.4,

// As uniforms:
const uGustAmp = uniform(params.gustAmplitude);
const uGustChaos = uniform(params.gustChaos);
const uGustDirX = uniform(params.gustDirectionX);
const uGustDirZ = uniform(params.gustDirectionZ);
```

**Step 2: Extend `positionNode` with gust term**

Between the `pos` definition and the `wave*` terms, insert:

```js
// Gust envelope: sin²(π·t), zero at 0 and 1, peaks at 0.5
const piT = uT.mul(3.14159);
const gustEnv = sin(piT).mul(sin(piT));   // sin²(πt)

// Low-frequency curl-ish displacement based on pos and a slow time
const gustPhaseX = pos.x.mul(2.3).add(pos.y.mul(1.1)).add(t.mul(0.4));
const gustPhaseY = pos.x.mul(1.7).sub(pos.y.mul(2.0)).add(t.mul(0.3));
const gustX = sin(gustPhaseX).mul(uGustAmp).mul(0.8).add(uGustDirX.mul(uGustAmp).mul(0.6));
const gustY = cos(gustPhaseY).mul(uGustAmp).mul(0.5);
const gustZ = sin(gustPhaseX.mul(0.7).add(1.3)).mul(uGustAmp).mul(0.35).add(uGustDirZ.mul(uGustAmp).mul(0.3));

// High-frequency per-particle chaos
const chaosX = sin(t.mul(1.4).add(seedX.mul(9.0))).mul(uGustChaos);
const chaosY = cos(t.mul(1.1).add(seedY.mul(8.0))).mul(uGustChaos);
const chaosZ = sin(t.mul(0.9).add(seedX.add(seedY).mul(7.0))).mul(uGustChaos.mul(0.7));

const gX = gustX.add(chaosX).mul(gustEnv);
const gY = gustY.add(chaosY).mul(gustEnv);
const gZ = gustZ.add(chaosZ).mul(gustEnv);
```

Update the final `return pos.add(vec3(...))` to include the gust:

```js
return pos.add(vec3(
  w1x.add(w2x).add(dx).add(gX),
  w1y.add(w2y).add(dy).add(gY),
  w1z.add(w2z).add(gZ),
));
```

**Step 3: Visual verification**

Reload. Drag `t` from 0 to 1:
- **t = 0**: painting 0 still, ambient drift only
- **t = 0.25**: painting starts morphing toward 1, with a growing leftward wind bending the particle paths
- **t = 0.5**: peak — the cloud looks blown by wind, shapes indistinct, particles streaming
- **t = 0.75**: wind dying, painting 1 starting to re-solidify
- **t = 1**: painting 1 still, no wind

If gust is too strong and washes out shape entirely, drop `params.gustAmplitude` to 0.10. Too weak: raise to 0.22.

**Step 4: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): curl-noise gust with peaked envelope"
```

---

## Task 8: Color dwell — desaturate toward dust at peak

At the peak of the storm, particles should briefly lose their image color and take on a warm dust tone, then pick up the destination color as the wind dies.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add dust uniforms**

```js
// In params:
  dustMixStrength: 0.55,
  dustColor: '#8c8070',

// As uniforms:
const uDustMix = uniform(params.dustMixStrength);
const uDustColor = uniform(new THREE.Color(params.dustColor));
```

**Step 2: Rewrite `colorNode` with dwell and dust mix**

```js
material.colorNode = Fn(() => {
  const idx = attribute('aIndex');
  const seed = attribute('aSeed');
  const delay = seed.z;

  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const toUV   = texelUV(uToIdx,   idx, uNumPaintings, uNumParticles);
  const colFrom = texture(colTex, fromUV).xyz;
  const colTo   = texture(colTex, toUV).xyz;

  // Dwell: each particle holds its From color until ~(0.4 + 0.3·delay),
  // then hands over by ~(0.6 + 0.3·delay). Delay staggers particles so
  // the color transition front sweeps, not pops.
  const cLo = float(0.4).add(delay.mul(0.3));
  const cHi = float(0.6).add(delay.mul(0.3));
  const cT = smoothstep(cLo, cHi, uT);
  const colInterp = mix(colFrom, colTo, cT);

  // Dust mix peaks at t=0.5 (sin³πt is sharper than sin²πt)
  const piT = uT.mul(3.14159);
  const sinT = sin(piT);
  const dustEnv = sinT.mul(sinT).mul(sinT);   // sin³(πt), peaked ≈ 1 at t=0.5

  const col = mix(colInterp, uDustColor, dustEnv.mul(uDustMix));
  return vec4(col, 1.0);
})();
```

**Step 3: Visual verification**

Reload. Drag `t` from 0 to 1 slowly:
- **t = 0.3**: painting 0 still mostly visible but starting to desaturate
- **t = 0.5**: image nearly erased — the cloud is a shifting mass of warm dust
- **t = 0.7**: painting 1 colors emerging from the dust

The emotional beat: when you pass through t=0.5, you shouldn't be able to read *either* painting — you're inside the storm. Then it clears. If the dust is too dominant, lower `params.dustMixStrength` to 0.4. If not strong enough, 0.7.

**Step 4: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): desaturate toward dust at storm peak"
```

---

## Task 9: Per-particle stagger in position as well

The color dwell already staggers via `delay`. Apply a smaller stagger to the position lerp so particles don't arrive at their destinations in lockstep.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Stagger the position lerp**

In `positionNode`, replace the position lerp with a per-particle-shifted version:

```js
const delay = seed.z;              // 0..1
// Shift this particle's effective uT by ±0.1 based on delay
const tLocal = uT.add(delay.sub(0.5).mul(0.2)).clamp(0, 1);
const tEased = smoothstep(float(0.0), float(1.0), tLocal);
const posXY = mix(posFrom, posTo, tEased);
```

`.clamp(0,1)` assumes TSL exposes it; if not, use `clamp(tLocal, float(0), float(1))` with `clamp` imported.

Add `clamp` to the TSL import if needed:

```js
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn, sin, cos, mix, smoothstep, clamp } from 'three/tsl';
```

And use `clamp(uT.add(delay.sub(0.5).mul(0.2)), float(0), float(1))`.

**Step 2: Visual verification**

Reload. Drag `t` slowly. The geometric transition should feel looser — some particles arrive early, some lag, giving the storm a front-and-back quality. Subtle but noticeable.

**Step 3: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): per-particle position stagger"
```

---

## Task 10: Arrow-key transition state machine

Replace the manual `t` slider with arrow keys that drive a 12-second eased transition.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Add transition state and key handler**

Before `setAnimationLoop`:

```js
const transition = {
  duration: 12.0,     // seconds
  from: 0,
  to: 0,
  start: 0,
  active: false,
};

function startTransition(toIdx) {
  if (transition.active) return;  // ignore input mid-storm
  const clamped = Math.max(0, Math.min(NUM_PAINTINGS - 1, toIdx));
  if (clamped === transition.to) return;
  transition.from = transition.to;
  transition.to = clamped;
  transition.start = performance.now();
  transition.active = true;
  uFromIdx.value = transition.from;
  uToIdx.value = transition.to;
  uT.value = 0;
}

window.addEventListener('keydown', (e) => {
  if (e.key === 'ArrowRight') startTransition(transition.to + 1);
  if (e.key === 'ArrowLeft')  startTransition(transition.to - 1);
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  }
  if (e.code === 'Space') { e.preventDefault(); controls.enabled = !controls.enabled; }
});
```

**Step 2: Drive `uT` from the animation loop**

Replace the animation loop with:

```js
renderer.setAnimationLoop(() => {
  uTime.value = performance.now() / 1000;
  if (transition.active) {
    const elapsed = (performance.now() - transition.start) / 1000;
    const t = Math.min(1, elapsed / transition.duration);
    uT.value = t;
    if (t >= 1) {
      // Settle: from = to so a subsequent transition starts from the new resting painting
      uFromIdx.value = transition.to;
      uT.value = 0;
      transition.from = transition.to;
      transition.active = false;
    }
  }
  controls.update();
  renderer.render(scene, camera);
});
```

**Step 3: Drop the debug GUI sliders**

Remove the debug `gui.add(debugParams, 'fromIdx'...)`, `toIdx`, and `t` lines. Keep the `GUI` import — we'll use it properly in Task 12.

**Step 4: Visual verification**

Reload. Press **Right arrow** once. Over the next 12 seconds:
- painting 0 drifts, then a sandstorm builds, then clears into painting 1
- pressing Right during the storm does nothing (input ignored)
- after ~12s, painting 1 is still and ready
- pressing **Right** again — storms to painting 2
- pressing **Left** — storms back to painting 1
- **Left** at painting 0 stays at painting 0 (clamped)

`F` toggles fullscreen. `Space` toggles OrbitControls for inspection.

**Step 5: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): arrow-key transition state machine"
```

---

## Task 11: Soft-splat look — radial alpha gradient

Points currently render as small opaque squares. Make each point a soft radial gaussian by sampling a procedural alpha gradient.

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Build a 64×64 canvas gradient texture once**

Above `init()`:

```js
function makeSoftDiscTexture(size = 64) {
  const c = document.createElement('canvas');
  c.width = size; c.height = size;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(size / 2, size / 2, 0, size / 2, size / 2, size / 2);
  g.addColorStop(0.0, 'rgba(255,255,255,1.0)');
  g.addColorStop(0.4, 'rgba(255,255,255,0.55)');
  g.addColorStop(0.8, 'rgba(255,255,255,0.08)');
  g.addColorStop(1.0, 'rgba(255,255,255,0.0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, size, size);
  const tex = new THREE.CanvasTexture(c);
  tex.needsUpdate = true;
  return tex;
}
```

**Step 2: Create the texture and sample it in `colorNode`**

After the `colTex` is packed and uniforms set up, add:

```js
const softDisc = makeSoftDiscTexture(64);
```

Add `uv` to the TSL import:

```js
import { attribute, uniform, texture, vec2, vec3, vec4, float, Fn, sin, cos, mix, smoothstep, clamp, uv } from 'three/tsl';
```

Add an opacity uniform:

```js
const uOpacity = uniform(params.opacity);
```

Modify `colorNode` to multiply alpha by the soft-disc sample and the opacity:

```js
material.colorNode = Fn(() => {
  const idx = attribute('aIndex');
  const seed = attribute('aSeed');
  const delay = seed.z;

  const fromUV = texelUV(uFromIdx, idx, uNumPaintings, uNumParticles);
  const toUV   = texelUV(uToIdx,   idx, uNumPaintings, uNumParticles);
  const colFrom = texture(colTex, fromUV).xyz;
  const colTo   = texture(colTex, toUV).xyz;

  const cLo = float(0.4).add(delay.mul(0.3));
  const cHi = float(0.6).add(delay.mul(0.3));
  const cT = smoothstep(cLo, cHi, uT);
  const colInterp = mix(colFrom, colTo, cT);

  const piT = uT.mul(3.14159);
  const sinT = sin(piT);
  const dustEnv = sinT.mul(sinT).mul(sinT);
  const col = mix(colInterp, uDustColor, dustEnv.mul(uDustMix));

  // Soft radial alpha from disc texture — uv() of a point is the point-sprite UV (0..1).
  const discAlpha = texture(softDisc, uv()).w;

  return vec4(col, uOpacity.mul(discAlpha));
})();
```

**Step 3: Scale size up to match the soft look**

Soft splats need bigger points than hard points to read. Bump default:

```js
// in params:
splatScale: 7.0,
```

(remember to replace the old 5.0 default)

**Step 4: Visual verification**

Reload. Each particle is now a soft fuzzy dot. Overlapping soft splats blend into continuous tone — the painting reads smooth and painterly, not as a mesh of hard pixels.

- Zoom in (press Space, drag with mouse): you should see individual soft blobs.
- Zoom out (Space again to lock): the painting's forms resolve cleanly.

If particles look too puffy and the image is blurry, drop `splatScale` to 5 or 6. Too grainy: bump to 9. If the page looks washed out, lower `params.opacity` to 0.75.

**Performance check**: open devtools → Performance monitor, take a quick snapshot. Should hold 60 fps at 1440p. If it tanks to 30, drop `PARTICLE_COUNT` to 500_000 at the top of the file.

**Step 5: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): soft radial alpha for gaussian splat look"
```

---

## Task 12: GUI panel for tuning

**Files:**
- Modify: `pointcloud-v5/main.js`

**Step 1: Build the GUI after init setup**

Replace the removed debug GUI block (near the bottom of `init()`) with:

```js
const gui = new GUI({ title: 'pointcloud-v5' });

const fR = gui.addFolder('Rendering');
fR.add(params, 'splatScale', 2, 14, 0.1).name('Splat Scale').onChange(v => uSize.value = v);
fR.add(params, 'opacity', 0, 1, 0.01).name('Opacity').onChange(v => uOpacity.value = v);
fR.addColor(params, 'bgColor').name('Background').onChange(v => renderer.setClearColor(v));

const fW = gui.addFolder('Ambient');
fW.add(params, 'waveAmplitude', 0, 0.05, 0.001).name('Wave Amp').onChange(v => uWaveAmp.value = v);
fW.add(params, 'waveSpeed', 0, 1, 0.01).name('Wave Speed').onChange(v => uWaveSpd.value = v);
fW.add(params, 'wave2Amplitude', 0, 0.05, 0.001).name('Wave 2 Amp').onChange(v => uWave2Amp.value = v);
fW.add(params, 'microDrift', 0, 0.01, 0.0005).name('Micro Drift').onChange(v => uMicroDrift.value = v);

const fT = gui.addFolder('Transition');
fT.add(transition, 'duration', 4, 30, 0.5).name('Duration (s)');
fT.add(params, 'gustAmplitude', 0, 0.4, 0.005).name('Gust Amp').onChange(v => uGustAmp.value = v);
fT.add(params, 'gustChaos', 0, 0.12, 0.002).name('Gust Chaos').onChange(v => uGustChaos.value = v);
fT.add(params, 'gustDirectionX', -1.5, 1.5, 0.05).name('Wind X').onChange(v => uGustDirX.value = v);
fT.add(params, 'gustDirectionZ', -1.5, 1.5, 0.05).name('Wind Z').onChange(v => uGustDirZ.value = v);
fT.add(params, 'dustMixStrength', 0, 1, 0.02).name('Dust Mix').onChange(v => uDustMix.value = v);
fT.addColor(params, 'dustColor').name('Dust Color').onChange(v => uDustColor.value.set(v));

const actions = {
  next: () => startTransition(transition.to + 1),
  prev: () => startTransition(transition.to - 1),
  toFirst: () => {
    transition.from = transition.to = 0;
    uFromIdx.value = 0;
    uToIdx.value = 0;
    uT.value = 0;
    transition.active = false;
  },
};
gui.add(actions, 'prev').name('← Prev');
gui.add(actions, 'next').name('→ Next');
gui.add(actions, 'toFirst').name('⟲ Reset to painting 0');
```

**Step 2: Visual verification**

Reload. GUI appears top-right.
- Click `→ Next` — same as Right arrow.
- Drag `Splat Scale` — points get bigger/smaller.
- Drag `Gust Amp` to 0.05 then click `→ Next` — visibly weaker storm.
- Drag `Duration` to 4 and `→ Next` — fast transition (still eased, but brisk).
- Drag `Duration` to 24 and `→ Next` — very slow, meditative.
- Change `Dust Color` — the mid-storm tone changes.
- Click `⟲ Reset to painting 0` — snaps back immediately.

**Step 3: Commit**

```bash
git add pointcloud-v5/main.js
git commit -m "feat(pointcloud-v5): GUI panel for tuning knobs"
```

---

## Task 13: Link from root index.html

**Files:**
- Modify: `index.html`

**Step 1: Add v5 to the Point cloud section**

In the root `index.html`, add above the v4 entry inside the `<h2>Point cloud</h2>` list:

```html
    <li><a href="./pointcloud-v5/"><span>v5 — soft-splat sandstorm across 9 Serie_1 paintings</span><span class="slug">pointcloud-v5</span></a></li>
```

**Step 2: Visual verification**

Visit `http://localhost:8000/`. The new v5 row appears at the top of the Point cloud list. Click it — it loads `pointcloud-v5/`.

**Step 3: Commit**

```bash
git add index.html
git commit -m "docs(index): link pointcloud-v5 prototype"
```

---

## Task 14: Tuning pass

Open `pointcloud-v5/` in a fresh reload. Walk through all 9 paintings with Right arrow. Watch for:

- Idle paintings read clearly and are visibly painterly — soft splats overlap into continuous tone
- Ambient drift is present but unobtrusive — you notice it only if you look
- Transitions feel like a physical weather event, not an alpha crossfade — dust peaks around t=0.5, image dissolves and reforms
- At the storm peak, neither the From nor the To painting is legible — it's genuinely "in the storm"
- No stutters; 60 fps throughout
- `F` fullscreen works and the piece fills the screen cleanly

Adjust defaults in `params` for the look you want. Common tweaks:
- If storms are too mild: raise `gustAmplitude` to 0.18, `dustMixStrength` to 0.65
- If storms are too chaotic: drop `gustChaos` to 0.02, `gustAmplitude` to 0.10
- If you want slower storms: bump `transition.duration` default to 16 or 20
- If the image reads too soft/blurry: drop `splatScale` to 5.5 and bump `opacity` to 1.0
- If image edges look hard: raise `splatScale` to 8 and lower `opacity` to 0.8

Commit the tuned defaults:

```bash
git add pointcloud-v5/main.js
git commit -m "tune(pointcloud-v5): exhibition defaults"
```

---

## Deferred (not v5)

- Depth / pseudo-3D from luminance — the design consciously avoided this; revisit only if the flat read feels thin.
- Mouse interaction — cursor wither/vortex from v4 is cut deliberately; could come back as a "disturb the cloud" gesture later.
- Auto-advance cycling — manual only per design.
- Additional painting series — the 9-address packing generalizes; swap the `PAINTING_IDS` list.
- Post-processing stack (DOF, trails, vignette) — add only if the raw render feels under-cooked.
- Multi-step snaps (jump from painting 1 to painting 5 in one gesture) — out of scope.
