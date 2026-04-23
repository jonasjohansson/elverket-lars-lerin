# pointcloud-v6 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a WebGPU particle cloud that disintegrates between two Lars Lerin trim PNGs on an auto-loop. Disintegration *is* the transition.

**Architecture:** Single `THREE.Points` mesh of ~800k particles; each particle carries address+color for both paintings (A, B) as geometry attributes. A per-particle clustered "death moment" `dM` drives a smoothstep crossfade against a global `uPhase`. A state machine animates `uPhase` through `holdA → AtoB → holdB → BtoA` on a loop, with a GUI override. No cursor interaction. No post-processing.

**Tech Stack:** three.js 0.170 (WebGPU renderer + TSL node material), lil-gui, vanilla ES modules via importmap.

**Design doc:** `docs/plans/2026-04-24-pointcloud-v6-design.md`

**Reference code:** `pointcloud-v4/main.js` (disintegration mechanic + soft-splat style, ambient wave), `pointcloud-v5/main.js` (trim-PNG loader + alpha sampling).

**Testing model:** This is a visual project with no test framework. Each task ends with a manual verification step: open `pointcloud-v6/index.html` in the browser and confirm the described visual state. Do NOT claim a task done unless you have visually verified it (or explicitly noted that it cannot be verified without the dev server running).

**Commit hygiene:** Commit after every task. Use the conventional `feat(pointcloud-v6):` / `fix(pointcloud-v6):` / `chore(pointcloud-v6):` scope.

---

## Task 1: Scaffold empty WebGPU scene

**Files:**
- Create: `pointcloud-v6/index.html`
- Create: `pointcloud-v6/main.js`

**Step 1: Create index.html**

Copy `pointcloud-v5/index.html` verbatim into `pointcloud-v6/index.html`. (Same title, same importmap, same canvas+info markup.)

**Step 2: Create a minimal main.js**

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
controls.target.set(0, 0, 0);

async function init() {
  await renderer.init();
  info.textContent = 'pointcloud-v6 scaffold';
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

**Step 3: Verify**

Serve the repo root (e.g. `python3 -m http.server` or equivalent). Open `http://localhost:<port>/pointcloud-v6/`. Expected: black screen, "pointcloud-v6 scaffold" text, no console errors.

**Step 4: Commit**

```bash
git add pointcloud-v6/
git commit -m "feat(pointcloud-v6): scaffold empty WebGPU scene"
```

---

## Task 2: Load both paintings and sample particles

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add loader + sampling utilities**

Above `init()`, add:

```js
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
```

**Step 2: Call them from init()**

Inside `init()` after `await renderer.init()`:

```js
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
```

**Step 3: Verify**

Reload. Expected: still black screen, info text updates through "Loading..." → "Sampling..." → "0.80M particles sampled". Console logs show two aspect-correct world sizes (both paintings should report roughly similar widths around 3.0-3.1).

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): load and sample both trim paintings"
```

---

## Task 3: Render painting A as flat points

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add TSL imports**

Top of file:

```js
import { attribute, uniform, float, vec2, vec3, vec4, Fn, mix, sin, cos, smoothstep, clamp } from 'three/tsl';
```

**Step 2: Build geometry + basic material in init()**

After sampling, before the render loop:

```js
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
```

**Step 3: Verify**

Reload. Expected: painting 0012 (sparse desert with figures, warm sky) visible as a dense point cloud, roughly filling the viewport. Orbit with mouse works. No painting B content visible yet.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): render painting A as flat points"
```

---

## Task 4: Add random Z scatter for cloud thickness

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Fill `aZ`**

Replace the `aZ[i] = 0.0;` line inside the per-particle loop with:

```js
aZ[i] = (Math.random() - 0.5) * 0.4;  // [-0.2, +0.2] default scatter
```

**Step 2: Use it in positionNode**

Update `positionNode`:

```js
material.positionNode = Fn(() => {
  const pA = attribute('aPosA');
  const z  = attribute('aZ');
  return vec3(pA.x, pA.y, z);
})();
```

**Step 3: Verify**

Reload. Orbit the camera (drag slowly). Expected: the cloud now has visible thickness — you can see particles at different depths as you rotate, no longer a flat plane. Painting still reads as 0012 from head-on.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): random Z scatter gives the cloud thickness"
```

---

## Task 5: Wire uPhase uniform + render B when phase = 1

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add uPhase + switch via mix**

Add near `uSize`:

```js
const uPhase = uniform(0.0);
```

Replace `positionNode` and `colorNode`:

```js
material.positionNode = Fn(() => {
  const pA = attribute('aPosA');
  const pB = attribute('aPosB');
  const z  = attribute('aZ');
  const xy = mix(pA, pB, uPhase);
  return vec3(xy.x, xy.y, z);
})();

material.colorNode = Fn(() => {
  const cA = attribute('aColA');
  const cB = attribute('aColB');
  const c  = mix(cA, cB, uPhase);
  return vec4(c.x, c.y, c.z, 1.0);
})();
```

**Step 2: Expose uPhase on window for manual testing**

Just before the render loop:

```js
window.__u = { uPhase };
```

**Step 3: Verify**

Reload. Expected: painting A. In the browser devtools console, run `__u.uPhase.value = 1; __u.uPhase.value` — the cloud should linearly morph into painting B (0029 ruins). Try 0.5 for an intermediate state (every particle halfway between its A and B positions — will look mushy, that's expected). Set back to 0 to return to A.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): uPhase uniform drives linear A→B crossfade"
```

---

## Task 6: Replace linear crossfade with clustered dieProg

**Files:**
- Modify: `pointcloud-v6/main.js`

This is the core mechanic. `dieProg` is a per-particle smoothstep gated by a spatially clustered death moment `dM`.

**Step 1: Add disintegration uniforms**

Near `uPhase`:

```js
const uPatchScale = uniform(3.0);
const uFadeWindow = uniform(0.08);
```

**Step 2: Add a TSL helper above the material definition**

```js
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
```

**Step 3: Use dieProgress in position and color nodes**

Replace `positionNode` and `colorNode`:

```js
material.positionNode = Fn(() => {
  const pA = attribute('aPosA');
  const pB = attribute('aPosB');
  const z  = attribute('aZ');
  const seed = attribute('aSeed');
  const delay = seed.z;
  // Patch is anchored in A-space so clusters are stable.
  const d = dieProgress(pA, delay);
  const xy = mix(pA, pB, d);
  return vec3(xy.x, xy.y, z);
})();

material.colorNode = Fn(() => {
  const cA = attribute('aColA');
  const cB = attribute('aColB');
  const pA = attribute('aPosA');
  const seed = attribute('aSeed');
  const delay = seed.z;
  const d = dieProgress(pA, delay);
  const c = mix(cA, cB, d);
  return vec4(c.x, c.y, c.z, 1.0);
})();
```

**Step 4: Verify**

Reload. In the console, slowly ramp `__u.uPhase.value` from 0 to 1 in steps of 0.1. Expected: painting A transitions to B not as a uniform fade but as **clustered patches** — some regions switch early, others late. At 0.5 you should see a visible mosaic of A-patches and B-patches.

Tune patch size by trying `window.__u.uPatchScale = uPatchScale; __u.uPatchScale.value = 6` (smaller patches) and `1.5` (larger patches). Default 3 should give medium patches. Add `uPatchScale` to the `window.__u` object so you can tune live.

**Step 5: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): clustered patchy death drives A↔B transition"
```

---

## Task 7: Add crumble envelope displacement

**Files:**
- Modify: `pointcloud-v6/main.js`

Particles should scatter outward and downward during their death window, snap clean on both sides.

**Step 1: Add uniforms + uTime**

Near the other uniforms:

```js
const uTime = uniform(0.0);
const uDispersalAmp = uniform(0.35);
const uGravity = uniform(0.25);
const uChaosAmp = uniform(0.18);
```

**Step 2: Drive uTime from the animation loop**

In the `setAnimationLoop` callback (or replace it):

```js
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime();
  uTime.value = t;
  controls.update();
  renderer.render(scene, camera);
});
```

**Step 3: Update positionNode with crumble displacement**

Replace `positionNode`:

```js
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

  return vec3(xy.x.add(dx), xy.y.add(dy), z.add(dz));
})();
```

**Step 4: Verify**

Reload. Slowly ramp `__u.uPhase.value` from 0 to 1. Expected: as each patch crosses its death moment, those particles scatter outward and settle — not a clean mosaic anymore but visible *crumbling* into dust and back into shape. At both ends (0 and 1) the cloud is clean. Around the patch edges you should see a gust/debris feel.

**Step 5: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): crumble envelope scatters dying particles"
```

---

## Task 8: Add darken + fade during crumble

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add uniforms**

```js
const uDarken = uniform(0.7);
const uOpacity = uniform(0.9);
```

**Step 2: Update colorNode**

Replace:

```js
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

  return vec4(litCol.x, litCol.y, litCol.z, alpha);
})();
```

**Step 3: Verify**

Reload. Ramp `__u.uPhase.value` manually. Expected: crumbling patches now also darken and fade — they rot as they die. You should see a visible dimming during each patch's transition, brighter on both ends.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): darken and fade during crumble peak"
```

---

## Task 9: Add size shrink during crumble

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add uniform**

```js
const uShrink = uniform(0.85);
```

**Step 2: Replace sizeNode**

```js
material.sizeNode = Fn(() => {
  const pA = attribute('aPosA');
  const seed = attribute('aSeed');
  const delay = seed.z;
  const d = dieProgress(pA, delay);
  const crumble = float(1.0).sub(smoothstep(0.0, 1.0, d.sub(0.5).abs().mul(2.0)));
  const shrink = float(1.0).sub(crumble.mul(uShrink));
  return uSize.mul(seed.w).mul(shrink);
})();
```

**Step 3: Verify**

Reload, ramp phase. Expected: particles in the middle of the crumble window now shrink as well as darken/fade. Patches read more clearly as "dying" — small, dark, and dispersed. Clean particles on both sides.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): shrink particles during crumble peak"
```

---

## Task 10: Add ambient wave motion

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add uniforms**

```js
const uWaveAmp = uniform(0.015);
const uWaveSpeed = uniform(0.25);
const uWaveFreq = uniform(2.0);
```

**Step 2: Add wave term inside positionNode**

At the end of positionNode, just before the final `vec3(...)` return, add a wave displacement applied to every particle (not gated by crumble):

```js
// ambient wave — always on, gentle breathing
const wavePhase = pA.x.mul(uWaveFreq).add(pA.y.mul(1.5)).add(t.mul(uWaveSpeed));
const wX = sin(wavePhase).mul(uWaveAmp);
const wY = cos(wavePhase.mul(0.7)).mul(uWaveAmp.mul(0.8));
const wZ = sin(wavePhase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.6));
```

Then update the final return to add `wX/wY/wZ`:

```js
return vec3(
  xy.x.add(dx).add(wX),
  xy.y.add(dy).add(wY),
  z.add(dz).add(wZ)
);
```

**Step 3: Verify**

Reload. Keep `uPhase = 0` (hold A). Expected: the cloud is no longer stone-still — there's a subtle, coherent breathing motion across the whole painting. Particles drift together in slow waves. Amplitude should be small enough that the painting still reads clearly.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): gentle ambient wave motion always on"
```

---

## Task 11: Add soft-splat gaussian alpha falloff

**Files:**
- Modify: `pointcloud-v6/main.js`

This makes each point a soft disc rather than a hard square — required for the points to blend into continuous tone.

**Step 1: Import the UV helper**

Add `uv` to the tsl imports (it may already be there; if not):

```js
import { attribute, uniform, float, vec2, vec3, vec4, Fn, mix, sin, cos, smoothstep, clamp, uv } from 'three/tsl';
```

**Step 2: Apply a gaussian falloff in colorNode**

Wrap the alpha in `colorNode` with a UV-based radial falloff. Replace the final `return` of `colorNode`:

```js
// soft-splat: radial gaussian from point-sprite center
const pUV = uv();                             // [0,1] over the point sprite
const r = pUV.sub(0.5).length();              // [0, ~0.7]
const softMask = smoothstep(0.5, 0.0, r);     // 1 at center, 0 at edges
const finalAlpha = alpha.mul(softMask);
return vec4(litCol.x, litCol.y, litCol.z, finalAlpha);
```

If `uv()` is not the correct TSL builtin for point-sprite UV, try `pointUV` or `uv()` from the default exports. If both fail, check `pointcloud-v4/main.js` for how v4 handles point falloff (it does not — v4 uses raw squares). Fallback: keep hard squares and reduce `uSize` to ~2.5 for a similar effect. Document which path you took.

**Step 3: Verify**

Reload. Expected: each particle is now a soft round splat rather than a hard square. The painting should read visibly softer and more continuous, especially in denser regions. Zoom in (scroll) to confirm individual points have round fades.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): soft gaussian splat alpha falloff"
```

---

## Task 12: Auto-loop state machine

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add params + state**

Above `init()`:

```js
const params = {
  holdDuration: 4.0,
  transitionDuration: 6.0,
  autoLoop: true,
};
```

**Step 2: Replace the animation loop**

```js
let stateStart = performance.now() / 1000;
let stateName = 'holdA';  // holdA | AtoB | holdB | BtoA

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
```

**Step 3: Verify**

Reload. Expected: the cloud automatically ping-pongs between paintings A and B. A holds ~4s, dissolves to B over ~6s (visible patchy crumbling), B holds ~4s, dissolves back to A over ~6s. Loop continues indefinitely. ~20s round trip.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): auto-loop state machine drives ping-pong"
```

---

## Task 13: Keyboard controls (space, ←/→, F)

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Add keyboard handler**

Before the render loop:

```js
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
```

**Step 2: Verify**

Reload. Expected:
- Spacebar pauses/resumes the auto-loop (watch the cloud freeze or start moving).
- ← snaps to painting A (instant — this is the pause-and-jump behavior, no animation).
- → snaps to painting B.
- F toggles fullscreen.

**Step 3: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): keyboard controls for loop/jump/fullscreen"
```

---

## Task 14: lil-gui tuning panel

**Files:**
- Modify: `pointcloud-v6/main.js`

**Step 1: Import lil-gui**

Top of file:

```js
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';
```

**Step 2: Build the panel after the material is set up**

```js
const gui = new GUI({ title: 'Pointcloud v6' });

const fLoop = gui.addFolder('Loop');
fLoop.add(params, 'autoLoop').listen().name('Auto Loop');
fLoop.add(params, 'holdDuration', 0, 20, 0.1).name('Hold (s)');
fLoop.add(params, 'transitionDuration', 1, 20, 0.1).name('Transition (s)');
fLoop.add({ phase: 0 }, 'phase', 0, 1, 0.001).name('Phase').onChange(v => {
  params.autoLoop = false;
  uPhase.value = v;
}).listen();

const fDis = gui.addFolder('Disintegration');
fDis.add({ patchScale: 3.0 }, 'patchScale', 0.5, 12, 0.1).name('Patch Scale').onChange(v => uPatchScale.value = v);
fDis.add({ fadeWindow: 0.08 }, 'fadeWindow', 0.02, 0.3, 0.005).name('Fade Window').onChange(v => uFadeWindow.value = v);
fDis.add({ dispersalAmp: 0.35 }, 'dispersalAmp', 0, 2, 0.01).name('Dispersal').onChange(v => uDispersalAmp.value = v);
fDis.add({ gravity: 0.25 }, 'gravity', 0, 1, 0.01).name('Gravity').onChange(v => uGravity.value = v);
fDis.add({ chaosAmp: 0.18 }, 'chaosAmp', 0, 0.6, 0.01).name('Chaos').onChange(v => uChaosAmp.value = v);
fDis.add({ shrink: 0.85 }, 'shrink', 0, 1, 0.01).name('Shrink').onChange(v => uShrink.value = v);
fDis.add({ darken: 0.7 }, 'darken', 0, 1, 0.01).name('Darken').onChange(v => uDarken.value = v);

const fR = gui.addFolder('Rendering');
fR.add({ size: 4.0 }, 'size', 1, 12, 0.1).name('Splat Size').onChange(v => uSize.value = v);
fR.add({ opacity: 0.9 }, 'opacity', 0, 1, 0.01).name('Opacity').onChange(v => uOpacity.value = v);
fR.add({ waveAmp: 0.015 }, 'waveAmp', 0, 0.05, 0.001).name('Wave Amp').onChange(v => uWaveAmp.value = v);

gui.add({
  jumpA: () => { params.autoLoop = false; stateName = 'holdA'; uPhase.value = 0; }
}, 'jumpA').name('⟵ Jump to A');
gui.add({
  jumpB: () => { params.autoLoop = false; stateName = 'holdB'; uPhase.value = 1; }
}, 'jumpB').name('Jump to B ⟶');
```

Note: the `{ patchScale: 3.0 }` literal pattern keeps each slider's local state tidy. If you prefer, pull all these into the `params` object up top and reference `params.X` — either is fine.

**Step 3: Verify**

Reload. Expected: lil-gui panel appears in the top-right. Each folder collapses/expands. Scrub `Phase` slider → auto-loop pauses, phase is driven manually, cloud responds live. Tune `Patch Scale`, `Dispersal`, etc. and watch the visual update in real time. `Jump to A` / `Jump to B` buttons work.

**Step 4: Commit**

```bash
git add pointcloud-v6/main.js
git commit -m "feat(pointcloud-v6): lil-gui panel for live tuning"
```

---

## Task 15: Link v6 from the index page

**Files:**
- Modify: `index.html`

**Step 1: Add v6 link to the Point cloud list**

In `index.html`, inside the `Point cloud` `<ul>`, insert above the v4 line:

```html
<li><a href="./pointcloud-v6/"><span>v6 — 2-painting disintegration transition</span><span class="slug">pointcloud-v6</span></a></li>
```

**Step 2: Verify**

Open `http://localhost:<port>/` — the v6 link appears at the top of the Point cloud list. Click it — v6 loads.

**Step 3: Commit**

```bash
git add index.html
git commit -m "feat(index): link pointcloud-v6"
```

---

## Final verification pass

After all tasks are complete, manually run through this checklist in the browser:

1. Open `pointcloud-v6/` — loads cleanly, no console errors.
2. Painting A (0012 desert) holds for ~4s with gentle ambient motion.
3. A visibly *disintegrates* into B (0029 ruins) over ~6s with clustered patchy crumbling, dimming, and shrinking.
4. Painting B holds for ~4s.
5. B disintegrates back into A over ~6s.
6. Spacebar pauses the loop; pressing again resumes.
7. ← and → jump instantly to A/B.
8. F toggles fullscreen.
9. GUI Phase slider lets you scrub the transition manually.
10. GUI tuning responds live (patch scale, dispersal, etc.).
11. Framerate holds 60fps on a modern Mac at 1440p.

If any step fails, file the failure as a follow-up fix commit — do not claim the plan complete.
