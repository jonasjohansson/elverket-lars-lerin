import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import {
  uniform, attribute, float, vec2, vec3, vec4,
  Fn, sin, cos, mix, clamp, smoothstep, cross,
  pass, viewportUV,
} from 'three/tsl';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
import { afterImage } from 'three/addons/tsl/display/AfterImageNode.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ─── Config ───
const PARTICLE_COUNT = 2_500_000;
const DUST_COUNT = 50_000;
const COLOR_IMG_A = '../shared/images/lerin2.jpg';
const DEPTH_IMG_A = '../shared/images/lerin2_depth.png';
const COLOR_IMG_B = '../shared/images/lerin3.jpg';
const DEPTH_IMG_B = '../shared/images/lerin3_depth.png';
const info = document.getElementById('info');

// ─── Params ───
const params = {
  splatScale: 2.5,
  opacity: 1.0,
  bgColor: '#000000',
  depthNear: 1.5,
  depthFar: -4.0,
  depthCurve: 0.8,
  depthScatter: 0.15,
  // Waves (continuous coordinated motion) — much gentler for image clarity
  waveAmplitude: 0.015,
  waveSpeed: 0.25,
  waveFrequency: 2.0,
  wave2Amplitude: 0.008,
  wave2Speed: 0.18,
  wave2Frequency: 3.5,
  microDrift: 0.0015,
  // Morphing state (0=calm, 0.5=wave, 1=turbulent)
  morphState: 0.0,
  morphSpeed: 0.08,
  autoMorph: false,
  // Dissolution
  dissolveProgress: 0.0,
  windStrength: 3.0,
  gravity: 0.2,
  turbulence: 0.5,
  curlStrength: 0.8,
  // Transition A ⇄ B
  transitionProgress: 0.0,
  transitionDuration: 14.0,
  transitionScatter: 0.12,
  // Cursor interaction (wither-on-hover)
  cursorRadius: 0.7,
  cursorPush: 0.18,
  cursorChaos: 0.05,
  cursorFade: 0.75,
  cursorDarken: 0.35,
  cursorFollow: 0.035,    // how slowly the field chases the cursor (lower = more lag)
  cursorRise: 0.04,       // strength ramp when hovering
  cursorDecay: 0.02,      // strength fade when leaving
  cursorPressScale: 2.5,  // radius multiplier while LEFT mouse is held
  cursorPressLerp: 0.12,  // how fast press expansion eases
  // Right-click vortex: particles get pulled in + swirled around the cursor ray
  cursorPull: 0.45,       // inward pull magnitude
  cursorSwirl: 0.55,      // tangential swirl around ray axis
  cursorVortexLerp: 0.06, // how fast vortex mix transitions
  // Disintegration: each particle picks its own moment across `duration` and falls once.
  disintegrateEnabled: true,
  disintegrateDuration: 120.0,   // seconds to fully disintegrate the image
  disintegrateFallDuration: 3.5, // seconds each individual particle spends falling
  disintegrateFall: 1.8,         // world units each particle falls
  disintegrateSway: 0.15,        // lateral drift during fall
  disintegrateStrength: 1.0,     // master multiplier (0 = disabled / preserved)
  // Lights
  ambientLight: 0.85,
  light1Color: '#ffffff',
  light1Intensity: 0.35,
  light2Color: '#ffffff',
  light2Intensity: 0.2,
  lightFalloff: 1.5,
  // Color shift
  hueShiftSpeed: 0.02,
  hueShiftAmount: 0.0,
  // Post FX
  vignetteStrength: 0.2,
  trailStrength: 0.0,
  dofFocus: 4.8,
  dofAperture: 0.001,
  dofMaxBlur: 0.005,
  // Dust
  dustOpacity: 0.05,
  dustSpeed: 0.05,
  // Auto
  autoPlay: false,
  cycleDuration: 12.0,
  holdDuration: 5.0,
  autoRotate: false,
  autoRotateSpeed: 0.15,
};

// ─── Renderer ───
const canvas = document.getElementById('c');
const renderer = new THREE.WebGPURenderer({ canvas, antialias: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setClearColor(params.bgColor);
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.2;

// ─── Scene & Camera ───
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.01, 100);
camera.position.set(0, 0, 3.8);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.04;
controls.rotateSpeed = 0.4;
controls.target.set(0, 0, -0.3);
// Free right-click for the vortex gesture (default RIGHT is PAN)
controls.mouseButtons = { LEFT: THREE.MOUSE.ROTATE, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: null };

// ─── Image helpers ───
function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`Failed: ${src}`));
    img.src = src;
  });
}

function getPixels(img, maxW = 2200) {
  let w = img.width, h = img.height;
  if (w > maxW) { const s = maxW / w; w = Math.round(w * s); h = Math.round(h * s); }
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(img, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

// ─── Build particles ───
async function buildParticles() {
  info.textContent = 'Loading images...';
  const [colorImgA, depthImgA, colorImgB, depthImgB] = await Promise.all([
    loadImage(COLOR_IMG_A), loadImage(DEPTH_IMG_A),
    loadImage(COLOR_IMG_B), loadImage(DEPTH_IMG_B),
  ]);
  const srcA = { color: getPixels(colorImgA), depth: getPixels(depthImgA) };
  const srcB = { color: getPixels(colorImgB), depth: getPixels(depthImgB) };

  info.textContent = `Building ${(PARTICLE_COUNT / 1e6).toFixed(1)}M particles...`;
  await new Promise(r => requestAnimationFrame(r));

  const sceneH = 2.0;
  const buildSrc = (src) => {
    const { width, height } = src.color;
    const aspect = width / height;
    const sceneW = sceneH * aspect;
    return {
      cPx: src.color.data, dPx: src.depth.data,
      w: width, h: height,
      sceneW,
      jX: sceneW / width, jY: sceneH / height,
    };
  };
  const A = buildSrc(srcA);
  const B = buildSrc(srcB);

  const positions = new Float32Array(PARTICLE_COUNT * 3);
  const colors = new Float32Array(PARTICLE_COUNT * 3);
  const randoms = new Float32Array(PARTICLE_COUNT * 4);
  const groups = new Float32Array(PARTICLE_COUNT);

  const halfCount = PARTICLE_COUNT >> 1;
  // Tighter jitter for crisper image (was 2.0)
  const scatter = 1.0;
  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const inB = i >= halfCount;
    const s = inB ? B : A;
    groups[i] = inB ? 1.0 : 0.0;

    const px = Math.random() * s.w | 0;
    const py = Math.random() * s.h | 0;
    const rgba = (py * s.w + px) * 4;
    const nx = px / s.w, ny = py / s.h;

    positions[i * 3] = (nx - 0.5) * s.sceneW + (Math.random() - 0.5) * s.jX * scatter;
    positions[i * 3 + 1] = (0.5 - ny) * sceneH + (Math.random() - 0.5) * s.jY * scatter;
    const rawD = s.dPx[rgba] / 255;
    const d = Math.pow(rawD, params.depthCurve);
    const depthBase = d * (params.depthNear - params.depthFar) + params.depthFar;
    const zScatter = (Math.random() - 0.5) * params.depthScatter * (1 + d);
    positions[i * 3 + 2] = depthBase + zScatter;

    let cr = s.cPx[rgba] / 255, cg = s.cPx[rgba + 1] / 255, cb = s.cPx[rgba + 2] / 255;
    // Lighter color perturbation (was 0.06 / -0.4 bias) — preserve painting colors
    const w = (Math.random() - 0.5) * 0.03;
    cr = Math.max(0, Math.min(1, cr + w));
    cg = Math.max(0, Math.min(1, cg + w * 0.4));
    cb = Math.max(0, Math.min(1, cb - w * 0.3));
    const lum = cr * 0.299 + cg * 0.587 + cb * 0.114;
    const sat = 1.0 + (Math.random() - 0.5) * 0.15;
    colors[i * 3] = Math.max(0, Math.min(1, lum + (cr - lum) * sat));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, lum + (cg - lum) * sat));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, lum + (cb - lum) * sat));

    randoms[i * 4] = (Math.random() - 0.5) * 2;
    randoms[i * 4 + 1] = (Math.random() - 0.5) * 2;
    randoms[i * 4 + 2] = Math.random();
    randoms[i * 4 + 3] = 0.3 + Math.random() * 0.7;
  }

  return { positions, colors, randoms, groups };
}

// ─── Build dust ───
function buildDust() {
  const positions = new Float32Array(DUST_COUNT * 3);
  const randoms = new Float32Array(DUST_COUNT * 3);
  for (let i = 0; i < DUST_COUNT; i++) {
    positions[i * 3] = (Math.random() - 0.5) * 6;
    positions[i * 3 + 1] = (Math.random() - 0.5) * 4;
    positions[i * 3 + 2] = (Math.random() - 0.5) * 4 - 0.5;
    randoms[i * 3] = (Math.random() - 0.5) * 2;
    randoms[i * 3 + 1] = (Math.random() - 0.5) * 2;
    randoms[i * 3 + 2] = Math.random();
  }
  return { positions, randoms };
}

// ─── Init ───
async function init() {
  await renderer.init();
  const data = await buildParticles();

  // ─── Uniforms ───
  const uScale = uniform(params.splatScale);
  const uOpacity = uniform(params.opacity);
  const uDissolve = uniform(0.0);
  const uTransition = uniform(params.transitionProgress);
  const uTransScatter = uniform(params.transitionScatter);
  // Cursor field uniforms — ray in world space
  const uCursorRayOrigin = uniform(new THREE.Vector3(0, 0, 10));
  const uCursorRayDir = uniform(new THREE.Vector3(0, 0, -1));
  const uCursorStrength = uniform(0.0);
  const uCursorRadius = uniform(params.cursorRadius);
  const uCursorPush = uniform(params.cursorPush);
  const uCursorChaos = uniform(params.cursorChaos);
  const uCursorFade = uniform(params.cursorFade);
  const uCursorDarken = uniform(params.cursorDarken);
  // Vortex: 0 = normal push-out, 1 = pull-in + swirl
  const uVortexMix = uniform(0.0);
  const uCursorPull = uniform(params.cursorPull);
  const uCursorSwirl = uniform(params.cursorSwirl);
  // Disintegration (one-shot slow decay, each particle dies once over uDisintDuration)
  const uDisintStart = uniform(0.0);  // global timestamp when disintegration was (re)started
  const uDisintDuration = uniform(params.disintegrateDuration);
  const uDisintFallDur = uniform(params.disintegrateFallDuration);
  const uDisintFall = uniform(params.disintegrateFall);
  const uDisintSway = uniform(params.disintegrateSway);
  const uDisintStrength = uniform(params.disintegrateEnabled ? params.disintegrateStrength : 0.0);
  const uWind = uniform(params.windStrength);
  const uGravity = uniform(params.gravity);
  const uTurb = uniform(params.turbulence);
  const uCurl = uniform(params.curlStrength);
  const uTime = uniform(0.0);
  // Waves
  const uWaveAmp = uniform(params.waveAmplitude);
  const uWaveSpd = uniform(params.waveSpeed);
  const uWaveFreq = uniform(params.waveFrequency);
  const uWave2Amp = uniform(params.wave2Amplitude);
  const uWave2Spd = uniform(params.wave2Speed);
  const uWave2Freq = uniform(params.wave2Frequency);
  const uMicroDrift = uniform(params.microDrift);
  const uMorph = uniform(params.morphState);
  // Lights
  const uLight1Pos = uniform(new THREE.Vector3(2, 1, 1));
  const uLight1Col = uniform(new THREE.Color(params.light1Color));
  const uLight1Int = uniform(params.light1Intensity);
  const uLight2Pos = uniform(new THREE.Vector3(-2, -0.5, 0.5));
  const uLight2Col = uniform(new THREE.Color(params.light2Color));
  const uLight2Int = uniform(params.light2Intensity);
  const uAmbient = uniform(params.ambientLight);
  const uLightFalloff = uniform(params.lightFalloff);
  // Color shift
  const uHueShift = uniform(0.0);

  // ─── Geometry ───
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('aColor', new THREE.BufferAttribute(data.colors, 3));
  geometry.setAttribute('aRandom', new THREE.BufferAttribute(data.randoms, 4));
  geometry.setAttribute('aGroup', new THREE.BufferAttribute(data.groups, 1));
  geometry.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);

  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: true,
    depthTest: true,
  });

  // ─── Position: waves + morph + dissolution ───
  material.positionNode = Fn(() => {
    const pos = attribute('position');
    const rnd = attribute('aRandom');
    const seedX = rnd.x, seedY = rnd.y, delay = rnd.z;
    const t = uTime;

    // ── Coordinated wave motion (always active, Anadol-style ocean) ──
    // Wave 1: large, slow, sweeps across X
    const wave1Phase = pos.x.mul(uWaveFreq).add(pos.y.mul(1.5)).add(t.mul(uWaveSpd));
    const w1x = sin(wave1Phase).mul(uWaveAmp);
    const w1y = cos(wave1Phase.mul(0.7)).mul(uWaveAmp.mul(0.8));
    const w1z = sin(wave1Phase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.6));

    // Wave 2: faster, perpendicular, creates cross-swell
    const wave2Phase = pos.y.mul(uWave2Freq).add(pos.z.mul(2.0)).add(t.mul(uWave2Spd));
    const w2x = cos(wave2Phase).mul(uWave2Amp.mul(0.5));
    const w2y = sin(wave2Phase).mul(uWave2Amp);
    const w2z = cos(wave2Phase.mul(1.3)).mul(uWave2Amp.mul(0.7));

    // Wave 3: diagonal, creates complex interference
    const wave3Phase = pos.x.add(pos.y).mul(1.8).add(t.mul(0.15));
    const w3x = sin(wave3Phase).mul(uWaveAmp.mul(0.3));
    const w3y = cos(wave3Phase.mul(1.2)).mul(uWaveAmp.mul(0.4));

    // Micro-drift per particle
    const dx = sin(t.mul(0.13).add(seedX.mul(7.0))).mul(uMicroDrift);
    const dy = cos(t.mul(0.11).add(seedY.mul(5.0))).mul(uMicroDrift);

    // ── Morph state modulates motion intensity ──
    // 0 = calm (waves only), 0.5 = flowing (waves + turbulence), 1 = turbulent (heavy noise)
    const morphTurb = uMorph.mul(uMorph); // quadratic: turbulence kicks in at higher morph
    const turbAmp = morphTurb.mul(0.06);
    const morphTurbX = sin(t.mul(1.7).add(seedX.mul(4.0)).add(pos.y.mul(3.0))).mul(turbAmp);
    const morphTurbY = cos(t.mul(2.3).add(seedY.mul(3.0)).add(pos.x.mul(2.5))).mul(turbAmp);
    const morphTurbZ = sin(t.mul(1.1).add(seedX.add(seedY).mul(5.0))).mul(turbAmp.mul(0.5));

    // ── Dissolution ──
    const nx = pos.x.add(1.5).div(3.0).clamp(0, 1);
    const activation = nx.mul(0.35).add(delay.mul(0.35));
    const rawD = uDissolve.mul(1.4).sub(activation).div(float(1.0).sub(activation).add(0.001)).clamp(0, 1);
    const d = rawD.mul(rawD).mul(rawD);

    const windVar = seedX.mul(0.4).add(1.0);
    const windX = d.mul(uWind).mul(windVar).negate();

    const ts1 = sin(t.mul(0.5).add(seedX.mul(2.5))).mul(0.6);
    const tm1 = sin(t.mul(1.3).add(seedY.mul(5.0))).mul(0.3);
    const tf = sin(t.mul(3.5).add(seedX.add(seedY).mul(8.0))).mul(0.1);
    const dissolveTurbX = ts1.add(tm1).add(tf).mul(uTurb).mul(d);
    const dissolveTurbY = cos(t.mul(0.4).add(seedY.mul(3.1))).mul(0.5)
      .add(cos(t.mul(1.7).add(seedX.mul(4.2))).mul(0.25)).mul(uTurb).mul(d);

    const angle = d.mul(3.14159).mul(windVar);
    const curlX = sin(angle.add(seedX.mul(6.28))).mul(uCurl).mul(d);
    const curlY = cos(angle.add(seedY.mul(6.28))).mul(uCurl).mul(d);

    const grav = d.mul(d).mul(uGravity).negate();
    const lift = d.mul(0.08).mul(seedY.add(0.5).clamp(0, 1));

    // ── Transition scatter: gentle per-particle drift that peaks while a particle is fading ──
    const groupT = attribute('aGroup');
    const transFade = mix(float(1.0).sub(uTransition), uTransition, groupT);
    const transOut = float(1.0).sub(transFade);               // 0 when visible, 1 when faded
    const tw = t.mul(0.25);
    const windDir = seedX.mul(0.6).add(0.4);                  // mostly leftward-ish wind
    const transX = transOut.mul(uTransScatter).mul(windDir.negate())
      .add(sin(tw.mul(1.1).add(seedY.mul(4.0))).mul(uTransScatter.mul(0.3)).mul(transOut));
    const transY = sin(tw.add(seedX.mul(3.0))).mul(uTransScatter.mul(0.35)).mul(transOut)
      .add(seedY.mul(uTransScatter.mul(0.25)).mul(transOut));
    const transZ = cos(tw.mul(0.7).add(seedX.add(seedY).mul(2.0))).mul(uTransScatter.mul(0.2)).mul(transOut);

    // ── Cursor field: perpendicular distance to mouse ray (depth-agnostic). Blends wither → vortex. ──
    const toP = pos.sub(uCursorRayOrigin);
    const parallelLen = toP.dot(uCursorRayDir);
    const perp = toP.sub(uCursorRayDir.mul(parallelLen));
    const curDist = perp.length().add(0.001);
    const curFall = float(1.0).sub(smoothstep(uCursorRadius.mul(0.15), uCursorRadius, curDist)).mul(uCursorStrength);
    const curFall2 = curFall.mul(curFall);
    const pushDir = perp.div(curDist);                    // outward from ray axis
    const swirlDir = cross(uCursorRayDir, pushDir);       // tangent around ray axis
    const outwardForce = pushDir.mul(uCursorPush);
    const vortexForce = pushDir.mul(uCursorPull.negate()).add(swirlDir.mul(uCursorSwirl));
    const cursorPushVec = mix(outwardForce, vortexForce, uVortexMix).mul(curFall2);
    const ct = t.mul(0.6);
    const cx = sin(ct.add(seedX.mul(5.0))).mul(uCursorChaos).mul(curFall);
    const cy = cos(ct.mul(1.2).add(seedY.mul(4.0))).mul(uCursorChaos).mul(curFall);
    const cz = sin(ct.mul(0.8).add(seedX.add(seedY).mul(3.0))).mul(uCursorChaos.mul(0.6)).mul(curFall);

    // ── Slow disintegration: each particle picks its own death moment once over uDisintDuration ──
    const deathMoment = uDisintStart.add(delay.mul(uDisintDuration));
    const dieRaw = t.sub(deathMoment).div(uDisintFallDur).clamp(0, 1);  // 0 alive → 1 fallen & gone
    const dieProg = dieRaw.mul(uDisintStrength);
    const ambFallY = dieProg.mul(dieProg).mul(uDisintFall).negate();    // accelerating downward
    const ambSwayX = sin(t.mul(0.9).add(seedX.mul(6.0))).mul(uDisintSway).mul(dieProg).mul(seedY.mul(0.5).add(0.5));
    const ambSwayZ = cos(t.mul(0.7).add(seedY.mul(5.0))).mul(uDisintSway.mul(0.3)).mul(dieProg);

    return pos.add(vec3(
      w1x.add(w2x).add(w3x).add(dx).add(morphTurbX).add(windX).add(dissolveTurbX).add(curlX).add(transX).add(cursorPushVec.x).add(cx).add(ambSwayX),
      w1y.add(w2y).add(w3y).add(dy).add(morphTurbY).add(dissolveTurbY).add(grav).add(lift).add(curlY).add(transY).add(cursorPushVec.y).add(cy).add(ambFallY),
      w1z.add(w2z).add(morphTurbZ).add(sin(t.mul(0.7).add(seedX.mul(seedY).mul(3.0))).mul(0.15).mul(uTurb).mul(d)).add(transZ).add(cursorPushVec.z).add(cz).add(ambSwayZ)
    ));
  })();

  // ─── Size: pulse with morph ───
  material.sizeNode = Fn(() => {
    const rnd = attribute('aRandom');
    const pulse = sin(uTime.mul(1.5).add(rnd.x.mul(6.28))).mul(0.1).add(1.0);
    // Morph makes particles slightly bigger (turbulence = expansion)
    const morphSize = float(1.0).add(uMorph.mul(0.2));
    return rnd.w.mul(uScale).mul(pulse).mul(morphSize);
  })();
  material.sizeAttenuation = true;

  // ─── Color: lighting + hue shift + dissolve ───
  material.colorNode = Fn(() => {
    const col = attribute('aColor');
    const rnd = attribute('aRandom');
    const pos = attribute('position');
    const delay = rnd.z;

    // Hue rotation (RGB shift approximation)
    const hueAngle = uHueShift;
    const cosH = cos(hueAngle);
    const sinH = sin(hueAngle);
    // Simplified hue rotation matrix applied to color
    const rr = col.x.mul(cosH.add(float(1.0).sub(cosH).mul(0.333))).add(col.y.mul(float(1.0).sub(cosH).mul(0.333).sub(sinH.mul(0.577)))).add(col.z.mul(float(1.0).sub(cosH).mul(0.333).add(sinH.mul(0.577))));
    const gg = col.x.mul(float(1.0).sub(cosH).mul(0.333).add(sinH.mul(0.577))).add(col.y.mul(cosH.add(float(1.0).sub(cosH).mul(0.333)))).add(col.z.mul(float(1.0).sub(cosH).mul(0.333).sub(sinH.mul(0.577))));
    const bb = col.x.mul(float(1.0).sub(cosH).mul(0.333).sub(sinH.mul(0.577))).add(col.y.mul(float(1.0).sub(cosH).mul(0.333).add(sinH.mul(0.577)))).add(col.z.mul(cosH.add(float(1.0).sub(cosH).mul(0.333))));
    const hueCol = vec3(rr, gg, bb);

    // Lighting
    const toLight1 = uLight1Pos.sub(pos);
    const dist1 = toLight1.length();
    const atten1 = uLight1Int.div(dist1.pow(uLightFalloff).add(0.3));
    const light1 = uLight1Col.mul(atten1);
    const toLight2 = uLight2Pos.sub(pos);
    const dist2 = toLight2.length();
    const atten2 = uLight2Int.div(dist2.pow(uLightFalloff).add(0.3));
    const light2 = uLight2Col.mul(atten2);
    const illumination = vec3(uAmbient, uAmbient, uAmbient).add(light1).add(light2);
    const litCol = hueCol.mul(illumination);

    // Dissolve fade + warm shift (unchanged, driven by uDissolve / spacebar)
    const nx = pos.x.add(1.5).div(3.0).clamp(0, 1);
    const activation = nx.mul(0.35).add(delay.mul(0.35));
    const rawD = uDissolve.mul(1.4).sub(activation).div(float(1.0).sub(activation).add(0.001)).clamp(0, 1);
    const dissolveFade = float(1.0).sub(smoothstep(float(0.3), float(1.0), rawD));
    const warmShift = rawD.mul(0.12);
    const finalCol = vec3(litCol.x.add(warmShift), litCol.y, litCol.z.sub(warmShift.mul(0.5)));

    // Transition A ⇄ B: pure per-group linear alpha crossfade (no motion).
    const group = attribute('aGroup');
    const transitionFade = mix(float(1.0).sub(uTransition), uTransition, group);

    // Cursor wither: fade and darken near cursor (perpendicular-to-ray distance).
    const toP = pos.sub(uCursorRayOrigin);
    const parallelLen = toP.dot(uCursorRayDir);
    const perp = toP.sub(uCursorRayDir.mul(parallelLen));
    const curDist = perp.length().add(0.001);
    const curFall = float(1.0).sub(smoothstep(uCursorRadius.mul(0.15), uCursorRadius, curDist)).mul(uCursorStrength);
    const cursorAlpha = float(1.0).sub(curFall.mul(uCursorFade));
    const cursorDark = float(1.0).sub(curFall.mul(uCursorDarken));
    const witheredCol = finalCol.mul(cursorDark);

    // Disintegration alpha: once a particle's death moment passes, it fades out permanently.
    const deathMoment = uDisintStart.add(delay.mul(uDisintDuration));
    const dieRaw = uTime.sub(deathMoment).div(uDisintFallDur).clamp(0, 1);
    const dieProg = dieRaw.mul(uDisintStrength);
    const ambAlpha = float(1.0).sub(smoothstep(float(0.05), float(0.9), dieProg));

    return vec4(witheredCol, uOpacity.mul(dissolveFade).mul(transitionFade).mul(cursorAlpha).mul(ambAlpha));
  })();

  const points = new THREE.Points(geometry, material);
  scene.add(points);

  // ─── Dust ───
  const dustData = buildDust();
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustData.positions, 3));
  dustGeo.setAttribute('aRandom', new THREE.BufferAttribute(dustData.randoms, 3));
  const uDustOpacity = uniform(params.dustOpacity);
  const uDustSpeed = uniform(params.dustSpeed);
  const dustMat = new THREE.PointsNodeMaterial({ transparent: true, blending: THREE.AdditiveBlending, depthWrite: false });
  dustMat.positionNode = Fn(() => {
    const pos = attribute('position');
    const rnd = attribute('aRandom');
    const t = uTime;
    const sx = sin(t.mul(uDustSpeed).add(rnd.x.mul(5.0))).mul(0.3);
    const sy = cos(t.mul(uDustSpeed.mul(0.7)).add(rnd.y.mul(4.0))).mul(0.2);
    const sz = sin(t.mul(uDustSpeed.mul(0.5)).add(rnd.z.mul(3.0))).mul(0.15);
    const drift = t.mul(0.01).add(rnd.z);
    const yMod = pos.y.add(drift).sub(float(2.0)).mod(float(4.0)).sub(float(2.0));
    return vec3(pos.x.add(sx), yMod.add(sy), pos.z.add(sz));
  })();
  dustMat.sizeNode = float(1.5);
  dustMat.sizeAttenuation = true;
  dustMat.colorNode = Fn(() => {
    const rnd = attribute('aRandom');
    const flicker = sin(uTime.mul(2.0).add(rnd.x.mul(10.0))).mul(0.3).add(0.7);
    return vec4(float(0.6), float(0.6), float(0.7), uDustOpacity.mul(flicker));
  })();
  scene.add(new THREE.Points(dustGeo, dustMat));

  // ─── Light helpers ───
  const lightSphereGeo = new THREE.SphereGeometry(0.05, 8, 8);
  const light1Helper = new THREE.Mesh(lightSphereGeo, new THREE.MeshBasicMaterial({ color: params.light1Color }));
  const light2Helper = new THREE.Mesh(lightSphereGeo, new THREE.MeshBasicMaterial({ color: params.light2Color }));
  light1Helper.position.copy(uLight1Pos.value);
  light2Helper.position.copy(uLight2Pos.value);
  scene.add(light1Helper);
  scene.add(light2Helper);

  let manualLightMode = true;
  let activeTransform = null;
  const createTransform = (helper, uniformPos) => {
    const tc = new TransformControls(camera, canvas);
    tc.attach(helper); tc.setSize(0.5);
    tc.addEventListener('dragging-changed', (e) => { controls.enabled = !e.value; if (!e.value) uniformPos.value.copy(helper.position); });
    tc.addEventListener('change', () => uniformPos.value.copy(helper.position));
    return tc;
  };
  let transform1 = null, transform2 = null;
  const showTransform = (which) => {
    if (activeTransform) { scene.remove(activeTransform.getHelper()); activeTransform = null; }
    if (which === 1) { if (!transform1) transform1 = createTransform(light1Helper, uLight1Pos); scene.add(transform1.getHelper()); activeTransform = transform1; }
    if (which === 2) { if (!transform2) transform2 = createTransform(light2Helper, uLight2Pos); scene.add(transform2.getHelper()); activeTransform = transform2; }
  };
  const hideTransforms = () => { if (activeTransform) { scene.remove(activeTransform.getHelper()); activeTransform = null; } };

  // ─── Post-processing: trails + DOF + vignette ───
  const uVignetteStrength = uniform(params.vignetteStrength);
  const uTrailStrength = uniform(params.trailStrength);
  const uDofFocus = uniform(params.dofFocus);
  const uDofAperture = uniform(params.dofAperture);
  const uDofMaxBlur = uniform(params.dofMaxBlur);

  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  const sceneViewZ = scenePass.getViewZNode();

  // Feedback trails (afterimage)
  const trailPass = afterImage(sceneColor, uTrailStrength);

  // DOF
  const dofPass = dof(trailPass, sceneViewZ, uDofFocus, uDofAperture, uDofMaxBlur);

  // Vignette
  const uv = viewportUV;
  const vigDist = uv.sub(0.5).length();
  const vignette = clamp(vigDist.mul(2.0), 0.0, 1.0);
  const vignetteFactor = float(1.0).sub(vignette.mul(vignette).mul(uVignetteStrength));

  postProcessing.outputNode = dofPass.mul(vignetteFactor);

  info.textContent = `${(PARTICLE_COUNT / 1e6).toFixed(1)}M · Slow disintegration · Hover: wither · Click: expand · Right-click: vortex · Space: dissolve · T: transition`;

  // ─── GUI ───
  const gui = new GUI({ title: 'Point Cloud' });

  const fR = gui.addFolder('Rendering');
  fR.add(params, 'splatScale', 0.5, 15, 0.25).name('Scale').onChange(v => uScale.value = v);
  fR.add(params, 'opacity', 0, 1, 0.01).name('Opacity').onChange(v => uOpacity.value = v);
  fR.addColor(params, 'bgColor').name('Background').onChange(v => renderer.setClearColor(v));

  const fW = gui.addFolder('Waves');
  fW.add(params, 'waveAmplitude', 0, 0.2, 0.005).name('Wave 1 Amp').onChange(v => uWaveAmp.value = v);
  fW.add(params, 'waveSpeed', 0, 1, 0.01).name('Wave 1 Speed').onChange(v => uWaveSpd.value = v);
  fW.add(params, 'waveFrequency', 0, 8, 0.1).name('Wave 1 Freq').onChange(v => uWaveFreq.value = v);
  fW.add(params, 'wave2Amplitude', 0, 0.15, 0.005).name('Wave 2 Amp').onChange(v => uWave2Amp.value = v);
  fW.add(params, 'wave2Speed', 0, 1, 0.01).name('Wave 2 Speed').onChange(v => uWave2Spd.value = v);
  fW.add(params, 'wave2Frequency', 0, 8, 0.1).name('Wave 2 Freq').onChange(v => uWave2Freq.value = v);
  fW.add(params, 'microDrift', 0, 0.02, 0.001).name('Micro Drift').onChange(v => uMicroDrift.value = v);

  const fM = gui.addFolder('Morph');
  fM.add(params, 'morphState', 0, 1, 0.01).name('State').listen().onChange(v => uMorph.value = v);
  fM.add(params, 'morphSpeed', 0, 0.3, 0.005).name('Auto Speed');
  fM.add(params, 'autoMorph').name('Auto Morph');

  const fDepth = gui.addFolder('Depth');
  fDepth.add(params, 'depthNear', -2, 5, 0.1).name('Near');
  fDepth.add(params, 'depthFar', -10, 2, 0.1).name('Far');
  fDepth.add(params, 'depthCurve', 0.2, 3, 0.1).name('Curve');
  fDepth.add(params, 'depthScatter', 0, 2, 0.05).name('Scatter');

  const fT = gui.addFolder('Transition');
  fT.add(params, 'transitionProgress', 0, 1, 0.001).name('A ⇄ B').listen().onChange(v => uTransition.value = v);
  fT.add(params, 'transitionDuration', 1, 30, 0.5).name('Duration (s)');
  fT.add(params, 'transitionScatter', 0, 0.6, 0.01).name('Wind Scatter').onChange(v => uTransScatter.value = v);

  const fC = gui.addFolder('Cursor');
  fC.add(params, 'cursorRadius', 0.1, 2.5, 0.05).name('Radius'); // driven via animate lerp
  fC.add(params, 'cursorPush', 0, 0.8, 0.01).name('Push (hover)').onChange(v => uCursorPush.value = v);
  fC.add(params, 'cursorChaos', 0, 0.3, 0.005).name('Chaos').onChange(v => uCursorChaos.value = v);
  fC.add(params, 'cursorFade', 0, 1, 0.01).name('Fade').onChange(v => uCursorFade.value = v);
  fC.add(params, 'cursorDarken', 0, 1, 0.01).name('Darken').onChange(v => uCursorDarken.value = v);
  fC.add(params, 'cursorFollow', 0.005, 0.25, 0.005).name('Follow Speed');
  fC.add(params, 'cursorRise', 0.005, 0.3, 0.005).name('Rise');
  fC.add(params, 'cursorDecay', 0.005, 0.3, 0.005).name('Decay');
  fC.add(params, 'cursorPressScale', 1, 5, 0.1).name('Click Scale');
  fC.add(params, 'cursorPressLerp', 0.02, 0.4, 0.01).name('Click Ease');
  fC.add(params, 'cursorPull', 0, 1.5, 0.01).name('Vortex Pull').onChange(v => uCursorPull.value = v);
  fC.add(params, 'cursorSwirl', 0, 2, 0.01).name('Vortex Swirl').onChange(v => uCursorSwirl.value = v);
  fC.add(params, 'cursorVortexLerp', 0.01, 0.3, 0.005).name('Vortex Ease');

  const fDis = gui.addFolder('Disintegration');
  fDis.add(params, 'disintegrateEnabled').name('Enabled').onChange(v => uDisintStrength.value = v ? params.disintegrateStrength : 0.0);
  fDis.add(params, 'disintegrateStrength', 0, 1, 0.01).name('Strength').onChange(v => uDisintStrength.value = params.disintegrateEnabled ? v : 0.0);
  fDis.add(params, 'disintegrateDuration', 10, 600, 1).name('Duration (s)').onChange(v => uDisintDuration.value = v);
  fDis.add(params, 'disintegrateFallDuration', 0.5, 10, 0.1).name('Fall Time (s)').onChange(v => uDisintFallDur.value = v);
  fDis.add(params, 'disintegrateFall', 0, 5, 0.05).name('Fall Depth').onChange(v => uDisintFall.value = v);
  fDis.add(params, 'disintegrateSway', 0, 0.5, 0.01).name('Sway').onChange(v => uDisintSway.value = v);


  const fD = gui.addFolder('Dissolution');
  fD.add(params, 'dissolveProgress', 0, 1, 0.001).name('Progress').listen().onChange(v => uDissolve.value = v);
  fD.add(params, 'windStrength', 0, 10, 0.1).name('Wind').onChange(v => uWind.value = v);
  fD.add(params, 'gravity', 0, 1, 0.01).name('Gravity').onChange(v => uGravity.value = v);
  fD.add(params, 'turbulence', 0, 2, 0.01).name('Turbulence').onChange(v => uTurb.value = v);
  fD.add(params, 'curlStrength', 0, 3, 0.05).name('Curl/Swirl').onChange(v => uCurl.value = v);

  const fLights = gui.addFolder('Lights');
  fLights.add(params, 'ambientLight', 0, 1, 0.01).name('Ambient').onChange(v => uAmbient.value = v);
  fLights.addColor(params, 'light1Color').name('Light 1').onChange(v => { uLight1Col.value.set(v); light1Helper.material.color.set(v); });
  fLights.add(params, 'light1Intensity', 0, 5, 0.1).name('Power 1').onChange(v => uLight1Int.value = v);
  fLights.addColor(params, 'light2Color').name('Light 2').onChange(v => { uLight2Col.value.set(v); light2Helper.material.color.set(v); });
  fLights.add(params, 'light2Intensity', 0, 5, 0.1).name('Power 2').onChange(v => uLight2Int.value = v);
  fLights.add(params, 'lightFalloff', 0.5, 3, 0.1).name('Falloff').onChange(v => uLightFalloff.value = v);

  const fFx = gui.addFolder('FX');
  fFx.add(params, 'hueShiftSpeed', 0, 0.1, 0.002).name('Hue Shift Speed');
  fFx.add(params, 'hueShiftAmount', 0, 0.5, 0.01).name('Hue Shift Range');
  fFx.add(params, 'trailStrength', 0, 0.97, 0.01).name('Trails').onChange(v => uTrailStrength.value = v);
  fFx.add(params, 'vignetteStrength', 0, 1.5, 0.05).name('Vignette').onChange(v => uVignetteStrength.value = v);
  fFx.add(params, 'dustOpacity', 0, 0.5, 0.01).name('Dust').onChange(v => uDustOpacity.value = v);

  const fDof = gui.addFolder('DOF');
  fDof.add(params, 'dofFocus', 0.1, 50, 0.1).name('Focus').onChange(v => uDofFocus.value = v);
  fDof.add(params, 'dofAperture', 0, 0.15, 0.001).name('Aperture').onChange(v => uDofAperture.value = v);
  fDof.add(params, 'dofMaxBlur', 0, 0.08, 0.001).name('Max Blur').onChange(v => uDofMaxBlur.value = v);

  const fA = gui.addFolder('Auto');
  fA.add(params, 'autoPlay').name('Auto Cycle');
  fA.add(params, 'cycleDuration', 4, 30, 0.5).name('Dissolve (s)');
  fA.add(params, 'holdDuration', 1, 15, 0.5).name('Hold (s)');
  fA.add(params, 'autoRotate').name('Rotate').onChange(v => controls.autoRotate = v);
  fA.add(params, 'autoRotateSpeed', 0, 1, 0.01).name('Rotate Speed').onChange(v => controls.autoRotateSpeed = v);

  const actions = {
    playDissolve: () => {
      params.dissolveProgress = 0;
      const start = performance.now();
      const dur = params.cycleDuration * 1000;
      function tick() {
        const p = Math.min(1, (performance.now() - start) / dur);
        params.dissolveProgress = p;
        uDissolve.value = p;
        if (p < 1) requestAnimationFrame(tick);
      }
      tick();
    },
    reset: () => { params.dissolveProgress = 0; uDissolve.value = 0; },
    playTransition: () => {
      const start = performance.now();
      const dur = params.transitionDuration * 1000;
      const from = params.transitionProgress;
      const to = from < 0.5 ? 1 : 0;
      function tick() {
        const p = Math.min(1, (performance.now() - start) / dur);
        // Linear easing — no curve
        const v = from + (to - from) * p;
        params.transitionProgress = v;
        uTransition.value = v;
        if (p < 1) requestAnimationFrame(tick);
      }
      tick();
    },
    resetCamera: () => { camera.position.set(0, 0, 3.8); controls.target.set(0, 0, -0.3); controls.update(); },
    restartDisintegration: () => { uDisintStart.value = uTime.value; },
  };
  gui.add(actions, 'playDissolve').name('▶ Play');
  gui.add(actions, 'playTransition').name('⇄ Transition');
  gui.add(actions, 'restartDisintegration').name('⟲ Restart Disintegration');
  gui.add(actions, 'reset').name('↺ Reset');
  gui.add(actions, 'resetCamera').name('⌂ Camera');

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); actions.playDissolve(); }
    if (e.code === 'KeyT') actions.playTransition();
    if (e.code === 'KeyR') actions.reset();
    if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen().catch(() => {});
    }
    if (e.key === '1') showTransform(1);
    if (e.key === '2') showTransform(2);
    if (e.key === 'Escape') hideTransforms();
  });

  // ─── Cursor field: wither on hover, expand on click, vortex on right-click ───
  const raycaster = new THREE.Raycaster();
  const pointer = new THREE.Vector2();
  const interactPlane = new THREE.Plane();
  const planeNormal = new THREE.Vector3();
  const cursorTarget = new THREE.Vector3(0, 0, -0.3);
  const cursorSmoothed = new THREE.Vector3(0, 0, -0.3);
  let cursorActive = false;
  let cursorInitialized = false;
  let cursorPressing = false;
  let cursorVortex = false;
  const hitTmp = new THREE.Vector3();

  canvas.addEventListener('pointermove', (e) => {
    const rect = canvas.getBoundingClientRect();
    pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    raycaster.setFromCamera(pointer, camera);
    camera.getWorldDirection(planeNormal);
    interactPlane.setFromNormalAndCoplanarPoint(planeNormal, controls.target);
    if (raycaster.ray.intersectPlane(interactPlane, hitTmp)) {
      cursorTarget.copy(hitTmp);
      if (!cursorInitialized) { cursorSmoothed.copy(hitTmp); cursorInitialized = true; }
      cursorActive = true;
    }
  });
  canvas.addEventListener('pointerleave', () => { cursorActive = false; });
  canvas.addEventListener('pointerenter', () => { cursorActive = true; });
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button === 2) cursorVortex = true;
    else if (e.button === 0) cursorPressing = true;
  });
  // Release on window so drag-outside-then-release still clears state
  window.addEventListener('pointerup', (e) => {
    if (e.button === 2) cursorVortex = false;
    else if (e.button === 0) cursorPressing = false;
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());

  // ─── Render loop ───
  const clock = new THREE.Clock();

  function animate() {
    const elapsed = clock.getElapsedTime();
    uTime.value = elapsed;
    controls.update();

    // Cursor field: smoothed world point → ray from camera, strength + press + vortex all lerped.
    cursorSmoothed.lerp(cursorTarget, params.cursorFollow);
    uCursorRayOrigin.value.copy(camera.position);
    uCursorRayDir.value.copy(cursorSmoothed).sub(camera.position).normalize();
    const targetStrength = cursorActive ? 1 : 0;
    const lerpRate = cursorActive ? params.cursorRise : params.cursorDecay;
    uCursorStrength.value += (targetStrength - uCursorStrength.value) * lerpRate;
    // Left-click expands radius, right-click swaps to vortex (pull + swirl).
    const targetRadius = params.cursorRadius * (cursorPressing || cursorVortex ? params.cursorPressScale : 1);
    uCursorRadius.value += (targetRadius - uCursorRadius.value) * params.cursorPressLerp;
    const targetVortex = cursorVortex ? 1 : 0;
    uVortexMix.value += (targetVortex - uVortexMix.value) * params.cursorVortexLerp;

    // Auto morph: smoothly oscillate between calm and turbulent
    if (params.autoMorph) {
      // Slow sine with multiple frequencies for organic feel
      const m = (Math.sin(elapsed * params.morphSpeed) * 0.5 + 0.5)
              * (0.7 + Math.sin(elapsed * params.morphSpeed * 0.37) * 0.3);
      params.morphState = m;
      uMorph.value = m;
    }

    // Continuous hue shift
    uHueShift.value = Math.sin(elapsed * params.hueShiftSpeed) * params.hueShiftAmount;

    // Auto-play cycle
    if (params.autoPlay) {
      const total = params.holdDuration + params.cycleDuration + 3.0;
      const ct = elapsed % total;
      if (ct < params.holdDuration) {
        params.dissolveProgress = 0;
      } else if (ct < params.holdDuration + params.cycleDuration) {
        params.dissolveProgress = (ct - params.holdDuration) / params.cycleDuration;
      } else {
        params.dissolveProgress = 1.0;
      }
      uDissolve.value = params.dissolveProgress;
    }

    postProcessing.render();
  }

  renderer.setAnimationLoop(animate);
}

// ─── Resize ───
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

init().catch(err => {
  info.textContent = 'Error: ' + err.message;
  console.error(err);
});
