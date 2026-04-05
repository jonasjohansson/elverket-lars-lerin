import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import {
  uniform, attribute, float, vec2, vec3, vec4,
  Fn, sin, cos, mix, clamp, smoothstep,
  pass, viewportUV,
} from 'three/tsl';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
import { afterImage } from 'three/addons/tsl/display/AfterImageNode.js';
import GUI from 'https://cdn.jsdelivr.net/npm/lil-gui@0.20/+esm';

// ─── Config ───
const PARTICLE_COUNT = 2_500_000;
const DUST_COUNT = 50_000;
const COLOR_IMG = '../shared/images/lerin2.jpg';
const DEPTH_IMG = '../shared/images/lerin2_depth.png';
const info = document.getElementById('info');

// ─── Params ───
const params = {
  splatScale: 3.5,
  opacity: 1.0,
  bgColor: '#000000',
  depthNear: 1.5,
  depthFar: -4.0,
  depthCurve: 0.8,
  depthScatter: 0.5,
  // Waves (continuous coordinated motion)
  waveAmplitude: 0.04,
  waveSpeed: 0.3,
  waveFrequency: 2.0,
  wave2Amplitude: 0.025,
  wave2Speed: 0.2,
  wave2Frequency: 3.5,
  microDrift: 0.003,
  // Morphing state (0=calm, 0.5=wave, 1=turbulent)
  morphState: 0.0,
  morphSpeed: 0.08, // how fast it drifts between states
  autoMorph: true,
  // Dissolution
  dissolveProgress: 0.0,
  windStrength: 3.0,
  gravity: 0.2,
  turbulence: 0.5,
  curlStrength: 0.8,
  // Lights
  ambientLight: 0.7,
  light1Color: '#ffffff',
  light1Intensity: 0.5,
  light2Color: '#ffffff',
  light2Intensity: 0.3,
  lightFalloff: 1.5,
  // Color shift
  hueShiftSpeed: 0.02,
  hueShiftAmount: 0.08,
  // Post FX
  vignetteStrength: 0.4,
  trailStrength: 0.0, // 0=off, 0.9=heavy trails
  dofFocus: 4.8,
  dofAperture: 0.003,
  dofMaxBlur: 0.015,
  // Dust
  dustOpacity: 0.15,
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

function getPixels(img, maxW = 1600) {
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
  const [colorImg, depthImg] = await Promise.all([loadImage(COLOR_IMG), loadImage(DEPTH_IMG)]);
  const colorData = getPixels(colorImg);
  const depthData = getPixels(depthImg);
  const { width: imgW, height: imgH } = colorData;

  info.textContent = `Building ${(PARTICLE_COUNT / 1e6).toFixed(1)}M particles...`;
  await new Promise(r => requestAnimationFrame(r));

  const aspect = imgW / imgH;
  const sceneH = 2.0, sceneW = sceneH * aspect;
  const cPx = colorData.data, dPx = depthData.data;
  const jX = sceneW / imgW, jY = sceneH / imgH;

  const positions = new Float32Array(PARTICLE_COUNT * 3);
  const colors = new Float32Array(PARTICLE_COUNT * 3);
  const randoms = new Float32Array(PARTICLE_COUNT * 4);

  for (let i = 0; i < PARTICLE_COUNT; i++) {
    const px = Math.random() * imgW | 0;
    const py = Math.random() * imgH | 0;
    const rgba = (py * imgW + px) * 4;
    const nx = px / imgW, ny = py / imgH;

    const scatter = 2.0;
    positions[i * 3] = (nx - 0.5) * sceneW + (Math.random() - 0.5) * jX * scatter;
    positions[i * 3 + 1] = (0.5 - ny) * sceneH + (Math.random() - 0.5) * jY * scatter;
    const rawD = dPx[rgba] / 255;
    const d = Math.pow(rawD, params.depthCurve);
    const depthBase = d * (params.depthNear - params.depthFar) + params.depthFar;
    const zScatter = (Math.random() - 0.5) * params.depthScatter * (1 + d);
    positions[i * 3 + 2] = depthBase + zScatter;

    let cr = cPx[rgba] / 255, cg = cPx[rgba + 1] / 255, cb = cPx[rgba + 2] / 255;
    const w = (Math.random() - 0.4) * 0.06;
    cr = Math.max(0, Math.min(1, cr + w));
    cg = Math.max(0, Math.min(1, cg + w * 0.4));
    cb = Math.max(0, Math.min(1, cb - w * 0.3));
    const lum = cr * 0.299 + cg * 0.587 + cb * 0.114;
    const sat = 1.0 + (Math.random() - 0.5) * 0.25;
    colors[i * 3] = Math.max(0, Math.min(1, lum + (cr - lum) * sat));
    colors[i * 3 + 1] = Math.max(0, Math.min(1, lum + (cg - lum) * sat));
    colors[i * 3 + 2] = Math.max(0, Math.min(1, lum + (cb - lum) * sat));

    randoms[i * 4] = (Math.random() - 0.5) * 2;
    randoms[i * 4 + 1] = (Math.random() - 0.5) * 2;
    randoms[i * 4 + 2] = Math.random();
    randoms[i * 4 + 3] = 0.3 + Math.random() * 0.7;
  }

  return { positions, colors, randoms };
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

    return pos.add(vec3(
      w1x.add(w2x).add(w3x).add(dx).add(morphTurbX).add(windX).add(dissolveTurbX).add(curlX),
      w1y.add(w2y).add(w3y).add(dy).add(morphTurbY).add(dissolveTurbY).add(grav).add(lift).add(curlY),
      w1z.add(w2z).add(morphTurbZ).add(sin(t.mul(0.7).add(seedX.mul(seedY).mul(3.0))).mul(0.15).mul(uTurb).mul(d))
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

    // Dissolve fade + warm shift
    const nx = pos.x.add(1.5).div(3.0).clamp(0, 1);
    const activation = nx.mul(0.35).add(delay.mul(0.35));
    const rawD = uDissolve.mul(1.4).sub(activation).div(float(1.0).sub(activation).add(0.001)).clamp(0, 1);
    const fadeAlpha = float(1.0).sub(smoothstep(float(0.3), float(1.0), rawD));
    const warmShift = rawD.mul(0.12);
    const finalCol = vec3(litCol.x.add(warmShift), litCol.y, litCol.z.sub(warmShift.mul(0.5)));

    return vec4(finalCol, uOpacity.mul(fadeAlpha));
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

  info.textContent = `${(PARTICLE_COUNT / 1e6).toFixed(1)}M · WebGPU · Space: dissolve · R: reset`;

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
    resetCamera: () => { camera.position.set(0, 0, 3.8); controls.target.set(0, 0, -0.3); controls.update(); },
  };
  gui.add(actions, 'playDissolve').name('▶ Play');
  gui.add(actions, 'reset').name('↺ Reset');
  gui.add(actions, 'resetCamera').name('⌂ Camera');

  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); actions.playDissolve(); }
    if (e.code === 'KeyR') actions.reset();
    if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen().catch(() => {});
    }
    if (e.key === '1') showTransform(1);
    if (e.key === '2') showTransform(2);
    if (e.key === 'Escape') hideTransforms();
  });

  // ─── Render loop ───
  const clock = new THREE.Clock();

  function animate() {
    const elapsed = clock.getElapsedTime();
    uTime.value = elapsed;
    controls.update();

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
