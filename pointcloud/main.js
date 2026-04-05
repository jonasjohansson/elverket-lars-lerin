import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { TransformControls } from 'three/addons/controls/TransformControls.js';
import {
  uniform, attribute, float, vec2, vec3, vec4,
  Fn, sin, cos, mix, clamp, smoothstep,
  pass, viewportUV,
} from 'three/tsl';
import { dof } from 'three/addons/tsl/display/DepthOfFieldNode.js';
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
  // Breathing
  breatheAmplitude: 0,
  breatheSpeed: 0.4,
  microDrift: 0.003,
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
  // Vignette
  vignetteStrength: 0.4,
  // DOF
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

// ─── Build main particles ───
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
    // Depth with generous scatter for volume
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

// ─── Build ambient dust particles ───
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
  const uBreatheAmp = uniform(params.breatheAmplitude);
  const uBreatheSpd = uniform(params.breatheSpeed);
  const uMicroDrift = uniform(params.microDrift);
  const uTime = uniform(0.0);
  const uLight1Pos = uniform(new THREE.Vector3(2, 1, 1));
  const uLight1Col = uniform(new THREE.Color(params.light1Color));
  const uLight1Int = uniform(params.light1Intensity);
  const uLight2Pos = uniform(new THREE.Vector3(-2, -0.5, 0.5));
  const uLight2Col = uniform(new THREE.Color(params.light2Color));
  const uLight2Int = uniform(params.light2Intensity);
  const uAmbient = uniform(params.ambientLight);
  const uLightFalloff = uniform(params.lightFalloff);

  // ─── Main particle geometry ───
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(data.positions, 3));
  geometry.setAttribute('aColor', new THREE.BufferAttribute(data.colors, 3));
  geometry.setAttribute('aRandom', new THREE.BufferAttribute(data.randoms, 4));

  const material = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.NormalBlending,
    depthWrite: true,
    depthTest: true,
  });

  // ─── Position: breathing + dissolution ───
  material.positionNode = Fn(() => {
    const pos = attribute('position');
    const rnd = attribute('aRandom');
    const seedX = rnd.x, seedY = rnd.y, delay = rnd.z;
    const t = uTime;

    // Breathing
    const bx = sin(t.mul(uBreatheSpd).add(seedX.mul(3.14))).mul(uBreatheAmp);
    const by = cos(t.mul(uBreatheSpd.mul(0.7)).add(seedY.mul(2.71))).mul(uBreatheAmp);
    const bz = sin(t.mul(uBreatheSpd.mul(0.5)).add(seedX.add(seedY))).mul(uBreatheAmp.mul(0.5));
    const dx = sin(t.mul(0.13).add(seedX.mul(7.0))).mul(uMicroDrift);
    const dy = cos(t.mul(0.11).add(seedY.mul(5.0))).mul(uMicroDrift);

    // Dissolution
    const nx = pos.x.add(1.5).div(3.0).clamp(0, 1);
    const activation = nx.mul(0.35).add(delay.mul(0.35));
    const rawD = uDissolve.mul(1.4).sub(activation).div(float(1.0).sub(activation).add(0.001)).clamp(0, 1);
    const d = rawD.mul(rawD).mul(rawD);

    const windVar = seedX.mul(0.4).add(1.0);
    const windX = d.mul(uWind).mul(windVar).negate();

    const ts1 = sin(t.mul(0.5).add(seedX.mul(2.5))).mul(0.6);
    const ts2 = cos(t.mul(0.4).add(seedY.mul(3.1))).mul(0.5);
    const tm1 = sin(t.mul(1.3).add(seedY.mul(5.0))).mul(0.3);
    const tf = sin(t.mul(3.5).add(seedX.add(seedY).mul(8.0))).mul(0.1);
    const turbX = ts1.add(tm1).add(tf).mul(uTurb).mul(d);
    const turbY = ts2.add(cos(t.mul(1.7).add(seedX.mul(4.2))).mul(0.25)).mul(uTurb).mul(d);

    const angle = d.mul(3.14159).mul(windVar);
    const curlX = sin(angle.add(seedX.mul(6.28))).mul(uCurl).mul(d);
    const curlY = cos(angle.add(seedY.mul(6.28))).mul(uCurl).mul(d);

    const grav = d.mul(d).mul(uGravity).negate();
    const lift = d.mul(0.08).mul(seedY.add(0.5).clamp(0, 1));

    return pos.add(vec3(
      bx.add(dx).add(windX).add(turbX).add(curlX),
      by.add(dy).add(turbY).add(grav).add(lift).add(curlY),
      bz.add(sin(t.mul(0.7).add(seedX.mul(seedY).mul(3.0))).mul(0.15).mul(uTurb).mul(d))
    ));
  })();

  // ─── Size: variation + pulse ───
  material.sizeNode = Fn(() => {
    const rnd = attribute('aRandom');
    const pulse = sin(uTime.mul(1.5).add(rnd.x.mul(6.28))).mul(0.1).add(1.0);
    return rnd.w.mul(uScale).mul(pulse);
  })();
  material.sizeAttenuation = true;

  // ─── Color: image color + dynamic lighting + dissolve fade ───
  material.colorNode = Fn(() => {
    const col = attribute('aColor');
    const rnd = attribute('aRandom');
    const pos = attribute('position');
    const delay = rnd.z;

    // Multiplicative lighting — particles are dark, lights illuminate them
    const toLight1 = uLight1Pos.sub(pos);
    const dist1 = toLight1.length();
    const atten1 = uLight1Int.div(dist1.pow(uLightFalloff).add(0.3));
    const light1 = uLight1Col.mul(atten1);

    const toLight2 = uLight2Pos.sub(pos);
    const dist2 = toLight2.length();
    const atten2 = uLight2Int.div(dist2.pow(uLightFalloff).add(0.3));
    const light2 = uLight2Col.mul(atten2);

    // Ambient + lights = total illumination, multiplied with particle color
    const illumination = vec3(uAmbient, uAmbient, uAmbient).add(light1).add(light2);
    const litCol = col.mul(illumination);

    // Dissolve alpha + warm shift
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

  // ─── Ambient dust particles ───
  const dustData = buildDust();
  const dustGeo = new THREE.BufferGeometry();
  dustGeo.setAttribute('position', new THREE.BufferAttribute(dustData.positions, 3));
  dustGeo.setAttribute('aRandom', new THREE.BufferAttribute(dustData.randoms, 3));

  const uDustOpacity = uniform(params.dustOpacity);
  const uDustSpeed = uniform(params.dustSpeed);

  const dustMat = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });

  dustMat.positionNode = Fn(() => {
    const pos = attribute('position');
    const rnd = attribute('aRandom');
    const t = uTime;
    const sx = sin(t.mul(uDustSpeed).add(rnd.x.mul(5.0))).mul(0.3);
    const sy = cos(t.mul(uDustSpeed.mul(0.7)).add(rnd.y.mul(4.0))).mul(0.2);
    const sz = sin(t.mul(uDustSpeed.mul(0.5)).add(rnd.z.mul(3.0))).mul(0.15);
    // Slow upward drift
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

  const dustPoints = new THREE.Points(dustGeo, dustMat);
  scene.add(dustPoints);

  // ─── Draggable light helpers ───
  const lightSphereGeo = new THREE.SphereGeometry(0.05, 8, 8);
  const light1Helper = new THREE.Mesh(lightSphereGeo, new THREE.MeshBasicMaterial({ color: params.light1Color }));
  const light2Helper = new THREE.Mesh(lightSphereGeo, new THREE.MeshBasicMaterial({ color: params.light2Color }));
  light1Helper.position.copy(uLight1Pos.value);
  light2Helper.position.copy(uLight2Pos.value);
  scene.add(light1Helper);
  scene.add(light2Helper);

  // Transform controls for placing lights
  let activeTransform = null;
  const transform1 = new TransformControls(camera, canvas);
  transform1.attach(light1Helper);
  transform1.setSize(0.5);
  transform1.addEventListener('dragging-changed', (e) => {
    controls.enabled = !e.value;
    if (!e.value) uLight1Pos.value.copy(light1Helper.position);
  });
  transform1.addEventListener('change', () => uLight1Pos.value.copy(light1Helper.position));

  const transform2 = new TransformControls(camera, canvas);
  transform2.attach(light2Helper);
  transform2.setSize(0.5);
  transform2.addEventListener('dragging-changed', (e) => {
    controls.enabled = !e.value;
    if (!e.value) uLight2Pos.value.copy(light2Helper.position);
  });
  transform2.addEventListener('change', () => uLight2Pos.value.copy(light2Helper.position));

  // Toggle: 1/2 keys to select light, Escape to deselect
  let manualLightMode = true; // lights are placeable by default
  const showTransform = (which) => {
    scene.remove(transform1.getHelper());
    scene.remove(transform2.getHelper());
    manualLightMode = true;
    if (which === 1) scene.add(transform1.getHelper());
    if (which === 2) scene.add(transform2.getHelper());
  };
  const hideTransforms = () => {
    scene.remove(transform1.getHelper());
    scene.remove(transform2.getHelper());
    manualLightMode = false;
  };

  // ─── Post-processing: DOF + vignette ───
  const uVignetteStrength = uniform(params.vignetteStrength);
  const uDofFocus = uniform(params.dofFocus);
  const uDofAperture = uniform(params.dofAperture);
  const uDofMaxBlur = uniform(params.dofMaxBlur);

  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  const sceneViewZ = scenePass.getViewZNode();

  // Bokeh DOF
  const dofPass = dof(sceneColor, sceneViewZ, uDofFocus, uDofAperture, uDofMaxBlur);

  // Vignette
  const uv = viewportUV;
  const vigDist = uv.sub(0.5).length();
  const vignette = clamp(vigDist.mul(2.0), 0.0, 1.0);
  const vignetteFactor = float(1.0).sub(vignette.mul(vignette).mul(uVignetteStrength));

  postProcessing.outputNode = dofPass.mul(vignetteFactor);

  info.textContent = `${(PARTICLE_COUNT / 1e6).toFixed(1)}M · WebGPU · Space: dissolve · R: reset · 1/2: drag lights · Esc: hide gizmo`;

  // ─── GUI ───
  const gui = new GUI({ title: 'Point Cloud' });

  const fR = gui.addFolder('Rendering');
  fR.add(params, 'splatScale', 0.5, 15, 0.25).name('Scale').onChange(v => uScale.value = v);
  fR.add(params, 'opacity', 0, 1, 0.01).name('Opacity').onChange(v => uOpacity.value = v);
  fR.addColor(params, 'bgColor').name('Background').onChange(v => renderer.setClearColor(v));

  const fB = gui.addFolder('Breathing');
  fB.add(params, 'breatheAmplitude', 0, 0.1, 0.001).name('Amplitude').onChange(v => uBreatheAmp.value = v);
  fB.add(params, 'breatheSpeed', 0, 2, 0.05).name('Speed').onChange(v => uBreatheSpd.value = v);
  fB.add(params, 'microDrift', 0, 0.02, 0.001).name('Micro Drift').onChange(v => uMicroDrift.value = v);

  const fDepth = gui.addFolder('Depth');
  fDepth.add(params, 'depthNear', -2, 5, 0.1).name('Near (white)');
  fDepth.add(params, 'depthFar', -10, 2, 0.1).name('Far (black)');
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
  fLights.addColor(params, 'light1Color').name('Light 1 Color').onChange(v => { uLight1Col.value.set(v); light1Helper.material.color.set(v); });
  fLights.add(params, 'light1Intensity', 0, 5, 0.1).name('Light 1 Power').onChange(v => uLight1Int.value = v);
  fLights.addColor(params, 'light2Color').name('Light 2 Color').onChange(v => { uLight2Col.value.set(v); light2Helper.material.color.set(v); });
  fLights.add(params, 'light2Intensity', 0, 5, 0.1).name('Light 2 Power').onChange(v => uLight2Int.value = v);
  fLights.add(params, 'lightFalloff', 0.5, 3, 0.1).name('Falloff').onChange(v => uLightFalloff.value = v);

  const fDof = gui.addFolder('Depth of Field');
  fDof.add(params, 'dofFocus', 0.1, 50, 0.1).name('Focus Distance').onChange(v => uDofFocus.value = v);
  fDof.add(params, 'dofAperture', 0, 0.15, 0.001).name('Aperture').onChange(v => uDofAperture.value = v);
  fDof.add(params, 'dofMaxBlur', 0, 0.08, 0.001).name('Max Blur').onChange(v => uDofMaxBlur.value = v);

  const fFx = gui.addFolder('FX');
  fFx.add(params, 'vignetteStrength', 0, 1.5, 0.05).name('Vignette').onChange(v => uVignetteStrength.value = v);
  fFx.add(params, 'dustOpacity', 0, 0.5, 0.01).name('Dust').onChange(v => uDustOpacity.value = v);
  fFx.add(params, 'dustSpeed', 0, 0.3, 0.005).name('Dust Speed').onChange(v => uDustSpeed.value = v);

  const fA = gui.addFolder('Auto');
  fA.add(params, 'autoPlay').name('Auto Cycle');
  fA.add(params, 'cycleDuration', 4, 30, 0.5).name('Dissolve (s)');
  fA.add(params, 'holdDuration', 1, 15, 0.5).name('Hold (s)');
  fA.add(params, 'autoRotate').name('Auto Rotate').onChange(v => controls.autoRotate = v);
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

  // Keyboard
  window.addEventListener('keydown', (e) => {
    if (e.code === 'Space') { e.preventDefault(); actions.playDissolve(); }
    if (e.code === 'KeyR') actions.reset();
    if (e.key === 'f' || e.key === 'F') {
      if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
      else document.exitFullscreen().catch(() => {});
    }
    // 1/2: select light to drag, Escape: deselect
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


    // Auto-play
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
