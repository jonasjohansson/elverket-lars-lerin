import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { attribute, vec3, vec4, float, Fn, sin, cos, uniform, smoothstep, mix, pass, clamp } from 'three/tsl';
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
camera.position.set(0, 0, 5.0);

const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.enabled = false;
controls.target.set(0, 0, 0);

const PARTICLES_PER_PAINTING = 900_000;
const SAND_COUNT = 500_000;
const PAINTING_HEIGHT = 2.0;
const ALPHA_THRESHOLD = 0.5 * 255;

const PAINTINGS = [
  '0012', '0013', '0018', '0020', '0022', '0029', '0031', '0032', '0038',
].map(n => `../shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_${n}-trim.png`);

const params = {
  waveAmplitude: 0.012,
  waveSpeed: 0.22,
  waveFrequency: 2.0,
  microDrift: 0.0012,
  particleSize: 2.0,
  cameraZ: 5.0,

  // Transition
  transitionDuration: 6.0,
  stormSoftness: 0.55,     // width of the color-blend band around the front, world units
  stormDisplace: 0.45,     // max particle displacement in the storm band
  stormBrightLift: 0.35,   // added luminance near the front
  stormSizeLift: 2.2,      // size multiplier at storm peak
  stormCurlAmp: 0.5,       // curl noise strength in storm band
  stormCurlFreq: 1.5,
  stormChaosAmp: 0.28,     // per-particle chaos in storm band

  // Sand cloud
  sandOpacity: 0.18,       // max airborne sand opacity at storm peak
  sandSize: 1.6,
  sandWindStrength: 1.2,   // lateral advection during storm
  sandCurlAmp: 0.35,

  // Post
  trailStrength: 0.72,
};

const uTime = uniform(0.0);
const uSize = uniform(params.particleSize);

// Ambient drift (stays on at rest)
const uWaveAmp = uniform(params.waveAmplitude);
const uWaveSpd = uniform(params.waveSpeed);
const uWaveFreq = uniform(params.waveFrequency);
const uMicroDrift = uniform(params.microDrift);

// Storm front — world X position of the leading edge. Travels from off-screen-start to off-screen-end.
const uFrontX = uniform(-99.0);           // at rest, parked far off-screen so nothing is passed
const uWindDir = uniform(1.0);            // +1 = front moves right, −1 = front moves left
const uStormSoft = uniform(params.stormSoftness);
const uStormDisplace = uniform(params.stormDisplace);
const uStormBright = uniform(params.stormBrightLift);
const uStormSize = uniform(params.stormSizeLift);
const uStormCurlAmp = uniform(params.stormCurlAmp);
const uStormCurlFreq = uniform(params.stormCurlFreq);
const uStormChaos = uniform(params.stormChaosAmp);

// Sand cloud
const uSandOpacity = uniform(0.0);        // animated: 0 at rest, peak during transition
const uSandSize = uniform(params.sandSize);
const uSandWind = uniform(params.sandWindStrength);
const uSandCurl = uniform(params.sandCurlAmp);

const uTrailStrength = uniform(0.0);      // gated by storm intensity in loop

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

function buildPaintingParticles(pixels, count) {
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

  const shuffled = opaque.slice();
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  const positions = new Float32Array(count * 3);
  const colors = new Float32Array(count * 3);
  const seeds = new Float32Array(count * 4);

  for (let p = 0; p < count; p++) {
    const pixelIdx = p < shuffled.length
      ? shuffled[p]
      : opaque[Math.floor(Math.random() * opaque.length)];
    const px = pixelIdx % iw;
    const py = Math.floor(pixelIdx / iw);
    const rgba = pixelIdx * 4;

    const nx = px / iw, ny = py / ih;
    const jx = (Math.random() - 0.5) * (worldW / iw);
    const jy = (Math.random() - 0.5) * (worldH / ih);
    positions[p * 3]     = (nx - 0.5) * worldW + jx;
    positions[p * 3 + 1] = (0.5 - ny) * worldH + jy;
    positions[p * 3 + 2] = (Math.random() - 0.5) * 0.1;

    colors[p * 3]     = data[rgba]     / 255;
    colors[p * 3 + 1] = data[rgba + 1] / 255;
    colors[p * 3 + 2] = data[rgba + 2] / 255;

    seeds[p * 4]     = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 1] = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 2] = Math.random();
    seeds[p * 4 + 3] = 0.3 + Math.random() * 0.7;
  }

  return { positions, colors, seeds, worldW, worldH };
}

// Airborne sand particles — live in a wider-than-painting box, always present,
// opacity and advection driven by storm uniforms.
function buildSandCloud(count, worldW, worldH) {
  const positions = new Float32Array(count * 3);
  const seeds = new Float32Array(count * 4);
  const boxW = worldW * 2.2;
  const boxH = worldH * 1.4;
  for (let p = 0; p < count; p++) {
    positions[p * 3]     = (Math.random() - 0.5) * boxW;
    positions[p * 3 + 1] = (Math.random() - 0.5) * boxH;
    positions[p * 3 + 2] = (Math.random() - 0.5) * 0.8;

    seeds[p * 4]     = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 1] = (Math.random() - 0.5) * 2;
    seeds[p * 4 + 2] = Math.random();
    seeds[p * 4 + 3] = 0.4 + Math.random() * 0.6;
  }
  return { positions, seeds, boxW, boxH };
}

async function init() {
  await renderer.init();

  info.textContent = `Loading ${PAINTINGS.length} paintings...`;
  const images = await Promise.all(PAINTINGS.map(loadImage));
  const pixels = images.map(img => getPixels(img));
  const builds = pixels.map(px => buildPaintingParticles(px, PARTICLES_PER_PAINTING));

  // Max painting bounds for the sand box and for parking uFrontX off-screen
  const maxWorldW = Math.max(...builds.map(b => b.worldW));
  const maxWorldH = Math.max(...builds.map(b => b.worldH));
  const offscreenDistance = maxWorldW * 1.3;

  // Per-mesh uniforms: uVisible (0/1 gate), uRole (1 = fade-out, 2 = fade-in)
  const meshUniforms = builds.map(() => ({
    uVisible: uniform(0.0),
    uRole: uniform(1.0),
  }));

  function makePaintingMaterial(uVisible, uRole) {
    const m = new THREE.PointsNodeMaterial({
      transparent: true,
      blending: THREE.NormalBlending,
      depthWrite: false,
    });

    // Distance from particle to storm front, along wind direction.
    // passed ∈ [0,1]: 0 = ahead of front (not yet passed), 1 = behind front (already passed).
    // When windDir=+1, passed rises as pos.x drops below uFrontX: pos.x < uFrontX → passed=1.
    const computePassed = (pos) => {
      const signedDist = uWindDir.mul(uFrontX.sub(pos.x));   // >0 if pos.x is behind front
      return smoothstep(uStormSoft.negate(), uStormSoft, signedDist);
    };
    const computeStormBand = (pos) => {
      // 1 at front, 0 at soft-distance away (on either side)
      const absDist = uFrontX.sub(pos.x).mul(uWindDir); // negative ahead, positive behind
      const inBand = float(1.0).sub(smoothstep(float(0.0), uStormSoft.mul(1.3), absDist.abs()));
      return clamp(inBand, 0, 1);
    };

    m.colorNode = Fn(() => {
      const col = attribute('aColor');
      const pos = attribute('position');
      const passed = computePassed(pos);
      const band = computeStormBand(pos);

      // Role 1 (fade-out): visible where passed == 0, fades to 0 where passed == 1
      // Role 2 (fade-in):  hidden where passed == 0, fades to visible where passed == 1
      const isFadeIn = uRole.sub(float(1.0)).clamp(0, 1);              // 0 for role 1, 1 for role 2
      const alphaForRole = mix(float(1.0).sub(passed), passed, isFadeIn);
      const alpha = alphaForRole.mul(uVisible);

      // Brightness lift in storm band
      const brightCol = col.add(vec3(uStormBright).mul(band));
      return vec4(brightCol, alpha);
    })();

    m.sizeNode = Fn(() => {
      const pos = attribute('position');
      const band = computeStormBand(pos);
      return uSize.mul(float(1.0).add(uStormSize.sub(float(1.0)).mul(band)));
    })();
    m.sizeAttenuation = true;

    m.positionNode = Fn(() => {
      const pos = attribute('position');
      const seed = attribute('aSeed');
      const seedX = seed.x, seedY = seed.y;
      const t = uTime;

      // Ambient wave (low amplitude, always on)
      const wavePhase = pos.x.mul(uWaveFreq).add(pos.y.mul(1.5)).add(t.mul(uWaveSpd));
      const wx = sin(wavePhase).mul(uWaveAmp);
      const wy = cos(wavePhase.mul(0.7)).mul(uWaveAmp.mul(0.8));
      const wz = sin(wavePhase.mul(0.5).add(1.0)).mul(uWaveAmp.mul(0.5));
      const dx = sin(t.mul(0.13).add(seedX.mul(7.0))).mul(uMicroDrift);
      const dy = cos(t.mul(0.11).add(seedY.mul(5.0))).mul(uMicroDrift);

      // Storm band displacement — only particles near the front get pushed
      const band = computeStormBand(pos);
      const cf = uStormCurlFreq;
      const ct = t.mul(0.55);
      const curlX = sin(pos.y.mul(cf).add(ct)).mul(cos(pos.z.mul(cf).add(ct.mul(1.3))));
      const curlY = sin(pos.z.mul(cf).add(ct.mul(1.7))).mul(cos(pos.x.mul(cf).add(ct)));
      const curlZ = sin(pos.x.mul(cf).add(ct.mul(0.7))).mul(cos(pos.y.mul(cf).add(ct.mul(1.3))));
      const curlMag = uStormCurlAmp.mul(band);

      const chaosT = t.mul(0.9);
      const chX = sin(chaosT.mul(0.9).add(seedX.mul(11.0))).mul(cos(chaosT.mul(1.1).add(seedY.mul(7.0))));
      const chY = cos(chaosT.mul(1.3).add(seedY.mul(9.0))).mul(sin(chaosT.mul(0.8).add(seedX.mul(8.0))));
      const chZ = sin(chaosT.mul(0.7).add(seedX.add(seedY).mul(5.0)));
      const chMag = uStormChaos.mul(band);

      // Overall bounded displacement
      const displaceMag = uStormDisplace.mul(band);

      return pos.add(vec3(
        wx.add(dx).add(curlX.mul(curlMag)).add(chX.mul(chMag)).add(uWindDir.mul(displaceMag)),
        wy.add(dy).add(curlY.mul(curlMag)).add(chY.mul(chMag)),
        wz.add(curlZ.mul(curlMag)).add(chZ.mul(chMag)),
      ));
    })();

    return m;
  }

  // Build meshes
  const meshes = builds.map((b, i) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(b.positions, 3));
    g.setAttribute('aColor',   new THREE.BufferAttribute(b.colors, 3));
    g.setAttribute('aSeed',    new THREE.BufferAttribute(b.seeds, 4));
    g.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);
    const u = meshUniforms[i];
    const mat = makePaintingMaterial(u.uVisible, u.uRole);
    const m = new THREE.Points(g, mat);
    scene.add(m);
    return m;
  });

  // Sand cloud (separate mesh)
  const sandData = buildSandCloud(SAND_COUNT, maxWorldW, maxWorldH);
  const sandGeo = new THREE.BufferGeometry();
  sandGeo.setAttribute('position', new THREE.BufferAttribute(sandData.positions, 3));
  sandGeo.setAttribute('aSeed',    new THREE.BufferAttribute(sandData.seeds, 4));
  sandGeo.boundingSphere = new THREE.Sphere(new THREE.Vector3(0, 0, 0), 100);

  const sandMat = new THREE.PointsNodeMaterial({
    transparent: true,
    blending: THREE.AdditiveBlending,
    depthWrite: false,
  });
  sandMat.colorNode = Fn(() => {
    const pos = attribute('position');
    const absDist = uFrontX.sub(pos.x).mul(uWindDir).abs();
    // Sand is brightest near the front, fades with distance
    const band = float(1.0).sub(smoothstep(uStormSoft, uStormSoft.mul(3.5), absDist));
    const alpha = uSandOpacity.mul(band);
    // Warm sand tone
    return vec4(0.85, 0.75, 0.55, alpha);
  })();
  sandMat.sizeNode = uSandSize;
  sandMat.sizeAttenuation = true;
  sandMat.positionNode = Fn(() => {
    const pos = attribute('position');
    const seed = attribute('aSeed');
    const seedX = seed.x, seedY = seed.y;
    const t = uTime;

    // Gentle idle drift
    const idleX = sin(t.mul(0.15).add(seedX.mul(4.0))).mul(0.06);
    const idleY = cos(t.mul(0.12).add(seedY.mul(3.0))).mul(0.04);

    // Strong advection toward wind direction near the storm
    const absDist = uFrontX.sub(pos.x).mul(uWindDir).abs();
    const band = float(1.0).sub(smoothstep(uStormSoft.mul(0.5), uStormSoft.mul(3.0), absDist));
    const windX = uWindDir.mul(uSandWind).mul(band);

    // Curl tumble in storm band
    const cf = float(2.2);
    const ct = t.mul(0.8);
    const curlX = sin(pos.y.mul(cf).add(ct)).mul(cos(pos.z.mul(cf).add(ct.mul(1.3))));
    const curlY = sin(pos.z.mul(cf).add(ct.mul(1.7))).mul(cos(pos.x.mul(cf).add(ct)));
    const curlMag = uSandCurl.mul(band);

    // Recycle: wrap sand back through the painting area based on front position so there's
    // always material to see. Simple wrap in X relative to front.
    const relX = pos.x.sub(uFrontX).add(windX).add(curlX.mul(curlMag)).add(idleX);
    const wrappedX = relX.mod(float(maxWorldW * 3.0)).sub(float(maxWorldW * 1.5)).add(uFrontX);

    return vec3(
      wrappedX,
      pos.y.add(idleY).add(curlY.mul(curlMag)),
      pos.z,
    );
  })();
  const sandPoints = new THREE.Points(sandGeo, sandMat);
  scene.add(sandPoints);

  // ─── State ───
  let currentIndex = 0;
  let targetIndex = 0;
  let transitioning = false;
  let transitionStart = 0;
  const TRANSITION_DURATION_ref = { value: params.transitionDuration };

  // Show painting 0 at rest
  meshUniforms[0].uVisible.value = 1.0;
  meshUniforms[0].uRole.value = 1.0;   // fade-out role; park uFrontX so nothing is "passed"
  uFrontX.value = -offscreenDistance;  // before painting's left edge; with windDir=+1, nothing behind front, so fade-out alpha = 1 everywhere

  info.textContent = `${PAINTINGS.length} paintings · ${(PARTICLES_PER_PAINTING / 1e6).toFixed(1)}M painting + ${(SAND_COUNT/1000)|0}k sand · WebGPU`;

  function easeInOut(t) { return t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2; }

  function snapTo(i) {
    if (transitioning) return;
    const clamped = Math.max(0, Math.min(PAINTINGS.length - 1, i));
    if (clamped === currentIndex) return;
    targetIndex = clamped;
    transitionStart = performance.now();
    transitioning = true;

    const dir = Math.sign(targetIndex - currentIndex);
    uWindDir.value = dir;

    // Park front off-screen on the upwind side so nothing has been "passed" yet
    uFrontX.value = dir > 0 ? -offscreenDistance : offscreenDistance;

    // Current fades out, target fades in — both visible
    meshUniforms[currentIndex].uVisible.value = 1.0;
    meshUniforms[currentIndex].uRole.value = 1.0;   // fade-out
    meshUniforms[targetIndex].uVisible.value = 1.0;
    meshUniforms[targetIndex].uRole.value = 2.0;    // fade-in
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

  // ─── GUI ───
  const gui = new GUI({ title: 'Storm Sweep' });

  const fR = gui.addFolder('Rendering');
  fR.add(params, 'particleSize', 0.5, 8, 0.1).name('Particle Size').onChange(v => uSize.value = v);
  fR.add(params, 'cameraZ', 1.5, 20, 0.1).name('Camera Distance').onChange(v => { camera.position.z = v; });

  const fW = gui.addFolder('Ambient');
  fW.add(params, 'waveAmplitude', 0, 0.1, 0.002).name('Wave Amp').onChange(v => uWaveAmp.value = v);
  fW.add(params, 'waveSpeed', 0, 1, 0.01).name('Wave Speed').onChange(v => uWaveSpd.value = v);
  fW.add(params, 'waveFrequency', 0, 8, 0.1).name('Wave Freq').onChange(v => uWaveFreq.value = v);
  fW.add(params, 'microDrift', 0, 0.01, 0.0005).name('Micro Drift').onChange(v => uMicroDrift.value = v);

  const fT = gui.addFolder('Transition');
  fT.add(params, 'transitionDuration', 0.5, 30, 0.1).name('Duration (s)').onChange(v => TRANSITION_DURATION_ref.value = v);
  fT.add(params, 'stormSoftness', 0.05, 2, 0.01).name('Storm Softness').onChange(v => uStormSoft.value = v);

  const fS = gui.addFolder('Storm (painting)');
  fS.add(params, 'stormDisplace', 0, 2, 0.02).name('Displace').onChange(v => uStormDisplace.value = v);
  fS.add(params, 'stormBrightLift', 0, 2, 0.02).name('Brightness').onChange(v => uStormBright.value = v);
  fS.add(params, 'stormSizeLift', 1, 6, 0.05).name('Size Lift').onChange(v => uStormSize.value = v);
  fS.add(params, 'stormCurlAmp', 0, 2, 0.02).name('Curl Amp').onChange(v => uStormCurlAmp.value = v);
  fS.add(params, 'stormCurlFreq', 0.2, 6, 0.1).name('Curl Freq').onChange(v => uStormCurlFreq.value = v);
  fS.add(params, 'stormChaosAmp', 0, 1, 0.02).name('Chaos Amp').onChange(v => uStormChaos.value = v);

  const fSand = gui.addFolder('Sand cloud');
  fSand.add(params, 'sandOpacity', 0, 1, 0.01).name('Max Opacity');
  fSand.add(params, 'sandSize', 0.3, 5, 0.1).name('Size').onChange(v => uSandSize.value = v);
  fSand.add(params, 'sandWindStrength', 0, 4, 0.05).name('Wind').onChange(v => uSandWind.value = v);
  fSand.add(params, 'sandCurlAmp', 0, 2, 0.05).name('Curl').onChange(v => uSandCurl.value = v);

  const fP = gui.addFolder('Post');
  fP.add(params, 'trailStrength', 0, 0.98, 0.01).name('Trails (peak)');

  const actions = {
    next: () => snapTo(currentIndex + 1),
    prev: () => snapTo(currentIndex - 1),
  };
  gui.add(actions, 'prev').name('← Prev');
  gui.add(actions, 'next').name('→ Next');

  // ─── Post-processing (afterimage trails) ───
  const postProcessing = new THREE.PostProcessing(renderer);
  const scenePass = pass(scene, camera);
  const sceneColor = scenePass.getTextureNode('output');
  postProcessing.outputNode = afterImage(sceneColor, uTrailStrength);

  // ─── Animation loop ───
  renderer.setAnimationLoop(() => {
    uTime.value = performance.now() / 1000;

    if (transitioning) {
      const progress = Math.min(1, (performance.now() - transitionStart) / (TRANSITION_DURATION_ref.value * 1000));
      const eased = easeInOut(progress);

      // Travel front from parked position to the opposite off-screen.
      const dir = uWindDir.value;
      const startX = dir > 0 ? -offscreenDistance : offscreenDistance;
      const endX   = dir > 0 ?  offscreenDistance : -offscreenDistance;
      uFrontX.value = startX + (endX - startX) * eased;

      // Storm envelope: bell curve peaking mid-transition
      const envelope = Math.sin(progress * Math.PI);
      uSandOpacity.value = params.sandOpacity * envelope;
      uTrailStrength.value = params.trailStrength * envelope;

      if (progress >= 1) {
        // Finalize: only new painting visible
        meshUniforms[currentIndex].uVisible.value = 0.0;
        meshUniforms[targetIndex].uVisible.value = 1.0;
        meshUniforms[targetIndex].uRole.value = 1.0; // back to fade-out role
        uFrontX.value = -offscreenDistance;          // park so fade-out alpha = 1 everywhere
        uWindDir.value = 1.0;
        uSandOpacity.value = 0;
        uTrailStrength.value = 0;
        currentIndex = targetIndex;
        transitioning = false;
      }
    } else {
      uSandOpacity.value = 0;
      uTrailStrength.value = 0;
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
