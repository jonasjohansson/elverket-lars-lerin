import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { ROOM, SURFACE_NAMES, CAMERA_HEIGHT } from './config.js';
import { Flock } from './birds.js';
import { PathEditor } from './path-editor.js';
import GUI from 'lil-gui';
import { HDRLoader } from 'three/addons/loaders/HDRLoader.js';

const info = document.getElementById('info');
const canvas = document.getElementById('c');

// Renderer
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: 'high-performance' });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.0;
renderer.outputColorSpace = THREE.SRGBColorSpace;

// Scene + camera
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x000000);

const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 200);
camera.position.set(ROOM.centerX, ROOM.floor + CAMERA_HEIGHT, ROOM.centerZ);

const controls = new OrbitControls(camera, canvas);
controls.target.set(ROOM.centerX, ROOM.centerY, ROOM.centerZ);
controls.update();

// WASD + QE movement
const keys = {};
const moveParams = { speed: 3.0 }; // units/sec; Shift doubles
window.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
  keys[e.code] = true;
});
window.addEventListener('keyup', (e) => { keys[e.code] = false; });
window.addEventListener('blur', () => { for (const k in keys) keys[k] = false; });

const _fwd = new THREE.Vector3();
const _right = new THREE.Vector3();
const _move = new THREE.Vector3();
function updateCameraMovement(dt) {
  _move.set(0, 0, 0);
  const forward = (keys['KeyW'] ? 1 : 0) - (keys['KeyS'] ? 1 : 0);
  const strafe  = (keys['KeyD'] ? 1 : 0) - (keys['KeyA'] ? 1 : 0);
  const vert    = (keys['KeyE'] ? 1 : 0) - (keys['KeyQ'] ? 1 : 0);
  if (!forward && !strafe && !vert) return;

  camera.getWorldDirection(_fwd);
  _fwd.y = 0; _fwd.normalize();             // project to XZ so W doesn't dive
  _right.set(_fwd.z, 0, -_fwd.x);            // horizontal right = fwd rotated -90 around Y

  _move.addScaledVector(_fwd, forward);
  _move.addScaledVector(_right, strafe);
  _move.y += vert;

  const boost = keys['ShiftLeft'] || keys['ShiftRight'] ? 2.5 : 1.0;
  _move.multiplyScalar(moveParams.speed * boost * dt);

  camera.position.add(_move);
  controls.target.add(_move);
}

// Lights
const ambientLight = new THREE.AmbientLight(0xffffff, 0.15);
scene.add(ambientLight);

const hemi = new THREE.HemisphereLight(0xffffff, 0x444444, 0.2);
scene.add(hemi);

const dir = new THREE.DirectionalLight(0xffffff, 0.8);
dir.position.set(ROOM.centerX, ROOM.ceiling - 0.5, ROOM.centerZ);
dir.target.position.set(ROOM.centerX, ROOM.floor, ROOM.centerZ);
scene.add(dir.target);
dir.castShadow = true;
dir.shadow.mapSize.set(4096, 4096);
dir.shadow.camera.near = 0.1;
dir.shadow.camera.far = ROOM.height + 5;
const shadowExtent = Math.max(ROOM.width, ROOM.depth) / 2 + 1;
dir.shadow.camera.left   = -shadowExtent;
dir.shadow.camera.right  =  shadowExtent;
dir.shadow.camera.top    =  shadowExtent;
dir.shadow.camera.bottom = -shadowExtent;
dir.shadow.bias = -0.001;
dir.shadow.radius = 3;
scene.add(dir);

let flock = null;
let editor = null;
let birdCount = 300;
let flockParamCtrls = [];
const surfaceMeshesRef = [];
const gui = new GUI({ title: 'birds-v1' });
const sceneFolder = gui.addFolder('Scene');
const pathFolder = gui.addFolder('Path');
const flockFolder = gui.addFolder('Flock');

// HDRI environment
const HDRIS = {
  'Kloppenheim (overcast)': '../shared/hdri/kloppenheim_02_2k.hdr',
  'Venice sunset':          '../shared/hdri/venice_sunset_2k.hdr',
  'Moonless golf (night)':  '../shared/hdri/moonless_golf_2k.hdr',
};
const sceneParams = {
  hdri: 'Kloppenheim (overcast)',
  showBackground: true,
  envIntensity: 1.0,
  exposure: 1.0,
  dirLight: 0.8,
  ambient: 0.15,
  wallBrightness: 0.7,
};
const hdrLoader = new HDRLoader();
let currentEnvTex = null;
function loadHDRI(url) {
  hdrLoader.load(url, (tex) => {
    tex.mapping = THREE.EquirectangularReflectionMapping;
    if (currentEnvTex) currentEnvTex.dispose();
    currentEnvTex = tex;
    scene.environment = tex;
    scene.background = sceneParams.showBackground ? tex : new THREE.Color(0x000000);
  });
}
loadHDRI(HDRIS[sceneParams.hdri]);

sceneFolder.add(sceneParams, 'hdri', Object.keys(HDRIS)).onChange(v => loadHDRI(HDRIS[v]));
sceneFolder.add(sceneParams, 'showBackground').name('show HDRI as sky').onChange(v => {
  scene.background = v && currentEnvTex ? currentEnvTex : new THREE.Color(0x000000);
});
sceneFolder.add(sceneParams, 'envIntensity', 0, 3, 0.05).name('env intensity').onChange(v => {
  scene.traverse(o => { if (o.isMesh && o.material && 'envMapIntensity' in o.material) o.material.envMapIntensity = v; });
});
sceneFolder.add(sceneParams, 'exposure', 0.1, 3, 0.05).onChange(v => { renderer.toneMappingExposure = v; });
sceneFolder.add(sceneParams, 'wallBrightness', 0, 2, 0.05).name('wall brightness').onChange(v => {
  for (const m of surfaceMeshesRef) m.material.color.setScalar(v);
});
sceneFolder.add(sceneParams, 'dirLight', 0, 3, 0.05).name('key light').onChange(v => { dir.intensity = v; });
sceneFolder.add(sceneParams, 'ambient', 0, 2, 0.05).name('ambient').onChange(v => { ambientLight.intensity = v; });
sceneFolder.add(moveParams, 'speed', 0.5, 15, 0.1).name('move speed (WASD)');

const birdAppearance = {
  color: '#222228',
};
const landingCtl = { landed: false };
flockFolder.add({ count: birdCount }, 'count', 50, 800, 10).onFinishChange((v) => {
  birdCount = v;
  if (editor && editor.curve) rebuildFlock(editor.curve);
});
flockFolder.add(landingCtl, 'landed').name('land on floor').onChange(v => {
  if (!flock) return;
  if (v) {
    flock.setMode('landing', {
      floor: ROOM.floor + 0.05,
      bounds: { xMin: ROOM.xMin + 0.5, xMax: ROOM.xMax - 0.5, zMin: ROOM.zMin + 0.5, zMax: ROOM.zMax - 0.5 },
    });
  } else {
    flock.setMode('flying');
  }
});

function rebuildFlock(curve) {
  if (flock) { scene.remove(flock.mesh); flock = null; }
  if (!curve) return;
  flock = new Flock({ count: birdCount, curve, scene });
  flock.material.color.set(birdAppearance.color);
  if (landingCtl.landed) {
    flock.setMode('landing', {
      floor: ROOM.floor + 0.05,
      bounds: { xMin: ROOM.xMin + 0.5, xMax: ROOM.xMax - 0.5, zMin: ROOM.zMin + 0.5, zMax: ROOM.zMax - 0.5 },
    });
  }
  flockParamCtrls.forEach(c => c.destroy());
  flockParamCtrls = [];
  const p = flock.params;
  // Appearance
  const app = flockFolder.addFolder('Appearance');
  flockParamCtrls.push(app);
  app.add(p, 'size', 0.3, 5.0, 0.05).name('size');
  app.addColor(birdAppearance, 'color').name('color').onChange(v => flock.material.color.set(v));
  app.add(flock.uniforms.uFlapRate, 'value', 2, 30, 0.5).name('flap rate');
  app.add(flock.uniforms.uFlapAmplitude, 'value', 0, 2.5, 0.05).name('flap amplitude');
  // Path motion
  const mot = flockFolder.addFolder('Motion');
  flockParamCtrls.push(mot);
  mot.add(p, 'speed', 0.5, 12, 0.1).name('path speed');
  mot.add(p, 'pathWeight', 0, 8, 0.05).name('path pull');
  mot.add(p, 'lateralSpread', 0, 6, 0.05).name('lateral spread');
  mot.add(p, 'wander', 0, 4, 0.05);
  mot.add(p, 'maxSpeed', 1, 15, 0.1).name('max speed');
  mot.add(p, 'minSpeed', 0, 10, 0.1).name('min speed');
  mot.add(p, 'maxForce', 0, 30, 0.5).name('max force');
  // Boids
  const bo = flockFolder.addFolder('Boids');
  flockParamCtrls.push(bo);
  bo.add(p, 'separationRadius', 0.05, 3, 0.05).name('sep radius');
  bo.add(p, 'separationWeight', 0, 4, 0.05).name('sep weight');
  bo.add(p, 'alignmentRadius',  0.1, 5, 0.05).name('ali radius');
  bo.add(p, 'alignmentWeight',  0, 4, 0.05).name('ali weight');
  bo.add(p, 'cohesionRadius',   0.1, 5, 0.05).name('coh radius');
  bo.add(p, 'cohesionWeight',   0, 4, 0.05).name('coh weight');
}

// Load room
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('../shared/vendor/three/examples/jsm/libs/draco/');
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);

gltfLoader.load('assets/elverket_v3.glb', (gltf) => {
  const model = gltf.scene;
  const surfaceMeshes = [];
  model.traverse((child) => {
    if (!child.isMesh) return;
    child.receiveShadow = true;
    if (SURFACE_NAMES.includes(child.name)) {
      // Replace baked material — no grid texture, plain light diffuse wall
      child.material = new THREE.MeshStandardMaterial({
        color: new THREE.Color().setScalar(sceneParams.wallBrightness),
        roughness: 1.0,
        metalness: 0.0,
        envMapIntensity: sceneParams.envIntensity,
        side: THREE.DoubleSide,
      });
      surfaceMeshes.push(child);
      surfaceMeshesRef.push(child);
    } else if (child.material) {
      child.material = child.material.clone();
      child.material.metalness = 0;
      child.material.roughness = 1;
      child.material.envMapIntensity = sceneParams.envIntensity;
    }
  });
  scene.add(model);

  const STORAGE_KEY = 'birds-v1-path';
  const autosave = (curve) => {
    rebuildFlock(curve);
    try { localStorage.setItem(STORAGE_KEY, JSON.stringify(editor.exportJSON())); } catch {}
  };

  editor = new PathEditor({
    scene, camera, renderer, orbitControls: controls, surfaceMeshes,
    onChange: autosave,
  });

  // Load saved path, else default
  let loadedFromStorage = false;
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const data = JSON.parse(raw);
      if (data && Array.isArray(data.points) && data.points.length > 0) {
        editor.importJSON(data);
        loadedFromStorage = true;
      }
    }
  } catch {}
  if (!loadedFromStorage) {
    editor.addFreePoint(new THREE.Vector3(ROOM.xMin - 1, ROOM.floor + 3, ROOM.zMin + 5));
    editor.addFreePoint(new THREE.Vector3(ROOM.xMax - 2, ROOM.floor + 6, ROOM.zMin + 10));
    editor.addFreePoint(new THREE.Vector3(ROOM.xMin + 2, ROOM.floor + 4, ROOM.zMax - 10));
    editor.addFreePoint(new THREE.Vector3(ROOM.xMax + 1, ROOM.floor + 5, ROOM.zMax - 5));
  }

  // GUI: Path controls (attached after editor exists)
  pathFolder.add({ addFree: () => {
    editor.addFreePoint(new THREE.Vector3(ROOM.centerX, ROOM.centerY, ROOM.centerZ));
  } }, 'addFree').name('+ Free-space waypoint');
  pathFolder.add({ clear: () => {
    while (editor.points.length) { scene.remove(editor.points[0].helper); editor.points.shift(); }
    editor._deselect();
    editor._rebuildCurve();
  } }, 'clear').name('Clear all');
  pathFolder.add(editor.curveLine, 'visible').name('Show curve').listen();
  pathFolder.add({ exp: () => {
    const blob = new Blob([JSON.stringify(editor.exportJSON(), null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'birds-path.json'; a.click();
    setTimeout(() => URL.revokeObjectURL(url), 0);
  } }, 'exp').name('Export JSON');
  pathFolder.add({ imp: () => {
    const input = document.createElement('input');
    input.type = 'file'; input.accept = 'application/json';
    input.onchange = () => {
      const file = input.files[0]; if (!file) return;
      file.text().then(t => editor.importJSON(JSON.parse(t)));
    };
    input.click();
  } }, 'imp').name('Import JSON');

  info.textContent = 'WASD + QE to move · drag to orbit · shift-click wall to add waypoint · Del to remove';
}, undefined, (err) => {
  info.textContent = 'room load failed';
  console.error(err);
});

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Loop
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const dt = clock.getDelta();
  updateCameraMovement(dt);
  if (flock) flock.update(dt);
  controls.update();
  renderer.render(scene, camera);
});
