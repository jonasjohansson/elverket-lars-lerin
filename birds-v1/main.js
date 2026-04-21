import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { DRACOLoader } from 'three/addons/loaders/DRACOLoader.js';
import { ROOM, SURFACE_NAMES, CAMERA_HEIGHT } from './config.js';
import { createStarlingGeometry, createStarlingMaterial } from './birds.js';

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

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.15);
scene.add(ambient);

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

// Load room
const dracoLoader = new DRACOLoader();
dracoLoader.setDecoderPath('https://unpkg.com/three@0.170.0/examples/jsm/libs/draco/');
const gltfLoader = new GLTFLoader();
gltfLoader.setDRACOLoader(dracoLoader);

gltfLoader.load('assets/elverket_v3.glb', (gltf) => {
  const model = gltf.scene;
  model.traverse((child) => {
    if (!child.isMesh) return;
    child.receiveShadow = true;
    if (child.material) {
      child.material = child.material.clone();
      child.material.metalness = 0;
      child.material.roughness = 1;
      if (SURFACE_NAMES.includes(child.name)) {
        child.material.color.set(0xbbbbbb);
      }
    }
  });
  scene.add(model);
  info.textContent = `room loaded · ${SURFACE_NAMES.length} surfaces`;
}, undefined, (err) => {
  info.textContent = 'room load failed';
  console.error(err);
});

// Single test bird at room center
const birdGeo = createStarlingGeometry();
const { material: birdMat, depthMaterial: birdDepth, uniforms: birdUniforms } = createStarlingMaterial();
const testBird = new THREE.Mesh(birdGeo, birdMat);
testBird.customDepthMaterial = birdDepth;
testBird.castShadow = true;
testBird.position.set(ROOM.centerX, ROOM.floor + 3, ROOM.centerZ);
// aPhase attribute must be present even for non-instanced use
birdGeo.setAttribute(
  'aPhase',
  new THREE.BufferAttribute(new Float32Array(birdGeo.attributes.position.count).fill(0), 1)
);
scene.add(testBird);

// Resize
window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

// Loop
const clock = new THREE.Clock();
renderer.setAnimationLoop(() => {
  const t = clock.getElapsedTime();
  birdUniforms.uTime.value = t;
  controls.update();
  renderer.render(scene, camera);
});
