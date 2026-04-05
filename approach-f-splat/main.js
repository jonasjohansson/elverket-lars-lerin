import * as GaussianSplats3D from '@mkkellogg/gaussian-splats-3d';

const SPLAT_PATH = '../shared/splats/lerin1.splat';
const loading = document.getElementById('loading');

const viewer = new GaussianSplats3D.Viewer({
  selfDrivenMode: true,
  useBuiltInControls: true,
  cameraUp: [0, 1, 0],
  initialCameraPosition: [0, 0, 10],
  initialCameraLookAt: [0, 0, 2],
  sphericalHarmonicsDegree: 0,
  gpuAcceleratedSort: true,
  sceneRevealMode: GaussianSplats3D.SceneRevealMode.Instant,
  renderMode: GaussianSplats3D.RenderMode.Always,
});

viewer.addSplatScene(SPLAT_PATH, {
  splatAlphaRemovalThreshold: 1,
  showLoadingUI: false,
  progressiveLoad: false,
})
.then(() => {
  loading.classList.add('hidden');
  viewer.start();

  // Debug: check if splats are actually being rendered
  const info = viewer.renderer.info;
  console.log('Renderer info:', JSON.stringify(info.render));
  console.log('Programs:', info.programs?.length);

  // Try moving camera to various positions to find the splats
  const positions = [
    [0, 0, 10], [0, 0, -10], [0, 0, 5], [0, 0, 20],
    [10, 0, 2], [-10, 0, 2], [0, 5, 2], [0, -5, 2],
  ];
  positions.forEach(([x, y, z]) => {
    viewer.camera.position.set(x, y, z);
    viewer.camera.lookAt(0, 0, 1.7);
    viewer.camera.updateProjectionMatrix();
  });

  // Reset to front view
  viewer.camera.position.set(0, 0, 10);
  viewer.camera.lookAt(0, 0, 1.7);

  // After a few frames check draw calls
  setTimeout(() => {
    const ri = viewer.renderer.info;
    console.log('After frames - draw calls:', ri.render.calls, 'triangles:', ri.render.triangles, 'points:', ri.render.points);

    // Check the splat mesh
    viewer.threeScene.traverse((obj) => {
      console.log('Scene object:', obj.type, obj.visible, obj.geometry?.attributes ? Object.keys(obj.geometry.attributes) : 'no geometry');
      if (obj.material) {
        console.log('  Material:', obj.material.type, 'visible:', obj.material.visible, 'opacity:', obj.material.opacity);
      }
    });
  }, 1000);
})
.catch(err => {
  console.error('Load error:', err);
  loading.textContent = 'Error: ' + err.message;
});

window.addEventListener('keydown', (e) => {
  if (e.key === 'f' || e.key === 'F') {
    if (!document.fullscreenElement) document.documentElement.requestFullscreen().catch(() => {});
    else document.exitFullscreen().catch(() => {});
  }
});
