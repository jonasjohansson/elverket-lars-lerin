import * as THREE from 'three';
import { TransformControls } from 'three/addons/controls/TransformControls.js';

export class PathEditor {
  constructor({ scene, camera, renderer, orbitControls, surfaceMeshes, onChange }) {
    this.scene = scene;
    this.camera = camera;
    this.renderer = renderer;
    this.orbit = orbitControls;
    this.surfaceMeshes = surfaceMeshes;
    this.onChange = onChange;

    this.points = [];
    this.markerGeo = new THREE.SphereGeometry(0.15, 16, 12);
    this.colorStart = 0x33dd66;
    this.colorMid   = 0xffffff;
    this.colorEnd   = 0xdd3355;

    this.curveLine = new THREE.Line(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({ color: 0x44aaff })
    );
    this.curveLine.visible = false;
    this.scene.add(this.curveLine);

    this.transform = new TransformControls(camera, renderer.domElement);
    this.transform.addEventListener('dragging-changed', (e) => {
      orbitControls.enabled = !e.value;
    });
    this.transform.addEventListener('change', () => {
      if (!this.selected) return;
      this.selected.position.copy(this.selected.helper.position);
      this._rebuildCurve();
    });
    const transformRoot = this.transform.getHelper ? this.transform.getHelper() : this.transform;
    this.scene.add(transformRoot);
    this.transformRoot = transformRoot;
    this.selected = null;

    this._raycaster = new THREE.Raycaster();
    this._ndc = new THREE.Vector2();

    renderer.domElement.addEventListener('pointerdown', (e) => this._onPointerDown(e));
    window.addEventListener('keydown', (e) => {
      if (e.key === 'Escape') this._deselect();
      if (e.key === 'Delete' || e.key === 'Backspace') this._deleteSelected();
    });
  }

  addFreePoint(pos) {
    this._add({ type: 'free', surface: null, position: pos.clone(), kind: 'approach' });
  }

  addSurfacePoint(surfaceName, pos, normal) {
    this._add({ type: 'surface', surface: surfaceName, position: pos.clone(), normal: normal.clone(), kind: 'approach' });
  }

  _add(p) {
    const mat = new THREE.MeshBasicMaterial({ color: this.colorMid, transparent: true, opacity: 0.9 });
    const mesh = new THREE.Mesh(this.markerGeo, mat);
    mesh.position.copy(p.position);
    mesh.userData.isWaypoint = true;
    this.scene.add(mesh);
    p.helper = mesh;
    this.points.push(p);
    this._updateColors();
    this._rebuildCurve();
  }

  _deleteSelected() {
    if (!this.selected) return;
    this.scene.remove(this.selected.helper);
    this.points = this.points.filter(p => p !== this.selected);
    this.transform.detach();
    this.selected = null;
    this._updateColors();
    this._rebuildCurve();
  }

  _deselect() {
    this.transform.detach();
    this.selected = null;
  }

  _onPointerDown(e) {
    if (e.button !== 0) return;
    if (this.transform.dragging) return;

    const rect = this.renderer.domElement.getBoundingClientRect();
    this._ndc.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
    this._ndc.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
    this._raycaster.setFromCamera(this._ndc, this.camera);

    // First: try hitting an existing waypoint marker
    const wpHits = this._raycaster.intersectObjects(this.points.map(p => p.helper), false);
    if (wpHits.length) {
      const p = this.points.find(pp => pp.helper === wpHits[0].object);
      this._select(p);
      return;
    }

    // Shift-click on a surface adds a waypoint
    if (!e.shiftKey) return;
    const sHits = this._raycaster.intersectObjects(this.surfaceMeshes, false);
    if (sHits.length) {
      const hit = sHits[0];
      const n = hit.face?.normal ?? new THREE.Vector3(0, 1, 0);
      this.addSurfacePoint(hit.object.name, hit.point, n);
    }
  }

  _select(p) {
    this.selected = p;
    this.transform.attach(p.helper);
  }

  _updateColors() {
    this.points.forEach((p, i) => {
      let c = this.colorMid;
      if (i === 0) c = this.colorStart;
      else if (i === this.points.length - 1) c = this.colorEnd;
      p.helper.material.color.setHex(c);
    });
  }

  _rebuildCurve() {
    if (this.points.length < 2) {
      this.curveLine.visible = false;
      this.curve = null;
      if (this.onChange) this.onChange(null);
      return;
    }
    this.curve = new THREE.CatmullRomCurve3(
      this.points.map(p => p.position),
      false, 'catmullrom', 0.5
    );
    const pts = this.curve.getPoints(200);
    this.curveLine.geometry.dispose();
    this.curveLine.geometry = new THREE.BufferGeometry().setFromPoints(pts);
    this.curveLine.visible = true;
    if (this.onChange) this.onChange(this.curve);
  }

  exportJSON() {
    return {
      points: this.points.map(p => ({
        type: p.type,
        surface: p.surface,
        position: p.position.toArray(),
        normal: p.normal ? p.normal.toArray() : null,
        kind: p.kind,
      })),
    };
  }

  importJSON(data) {
    while (this.points.length) { this.scene.remove(this.points[0].helper); this.points.shift(); }
    this._deselect();
    for (const rec of (data.points || [])) {
      if (rec.type === 'surface') {
        this.addSurfacePoint(
          rec.surface,
          new THREE.Vector3().fromArray(rec.position),
          rec.normal ? new THREE.Vector3().fromArray(rec.normal) : new THREE.Vector3(0,1,0),
        );
      } else {
        this.addFreePoint(new THREE.Vector3().fromArray(rec.position));
      }
      this.points[this.points.length-1].kind = rec.kind || 'approach';
    }
  }
}
