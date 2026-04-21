import * as THREE from 'three';

// Low-poly starling: small elongated body + two triangle wings.
// wingSide attribute: 0 = body, -1 = left wing tip, +1 = right wing tip.
// aPhase (per-instance) shifts each bird's wing flap so the flock isn't in sync.
export function createStarlingGeometry() {
  const g = new THREE.BufferGeometry();

  const S = 0.2; // size scale (meters) — ~40cm wingspan
  const nose  = [ 0.0, 0.0, -0.5 * S];
  const tail  = [ 0.0, 0.0,  0.5 * S];
  const leftShoulder  = [-0.05 * S, 0.0, -0.05 * S];
  const rightShoulder = [ 0.05 * S, 0.0, -0.05 * S];
  const leftWingTip   = [-0.5  * S, 0.0,  0.0      ];
  const rightWingTip  = [ 0.5  * S, 0.0,  0.0      ];
  const leftWingBack  = [-0.3  * S, 0.0,  0.2 * S];
  const rightWingBack = [ 0.3  * S, 0.0,  0.2 * S];

  const positions = [];
  const sides = [];

  // Body tri 1: nose, leftShoulder, rightShoulder
  positions.push(...nose, ...leftShoulder, ...rightShoulder);
  sides.push(0, 0, 0);
  // Body tri 2: leftShoulder, tail, rightShoulder
  positions.push(...leftShoulder, ...tail, ...rightShoulder);
  sides.push(0, 0, 0);
  // Left wing: shoulder, tip, back
  positions.push(...leftShoulder, ...leftWingTip, ...leftWingBack);
  sides.push(0, -1, -1);
  // Right wing: shoulder, back, tip
  positions.push(...rightShoulder, ...rightWingBack, ...rightWingTip);
  sides.push(0, 1, 1);

  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  g.setAttribute('wingSide', new THREE.Float32BufferAttribute(sides, 1));
  g.computeVertexNormals();
  return g;
}

// Returns { material, depthMaterial, uniforms }. Both materials share uniforms
// so shadows and main pass stay in sync.
export function createStarlingMaterial() {
  const uniforms = {
    uTime: { value: 0 },
    uFlapRate: { value: 14.0 },
    uFlapAmplitude: { value: 1.2 },
  };

  const vertexHeader = `
    attribute float wingSide;
    attribute float aPhase;
    uniform float uTime;
    uniform float uFlapRate;
    uniform float uFlapAmplitude;
  `;

  // Wing tips rotate around the Z axis (body forward). abs(wingSide) masks body vertices.
  const vertexTransform = `
    float flap = sin(uTime * uFlapRate + aPhase) * uFlapAmplitude * abs(wingSide);
    float s = sin(flap);
    float c = cos(flap);
    vec3 p = transformed;
    float wx = p.x;
    float wy = p.y;
    float dir = sign(wingSide + 0.0001);
    p.x = wx * c - wy * s * dir;
    p.y = wx * s * dir + wy * c;
    transformed = mix(transformed, p, abs(wingSide));
  `;

  const material = new THREE.MeshStandardMaterial({
    color: 0x222228,
    roughness: 0.9,
    metalness: 0.0,
    side: THREE.DoubleSide,
  });
  material.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + vertexHeader)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + vertexTransform);
  };

  const depthMaterial = new THREE.MeshDepthMaterial({
    depthPacking: THREE.RGBADepthPacking,
    side: THREE.DoubleSide,
  });
  depthMaterial.onBeforeCompile = (shader) => {
    Object.assign(shader.uniforms, uniforms);
    shader.vertexShader = shader.vertexShader
      .replace('#include <common>', '#include <common>\n' + vertexHeader)
      .replace('#include <begin_vertex>', '#include <begin_vertex>\n' + vertexTransform);
  };

  return { material, depthMaterial, uniforms };
}

export class Flock {
  constructor({ count, curve, scene }) {
    this.count = count;
    this.curve = curve;
    this.speed = 2.0;
    this.t = 0;
    this.geo = createStarlingGeometry();
    const mat = createStarlingMaterial();
    this.material = mat.material;
    this.depthMaterial = mat.depthMaterial;
    this.uniforms = mat.uniforms;

    const phases = new Float32Array(count);
    for (let i = 0; i < count; i++) phases[i] = Math.random() * Math.PI * 2;
    this.geo.setAttribute('aPhase', new THREE.InstancedBufferAttribute(phases, 1));

    this.mesh = new THREE.InstancedMesh(this.geo, this.material, count);
    this.mesh.castShadow = true;
    this.mesh.customDepthMaterial = this.depthMaterial;
    this.mesh.frustumCulled = false;
    scene.add(this.mesh);

    this._tmpObj = new THREE.Object3D();
    this._tmpPos = new THREE.Vector3();
    this._tmpTan = new THREE.Vector3();

    this.tOffsets = new Float32Array(count);
    for (let i = 0; i < count; i++) this.tOffsets[i] = (i / count) * 0.08;
  }

  update(dt) {
    this.uniforms.uTime.value += dt;
    const length = this.curve.getLength();
    this.t += (this.speed * dt) / Math.max(length, 0.001);
    if (this.t > 1) this.t -= 1;

    for (let i = 0; i < this.count; i++) {
      let ti = this.t + this.tOffsets[i];
      ti = ti - Math.floor(ti);
      this.curve.getPointAt(ti, this._tmpPos);
      this.curve.getTangentAt(ti, this._tmpTan);
      this._tmpObj.position.copy(this._tmpPos);
      this._tmpObj.lookAt(this._tmpPos.clone().add(this._tmpTan));
      this._tmpObj.updateMatrix();
      this.mesh.setMatrixAt(i, this._tmpObj.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
