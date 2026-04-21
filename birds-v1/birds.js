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

    this.params = {
      speed: 1.5,
      separationRadius: 0.5,
      separationWeight: 1.6,
      alignmentRadius: 1.2,
      alignmentWeight: 1.0,
      cohesionRadius: 1.6,
      cohesionWeight: 0.8,
      pathWeight: 2.0,
      maxForce: 6.0,
      maxSpeed: 2.5,
      minSpeed: 0.8,
      wander: 0.3,          // random acceleration strength
      size: 1.0,             // scales each instance (1.0 = ~40cm wingspan)
      lateralSpread: 2.0,    // how loose the cloud around the path is (applied at update time)
      landSpread: 6.0,       // how far apart birds spread when landing
      landDescentSpeed: 3.0, // how hard they pull toward the floor target
    };

    // Mode: 'flying' | 'landing' | 'landed'
    // Per-bird `landed` flag used so each bird settles independently when it arrives.
    this.mode = 'flying';
    this.landTargets = new Float32Array(count * 3);
    this.landed = new Uint8Array(count);

    this.t = 0;

    this.positions      = new Float32Array(count * 3);
    this.velocities     = new Float32Array(count * 3);
    this.tOffsets       = new Float32Array(count);
    this.lateralOffsets = new Float32Array(count * 3);

    // Unit-magnitude random offsets; scaled by params.lateralSpread at update time
    const start = curve.getPointAt(0);
    for (let i = 0; i < count; i++) {
      this.tOffsets[i] = (i / count) * 0.12;
      this.lateralOffsets[i*3+0] = Math.random() - 0.5;
      this.lateralOffsets[i*3+1] = Math.random() - 0.5;
      this.lateralOffsets[i*3+2] = Math.random() - 0.5;
      const sp = this.params.lateralSpread;
      this.positions[i*3+0] = start.x + this.lateralOffsets[i*3+0] * sp;
      this.positions[i*3+1] = start.y + this.lateralOffsets[i*3+1] * sp;
      this.positions[i*3+2] = start.z + this.lateralOffsets[i*3+2] * sp;
      this.velocities[i*3+0] = (Math.random() - 0.5) * 2;
      this.velocities[i*3+1] = (Math.random() - 0.5) * 2;
      this.velocities[i*3+2] = (Math.random() - 0.5) * 2;
    }

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
    this._target = new THREE.Vector3();
    this._tmp    = new THREE.Vector3();
  }

  // Switch to landing/flying. `floor` = y-coordinate of the ground plane.
  // `bounds` = { xMin, xMax, zMin, zMax } inside which birds may land.
  setMode(mode, { floor, bounds } = {}) {
    if (mode === this.mode) return;
    if (mode === 'landing') {
      const spread = this.params.landSpread;
      for (let i = 0; i < this.count; i++) {
        // Uniform random inside provided bounds, with a little extra jitter
        const jx = (Math.random() - 0.5) * spread;
        const jz = (Math.random() - 0.5) * spread;
        const cx = bounds ? (bounds.xMin + bounds.xMax) / 2 : 0;
        const cz = bounds ? (bounds.zMin + bounds.zMax) / 2 : 0;
        const xRange = bounds ? (bounds.xMax - bounds.xMin) / 2 - 0.5 : 5;
        const zRange = bounds ? (bounds.zMax - bounds.zMin) / 2 - 0.5 : 5;
        this.landTargets[i*3+0] = cx + (Math.random() - 0.5) * 2 * xRange + jx * 0.1;
        this.landTargets[i*3+1] = floor;
        this.landTargets[i*3+2] = cz + (Math.random() - 0.5) * 2 * zRange + jz * 0.1;
        this.landed[i] = 0;
      }
    } else if (mode === 'flying') {
      for (let i = 0; i < this.count; i++) this.landed[i] = 0;
    }
    this.mode = mode;
  }

  _buildGrid() {
    const cell = Math.max(this.params.alignmentRadius, this.params.cohesionRadius);
    const grid = new Map();
    const invCell = 1 / cell;
    for (let i = 0; i < this.count; i++) {
      const x = Math.floor(this.positions[i*3+0] * invCell);
      const y = Math.floor(this.positions[i*3+1] * invCell);
      const z = Math.floor(this.positions[i*3+2] * invCell);
      const key = x + ',' + y + ',' + z;
      let bucket = grid.get(key);
      if (!bucket) { bucket = []; grid.set(key, bucket); }
      bucket.push(i);
    }
    return { grid, invCell };
  }

  _neighbors(i, grid, invCell, cb) {
    const x = Math.floor(this.positions[i*3+0] * invCell);
    const y = Math.floor(this.positions[i*3+1] * invCell);
    const z = Math.floor(this.positions[i*3+2] * invCell);
    for (let dx = -1; dx <= 1; dx++)
      for (let dy = -1; dy <= 1; dy++)
        for (let dz = -1; dz <= 1; dz++) {
          const bucket = grid.get((x+dx)+','+(y+dy)+','+(z+dz));
          if (!bucket) continue;
          for (const j of bucket) if (j !== i) cb(j);
        }
  }

  update(dt) {
    this.uniforms.uTime.value += dt;
    const p = this.params;

    if (this.mode === 'flying') {
      const length = this.curve.getLength();
      this.t += (p.speed * dt) / Math.max(length, 0.001);
      const slack = Math.max.apply(null, Array.from(this.tOffsets)) + 0.05;
      if (this.t > 1 + slack) {
        this.t = 0;
        const start = this.curve.getPointAt(0);
        for (let i = 0; i < this.count; i++) {
          this.positions[i*3+0] = start.x + this.lateralOffsets[i*3+0];
          this.positions[i*3+1] = start.y + this.lateralOffsets[i*3+1];
          this.positions[i*3+2] = start.z + this.lateralOffsets[i*3+2];
        }
      }
    }

    const { grid, invCell } = this._buildGrid();

    const sepR2 = p.separationRadius * p.separationRadius;
    const aliR2 = p.alignmentRadius * p.alignmentRadius;
    const cohR2 = p.cohesionRadius * p.cohesionRadius;

    for (let i = 0; i < this.count; i++) {
      const px = this.positions[i*3+0], py = this.positions[i*3+1], pz = this.positions[i*3+2];
      let vx = this.velocities[i*3+0], vy = this.velocities[i*3+1], vz = this.velocities[i*3+2];

      // Bird already landed — sit still and write a flat-pose matrix
      if (this.landed[i]) {
        this._tmpObj.position.set(px, py, pz);
        this._tmpObj.rotation.set(0, this.lateralOffsets[i*3+0] * Math.PI * 2, 0); // random yaw per bird
        this._tmpObj.scale.setScalar(p.size);
        this._tmpObj.updateMatrix();
        this.mesh.setMatrixAt(i, this._tmpObj.matrix);
        continue;
      }

      let ax = 0, ay = 0, az = 0;

      // Separation (useful in both modes — prevents overlap)
      let sepX=0, sepY=0, sepZ=0, sepN=0;
      let aliX=0, aliY=0, aliZ=0, aliN=0;
      let cohX=0, cohY=0, cohZ=0, cohN=0;

      this._neighbors(i, grid, invCell, (j) => {
        const dx = px - this.positions[j*3+0];
        const dy = py - this.positions[j*3+1];
        const dz = pz - this.positions[j*3+2];
        const d2 = dx*dx + dy*dy + dz*dz;
        if (d2 < sepR2 && d2 > 1e-6) {
          const inv = 1 / Math.sqrt(d2);
          sepX += dx * inv; sepY += dy * inv; sepZ += dz * inv; sepN++;
        }
        if (this.mode === 'flying') {
          if (d2 < aliR2) {
            aliX += this.velocities[j*3+0];
            aliY += this.velocities[j*3+1];
            aliZ += this.velocities[j*3+2];
            aliN++;
          }
          if (d2 < cohR2) {
            cohX += this.positions[j*3+0];
            cohY += this.positions[j*3+1];
            cohZ += this.positions[j*3+2];
            cohN++;
          }
        }
      });

      if (sepN > 0) { ax += (sepX/sepN) * p.separationWeight; ay += (sepY/sepN) * p.separationWeight; az += (sepZ/sepN) * p.separationWeight; }

      if (this.mode === 'flying') {
        if (aliN > 0) { ax += (aliX/aliN - vx) * p.alignmentWeight; ay += (aliY/aliN - vy) * p.alignmentWeight; az += (aliZ/aliN - vz) * p.alignmentWeight; }
        if (cohN > 0) { ax += (cohX/cohN - px) * p.cohesionWeight;  ay += (cohY/cohN - py) * p.cohesionWeight;  az += (cohZ/cohN - pz) * p.cohesionWeight; }

        let ti = this.t + this.tOffsets[i];
        if (ti > 1) ti = 1;
        this.curve.getPointAt(ti, this._target);
        const ls = p.lateralSpread;
        const lox = this.lateralOffsets[i*3+0] * ls;
        const loy = this.lateralOffsets[i*3+1] * ls;
        const loz = this.lateralOffsets[i*3+2] * ls;
        ax += (this._target.x + lox - px) * p.pathWeight;
        ay += (this._target.y + loy - py) * p.pathWeight;
        az += (this._target.z + loz - pz) * p.pathWeight;

        if (p.wander > 0) {
          ax += (Math.random() - 0.5) * p.wander * 2;
          ay += (Math.random() - 0.5) * p.wander * 2;
          az += (Math.random() - 0.5) * p.wander * 2;
        }
      } else if (this.mode === 'landing') {
        // Strong pull toward this bird's floor target; slight velocity damping
        const tx = this.landTargets[i*3+0];
        const ty = this.landTargets[i*3+1];
        const tz = this.landTargets[i*3+2];
        ax += (tx - px) * p.landDescentSpeed;
        ay += (ty - py) * p.landDescentSpeed;
        az += (tz - pz) * p.landDescentSpeed;
        // Air brake — damp horizontal velocity near the ground
        const heightAbove = py - ty;
        if (heightAbove < 1.5) {
          const brake = Math.max(0, 1 - heightAbove / 1.5);
          ax -= vx * brake * 3;
          az -= vz * brake * 3;
        }
      }

      const aLen = Math.sqrt(ax*ax + ay*ay + az*az);
      if (aLen > p.maxForce) { const s = p.maxForce / aLen; ax *= s; ay *= s; az *= s; }

      vx += ax * dt; vy += ay * dt; vz += az * dt;

      // Speed clamp — skip minSpeed during landing so birds can fully stop
      const sp = Math.sqrt(vx*vx + vy*vy + vz*vz);
      if (sp > p.maxSpeed) { const s = p.maxSpeed / sp; vx *= s; vy *= s; vz *= s; }
      else if (this.mode === 'flying' && sp < p.minSpeed && sp > 1e-3) {
        const s = p.minSpeed / sp; vx *= s; vy *= s; vz *= s;
      }

      this.velocities[i*3+0] = vx;
      this.velocities[i*3+1] = vy;
      this.velocities[i*3+2] = vz;

      this.positions[i*3+0] = px + vx * dt;
      this.positions[i*3+1] = py + vy * dt;
      this.positions[i*3+2] = pz + vz * dt;

      // Landing settle — snap when close enough
      if (this.mode === 'landing') {
        const dx = this.positions[i*3+0] - this.landTargets[i*3+0];
        const dy = this.positions[i*3+1] - this.landTargets[i*3+1];
        const dz = this.positions[i*3+2] - this.landTargets[i*3+2];
        if (dx*dx + dy*dy + dz*dz < 0.04) { // ~20cm
          this.positions[i*3+0] = this.landTargets[i*3+0];
          this.positions[i*3+1] = this.landTargets[i*3+1];
          this.positions[i*3+2] = this.landTargets[i*3+2];
          this.velocities[i*3+0] = 0; this.velocities[i*3+1] = 0; this.velocities[i*3+2] = 0;
          this.landed[i] = 1;
        }
      }

      this._tmp.set(this.positions[i*3+0] + vx, this.positions[i*3+1] + vy, this.positions[i*3+2] + vz);
      this._tmpObj.position.set(this.positions[i*3+0], this.positions[i*3+1], this.positions[i*3+2]);
      this._tmpObj.lookAt(this._tmp);
      this._tmpObj.scale.setScalar(p.size);
      this._tmpObj.updateMatrix();
      this.mesh.setMatrixAt(i, this._tmpObj.matrix);
    }
    this.mesh.instanceMatrix.needsUpdate = true;
  }
}
