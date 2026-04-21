# Birds v1 — Starling flock through painting waypoints

**Date:** 2026-04-21
**Target:** `elverket-lars-lerin/birds-v1/` (standalone prototype)

## Goal

Hundreds of low-poly 3D starlings flocking through the Elverket gallery room (copied from the `/elverket/` repo as reference), following a user-authored 3D spline whose control points sit on the wall paintings or in free space. Birds cast real shadows on the floor. Continuous migration loop: as the last bird exits, a new flock spawns.

## Scope & non-goals

**In scope**
- Low-poly starling geometry, instanced, GPU-animated wing flap
- Hybrid flocking: boids + path attractor
- 3D spline (Catmull-Rom) through user-placed waypoints
- 3D click-to-place + TransformControls drag + lil-gui list editor
- Shadow casting on floor via existing directional light setup
- Continuous flock spawn/exit loop

**Out of scope (v1)**
- Skeletal animation / imported bird GLTF
- Shadows on walls / on painting projections
- Multiple simultaneous flocks (staggered waves)
- Real-time path editing via VR controllers
- Sound / audio cues
- Painting-surface content — walls stay neutral grey; paintings will come from the existing projection pipeline elsewhere

## Architecture

```
elverket-lars-lerin/birds-v1/
  index.html         importmap (three + addons + lil-gui), canvas
  main.js            scene setup, room load, RAF loop, GUI wiring
  birds.js           InstancedMesh of starlings, boids sim, wing-flap shader
  path-editor.js     click-to-place, TransformControls, curve rendering
  config.js          ROOM dims + surface names (copied from /elverket/)
  assets/
    elverket_v3.glb  copied from /elverket/
```

**Renderer:** `THREE.WebGLRenderer`, PCF soft shadow map, 4096² shadow resolution (same as elverket repo). Not WebGPU — shadows are less battle-tested there.

**Integration:** no cross-repo imports. The GLB and the subset of `config.js` we need (ROOM dims, SURFACE_NAMES) are copied into `birds-v1/`.

## Components

### 1. Scene & room

- Load `assets/elverket_v3.glb` via `GLTFLoader`, traverse and set `receiveShadow = true` on all surfaces
- Directional light from ceiling center pointing at floor, `castShadow = true`, shadow frustum sized to room extent
- Hemisphere fill light for soft ambient, matching elverket repo values
- Orbit controls constrained to roughly-inside-the-room for authoring

### 2. Path editor (`path-editor.js`)

**Data model:**
```js
path = [
  { type: 'surface', surface: 'Front_LongWallA', point: Vec3, normal: Vec3, kind: 'approach' },
  { type: 'free',    point: Vec3, kind: 'orbit' },
  ...
]
```

Waypoint kinds: `approach` (pass by), `dive-through` (cross the surface plane), `orbit` (N loops then continue), `exit` (final — flock dissolves past this).

**Placement:**
- Click on a surface mesh → raycast → push a `surface` waypoint at hit point/normal
- lil-gui "+ Free-space" button → push a `free` waypoint at room center, selected and ready to drag

**Editing:**
- Each waypoint renders as a colored sphere + floating index sprite
- Click a waypoint sphere → `TransformControls` attaches for drag
- On drop, surface-type points re-raycast along nearest wall normal to stay glued to a painting
- lil-gui "Path" folder: ordered list, per-row ↑ ↓ × and kind dropdown

**Curve:**
- `THREE.CatmullRomCurve3` built from all waypoint points, rebuilt on any edit
- Rendered as a thick tube-like line, toggleable
- Provides `getPointAt(t)` and `getTangentAt(t)` for the flock sim

**Persistence:**
- Auto-save to `localStorage['birds-v1-path']` on every edit
- Export / Import JSON buttons
- Default path provided on first load so there's something to watch immediately

### 3. Flock simulation (`birds.js`)

**Bird state (per instance):**
- Position, velocity, per-bird phase offset (for wing flap + flock noise)
- No complex FSM — one flock, one leader `t` along the curve

**Flock leader:**
- Scalar `t ∈ [0,1]` advancing at `speed / curve.getLength()` per frame
- Each bird has an offset `(tOffset, radialOffset)` — its nominal target is `curve.getPointAt(t + tOffset)` plus a lateral offset along the curve's normal frame
- When the leader reaches `t=1`, the whole flock is flagged exiting; once all birds are past the final waypoint, respawn at `t=0`

**Boids forces per bird:**
- **Separation** (repel near neighbors) — spatial hash grid keyed on integer cell, cell size ≈ 2× desired neighbor radius
- **Alignment** (match neighbor velocity)
- **Cohesion** (steer toward local centroid)
- **Path attraction** — steer toward bird's nominal target on the curve
- **Speed damping** — clamp to `[vMin, vMax]`

Tunables (lil-gui "Flock"): count (default 300), separation/alignment/cohesion weights, path weight, flap rate, wander noise.

**Orientation:** each bird's instance matrix built from position + `lookAt(position + velocity)` so heading = velocity direction.

### 4. Geometry & rendering (`birds.js`)

**Bird mesh (procedural, single `BufferGeometry`):**
- Body: small elongated prism (~6–8 tris)
- Wings: two triangle pairs attached at the shoulder with a vertex attribute `wingSide ∈ {-1, 0, +1}` (center = body, -1/+1 = left/right wing tip)
- Total ~12–16 tris

**Custom `ShaderMaterial` / onBeforeCompile on MeshStandardMaterial:**
- `uniform float uTime`
- `attribute float aPhase` (per-instance, via `InstancedBufferAttribute`)
- Wing vertices rotated about the shoulder axis by `sin(uTime * flapRate + aPhase) * amplitude * wingSide`
- Ensures wings animate independently per bird while still using `MeshStandardMaterial` lighting

**Instancing:**
- Single `InstancedMesh`, `castShadow = true`, `receiveShadow = false`
- `customDepthMaterial` using the same wing-flap shader so shadow silhouettes match the animated wings

**Scale:** starlings at ~20cm wingspan in real units; in room-scale (meters) this is tiny but the flock reads well against 8m walls

### 5. Shadows

- Floor (`Floor` mesh from GLB) already has `receiveShadow = true` from the room load step
- The existing directional light's shadow frustum covers the full floor
- Birds' instanced mesh casts via `castShadow = true`; Three.js handles the shadow pass per-instance automatically
- `customDepthMaterial` required so the wing-flap vertex displacement shows up in shadows; without it, shadow silhouettes would show static wings

### 6. GUI (lil-gui)

Single top-right panel, folders:
- **Path** — waypoint list, add free-space button, clear, export/import, show curve toggle
- **Flock** — count, speed, boid weights, flap rate, wander
- **Scene** — camera preset (overview / ground-level), shadow intensity, show/hide room

## Data flow

```
[click/drag] → path-editor.js updates path[] → rebuilds CatmullRomCurve3
                                           ↓
[RAF tick] → birds.js:
               advance leader t
               for each bird:
                 spatial grid lookup → neighbors
                 compute forces (boids + curve attractor)
                 integrate velocity, position
                 build instance matrix
               upload InstancedMesh matrices
               uTime += dt
                                           ↓
[render] → WebGLRenderer → shadow pass (customDepthMaterial) → main pass
```

## Risks & open questions

- **Perf at 500+ birds with shadows:** spatial grid + InstancedMesh should handle it, but the shadow pass doubles vertex work. Mitigation: shadow LOD (skip shadow for distant birds) if we hit a wall.
- **Surface-stuck waypoints jitter on drag:** re-raycasting every drag tick could be noisy. Mitigation: only re-snap on drag *end*, not continuously.
- **Dive-through kind:** needs a defined behavior. Proposed: when the bird's nominal `t` is within ε of a dive-through waypoint, temporarily disable room-boundary clamping so the flock briefly crosses outside the room; after the bird passes, clamp resumes. First pass can treat dive-through as a regular waypoint and iterate.
- **Default path:** ship a reasonable 4-point default so a first-time visitor sees birds immediately.

## Implementation order

1. Scaffold folder, copy GLB, load room with shadow receivers, directional light
2. Add a placeholder single cube bird, get it following a hardcoded Catmull-Rom curve (proves integration + shadows)
3. Replace placeholder with procedural starling geometry + wing-flap shader
4. Turn single bird into `InstancedMesh` of N, each with phase offset, still path-following
5. Add boids forces, spatial grid, path attractor
6. Add path editor: click-to-place, TransformControls, curve rendering
7. lil-gui wiring for path + flock params
8. Persistence (localStorage + export/import)
9. Continuous loop (spawn/exit)
10. Polish: waypoint kinds (orbit, dive-through, exit behavior), default path, camera presets
