# Series v1 — Painting ribbon with sandstorm transitions

**Date:** 2026-04-24
**Target:** `elverket-lars-lerin/series-v1/` (standalone prototype)

## Goal

Present the nine trimmed PNGs from `17_RESAN_OCH_ORIENTEN/Image/Serie_1/web` as a horizontal ribbon of particle-cloud paintings. Arrow-key snap navigates between them; the snap transition is a directional sandstorm gust that carries particles from the outgoing painting's downwind edge into the incoming painting, while each painting's core stays intact and recognizable.

## Scope & non-goals

**In scope**
- Per-painting particle clouds built from `-trim.png` alpha + RGB
- Ribbon layout along X, camera X-snap between paintings
- Ambient wave drift at rest (from pointcloud-v2, low amplitude)
- Edge-bleed sandstorm transition driven by precomputed per-particle edge-distance
- Directional wind that flips with arrow direction
- GUI knobs for spacing, transition duration, gust, edge falloff, waves
- Index.html entry under a new "Painting series" section

**Out of scope (v1)**
- Depth maps / depth-driven Z displacement (paintings are flat with small Z jitter)
- Gaussian splats
- Dissolve/wither/cursor interactions from pointcloud-v3/v4
- Audio
- Loading indicators beyond the existing `#info` pattern
- Mobile touch navigation (desktop/arrow-key only for v1)

## Approach chosen

**One `Points` mesh per painting, shared material pattern** (approach B in brainstorm). Each painting is its own ~250k-particle cloud, positioned on the ribbon by an X offset. Transitions are driven by per-mesh uniforms that are 0 for the 7 inactive paintings and 1 for the outgoing/incoming pair during a snap. Keeps per-painting identity strong, makes the edge-bleed bookkeeping local, and scales naturally if more paintings get added later.

## Architecture

```
series-v1/
  index.html         # canvas + import map, same pattern as pointcloud-v2
  main.js            # scene, ribbon builder, per-painting clouds, snap state
                     # shared TSL material factory, shared uniforms for wind

shared/images/series-1/
  17_RESAN_OCH_ORIENTEN_LL_0012-trim.png
  ... (9 files copied from the Google Drive `web/` folder)
```

### Ribbon layout

- Painting height normalized to 2.0 world units; width derived from PNG aspect ratio
- `stride = paintingWidth + gap`, where `gap ≈ 0.6 × avgPaintingWidth`
- Painting `i` centered at `x = i * stride` (painting 0 at origin)
- Camera at `(currentX, 0, 3.8)`, target `(currentX, 0, -0.3)`, FOV 45
- OrbitControls disabled by default; space toggles a free-look mode

### Per-painting particle data

For each painting, sample the trimmed PNG into a particle buffer:

1. Read pixels; keep only pixels with `alpha > 0.5`
2. Randomly pick ~250k such pixels (with replacement is fine if painting has fewer unique pixels — most will have plenty)
3. Precompute `aEdgeDist` per particle: distance to nearest transparent pixel, in pixels, normalized by a target edge-band width (e.g. 40px), clamped 0..1. Runs once at load via a simple multi-pass distance transform on the alpha mask.
4. Store per particle: `position` (home XY in world, Z with ±0.08 jitter), `aColor` (RGB sampled), `aSeed` (4 random floats for noise offsets), `aEdgeDist` (1 float)

Total: 9 × 250k = 2.25M particles.

### Shared TSL material

All 9 meshes share one `PointsNodeMaterial` built by a factory function. The material reads per-painting uniforms passed at construction time (`uPaintingGust`) alongside global uniforms (`uTime`, `uWindDir`, `uWindGust`, ambient wave params).

**Position node** combines:
- Home position (attribute)
- Ambient wave drift (v2 wave stack at low amplitude, always on)
- Micro per-particle drift
- Edge-bleed displacement: `detach * uWindDir * stride * travelCurve(uWindGust)` on X, with cross-wind Y and swirl Z taken from v2's transition scatter pattern, all multiplied by `uPaintingGust` so idle paintings don't move
- `detach = smoothstep(edgeFalloff, 0.0, aEdgeDist) * uWindGust`

**Color/alpha node** is the sampled color (optionally lit the way v2 is) with alpha dipping by `1 - 0.3 * detach * uWindGust` so airborne sand reads thinner than the anchored core.

### Transition state machine

```
State: idle | transitioning
currentIndex: int
targetIndex: int   (== currentIndex when idle)
progress: 0..1     (ease-in-out)
duration: 2.5s (tunable)
```

On arrow press (ignored if transitioning or at boundary):
1. Set `targetIndex`, `progress = 0`, `uWindDir.value = sign(targetIndex - currentIndex)`
2. Mark `uPaintingGust[currentIndex] = 1`, `uPaintingGust[targetIndex] = 1`, all others 0
3. Each frame: advance `progress`, update `uWindGust.value = sin(progress * PI)` (peaks at 0.5, zero at 0 and 1), lerp camera X from `currentX` toward `targetX`
4. When `progress >= 1`: clear all `uPaintingGust`, set `currentIndex = targetIndex`

Only two paintings ever have non-zero gust at once, so the other seven keep running only the cheap ambient-wave path.

### Wind direction semantics

- Right arrow: `uWindDir = +1`, outgoing painting's right edge streams rightward, incoming painting's left edge receives
- Left arrow: `uWindDir = -1`, mirror
- Because `detach` depends on `aEdgeDist` (distance-to-transparent), both edges are equally eligible; the wind direction alone decides which side streams

### Controls

- `→` / `←` — snap next / previous
- `Space` — toggle free-look OrbitControls
- `F` — fullscreen
- lil-gui panel (top-right): spacing, particle count, transition duration, wind gust amplitude, edge falloff band, wave amplitudes, ambient drift, vignette/dust/trails

## Performance budget

- 2.25M particles, 9 draw calls (trivial on WebGPU)
- Inactive paintings skip the edge-bleed branch via `uPaintingGust === 0` short-circuit (branch predictable per mesh since it's a uniform, not an attribute)
- Target 60fps on modern MacBook M-series; no WebGL fallback (repo is WebGPU-only)

## Assets

The 9 `-trim.png` files live in a Google Drive folder the browser cannot read directly. One-time copy to `shared/images/series-1/` at the start of implementation. Filename casing preserved.

## Open questions / deferred

- Whether to add a subtle global dust layer (like v2's) on top — deferred, easy to add later
- Whether the transition-duration should scale with travel distance if someone jumps multiple paintings at once — v1 only supports single-step snaps, so not yet relevant
- Loading strategy: v1 loads all 9 paintings up front (it's ~90MB raw and processes into ~2.25M particles; acceptable for a gallery piece). Lazy loading is a v2 concern.
