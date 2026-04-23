# pointcloud-v6 — two-painting disintegration transition

## Goal

A single particle cloud that reads as painting A, then patchily rots away as painting B emerges in the same holes. Auto-loops ping-pong between the two paintings. No cursor interaction, no slideshow — one focused effect: disintegration *is* the transition.

## Source material

Two Serie_1 trim PNGs:

- `shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_0012-trim.png` — sparse desert, figures, warm sky-heavy composition.
- `shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_0029-trim.png` — dense stone ruins, architectural mass.

Chosen for contrast: sparse-to-dense, sky-to-stone, similar earthy palette so the crossfade reads coherent. Flat alpha PNGs — no depth maps.

## Architecture

- New directory `pointcloud-v6/` — `index.html` + `main.js`, same importmap as v4/v5.
- WebGPU + TSL node material.
- Stationary camera. OrbitControls enabled (damped), target at origin.
- Soft-splat points, persistent pool.
- ~800k particles. Tunable.

## Particle model — dual addresses (A and B)

At load time, every particle is sampled twice — once against A's opaque pixels, once against B's opaque pixels. Each particle carries:

- `posA = (x, y)` sampled from A's alpha-masked pixels, sub-pixel jitter.
- `colA = RGB` at that pixel.
- `posB = (x, y)` sampled independently from B.
- `colB = RGB` at that B pixel.
- `z` — single per-particle random in `[-0.2, +0.2]`. Same Z for both addresses (particles stay at a fixed depth through the transition — the cloud has thickness, not per-image thickness).
- `seed` — vec4: `seedX, seedY, delay, sizeJitter`.

**Independent A and B sampling** (no spatial correlation between a particle's A and B position). This is deliberate: disintegration is clustered in world space, not tied to "particle travels from its A spot to its B spot". A particle that dies in A's patch doesn't necessarily reform in the matching B patch — B's new patches occupy their own regions.

World framing: height = 2.0, width = 2.0 × aspect per painting. Since both paintings are from the same series they should have matching aspect; we'll use max(aspectA, aspectB) for the world width so neither painting clips.

Memory: 800k × (4 floats pos + 3 floats col) × 2 paintings ≈ 22 MB of geometry attributes. Cheap — no atlas textures needed. Store as raw attributes.

## The transition mechanism — disintegration drives crossfade

One master phase `uPhase ∈ [0, 1]` drives the whole cycle. A per-particle **death moment** `dM ∈ [0, 1]` determines when that particle crosses from rendering A to rendering B.

`dM` is clustered via a patch noise field so neighbors die together (v4's rot-like look):

```
patch(pos) = patchA * 0.55 + patchB * 0.35   // scaled to [0, 1]
dM = patch(pos) + jitter(delay) * 0.15        // small per-particle jitter
```

Per-particle transition progress:

```
dieProg = smoothstep(dM - fade, dM + fade, uPhase)   // fade ≈ 0.08
```

### Position

```
posFlat = mix(posA, posB, dieProg)       // instant teleport, clustered
// during the decay window, add dispersion + chaos
crumble = smoothstep(0, 1, abs(dieProg - 0.5) * 2)  // peaks at dM, 0 at ends
dispersion = crumble * (dirSeed * dispersalAmp - vec3(0, gravity, 0))
chaos = crumble * sin(t * phase + seeds) * chaosAmp
finalPos = posFlat + vec3(0, 0, z) + dispersion + chaos + ambientWave(pos, t)
```

Particles crumble outward/downward as they die, then snap to their B address on the far side. The `crumble` envelope is zero at both ends, peaks mid-death, so A-particles and B-particles settle cleanly into their painting shape.

### Color

```
finalColor = mix(colA, colB, dieProg) * decayDarken(crumble)
alpha = baseAlpha * decayFade(crumble)
```

Particles darken and fade during the crumble peak, then recover as the new painting's color. This is what sells "rotting" vs a pure alpha blend.

### Size

Particle shrinks at `crumble` peak, recovers on the B side — same envelope.

## Lifecycle — ping-pong auto-loop

`uPhase` is driven by a state machine ticking against real time:

| State | Duration (s) | `uPhase` behavior |
|-------|--------------|-------------------|
| `holdA` | 4 | fixed at 0 |
| `AtoB` | 6 | ramps 0 → 1 |
| `holdB` | 4 | fixed at 1 |
| `BtoA` | 6 | ramps 1 → 0 |

`~20s` round-trip. On entering `BtoA`, the logic is the same as `AtoB` but ramps backward — because the per-particle death is `smoothstep(dM - fade, dM + fade, uPhase)`, running `uPhase` in reverse naturally plays the disintegration backward. No separate code path.

State machine and ramp respect an `autoLoop` toggle:

- `autoLoop = true` (default) — runs forever.
- `autoLoop = false` — holds the current phase. User can drag a slider, click a button, or press keys to advance.

## Ambient motion

Always on, even during holds. Reuse v4's two-wave drift at gentle amplitudes (~0.015). Keeps the cloud breathing.

## Rendering

- `PointsNodeMaterial`, `sizeAttenuation: true`, `transparent: true`, `depthWrite: false`, `blending: NormalBlending`.
- `sizeNode`: `uSize * sizeJitter * (1 + small pulse) * crumbleShrink`. Default `uSize ≈ 6`.
- `colorNode`: returns `vec4(finalColor, opacity * alpha)` with a radial gaussian falloff from the point UV (soft splat).
- Background `#000`.
- No post-processing (no DOF, no trails, no vignette) for v6. Add only if the effect needs it.

## Camera / interaction

- OrbitControls enabled, damped. Default framing: `camera.position = (0, 0, 3.2)`, `target = (0, 0, 0)`.
- **No cursor field.** No click, no vortex, no wither. Mouse drives orbit only.
- **F** toggles fullscreen.
- **Space** toggles `autoLoop`. While paused, **←/→** jumps to `holdA` / `holdB`.

## GUI (lil-gui)

Tunable live:

| Knob | Range | Default |
|------|-------|---------|
| `holdDuration` | 0–20 s | 4 |
| `transitionDuration` | 1–20 s | 6 |
| `autoLoop` | bool | true |
| `phase` (slider, live) | 0–1 | driven by loop |
| `patchScale` | 0.5–12 | 3.0 |
| `dispersalAmp` | 0–2 | 0.35 |
| `gravity` | 0–1 | 0.25 |
| `chaosAmp` | 0–0.6 | 0.18 |
| `shrink` | 0–1 | 0.85 |
| `darken` | 0–1 | 0.7 |
| `fadeWindow` | 0.02–0.3 | 0.08 |
| `splatScale` | 2–12 | 6 |
| `opacity` | 0–1 | 0.9 |
| `ambientWaveAmp` | 0–0.05 | 0.015 |
| `zScatter` | 0–0.6 | 0.2 |
| Action | — | "Jump to A / Jump to B" |

## Explicitly cut from v4

- All cursor behavior (wither, click expand, right-click vortex).
- Depth maps / 3D painting projection (trim images have none; we use random Z scatter for cloud thickness instead).
- Spacebar dissolve (repurposed: spacebar toggles auto-loop).
- DOF, afterimage, vignette.
- Dust layer, lights with transform controls.
- Hue shift, auto-rotate.
- v4's separate transition mechanism — in v6, disintegration *is* the transition, not a parallel effect.

## Load & init

1. `await renderer.init()`.
2. Load both PNGs in parallel.
3. For each: `getImageData`, build opaque-pixel index list (alpha > threshold), sample N particles, store `(x, y, r, g, b)`.
4. Build one `BufferGeometry` with attributes: `position` (placeholder zeroes), `aPosA`, `aColA`, `aPosB`, `aColB`, `aSeed`, `aZ`.
5. Build `Points` mesh with the TSL material described above.
6. Start the phase state machine and animation loop.

Progress text in `#info` during the sampling phase.

## Risks and mitigations

- **Clustered death looking artificial with only 2 paintings**: the patch field is what makes v4 look organic. Keep `patchScale` tunable; default ~3.0 gives medium patches against a 2-unit-wide world.
- **Ambient motion fighting the transition**: keep ambient amplitude small (< 0.02). v4 handled this fine.
- **B particles being static while A is holding** (and vice versa): at `uPhase = 0`, all particles render at `posA, colA` — no B-ghost. That's correct; B doesn't exist until `dieProg > 0` for each particle. Verified by `mix(posA, posB, 0) = posA`.
- **Hard transitions at death window edges**: `fadeWindow` controls the smoothstep slope. Too narrow = visible popping; too wide = mushy crossfade. Start 0.08.
- **Fill rate at 800k + splatScale 6**: same as v5 concern. Drop particle count if needed.

## Out of scope

- Third painting, slideshow, or picker UI.
- Any cursor interaction.
- Post-processing chain.
- Real 3D / depth from painting content.

## Success criteria

1. Painting A reads clearly when held. Painting B reads clearly when held.
2. The transition is visibly a *disintegration* — patchy, clustered death — not an alpha crossfade. Dark-to-light and light-to-dark regions both convert believably.
3. Auto-loop runs without stutter; round-trip ≈ 20 s at defaults.
4. Spacebar pauses the loop; slider lets me scrub `phase` manually.
5. Holds 60 fps on a modern Mac at 1440p with default settings.
