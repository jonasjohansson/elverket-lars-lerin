# pointcloud-v5 — soft-splat paintings with sandstorm transitions

## Goal

A single, stationary particle cloud that reads clearly as each of the 9 Serie_1 paintings. Moving forward or back with arrow keys triggers a slow sandstorm: the cloud's particles physically drift, scatter, desaturate, and re-address to form the next painting.

## Source material

9 trim PNGs already at `shared/images/series-1/17_RESAN_OCH_ORIENTEN_LL_{0012,0013,0018,0020,0022,0029,0031,0032,0038}-trim.png`. 3176×2064 RGBA, hand-trimmed alpha.

## Architecture

- New directory `pointcloud-v5/` — `index.html` + `main.js`, same importmap as v4.
- WebGPU + TSL node material.
- Flat scene (no depth maps). Stationary camera. Target at origin.
- Soft gaussian splat particle look.
- Single pool of ~800k particles, persistent for the session.

## Particle model — shared pool with 9 addresses

At load time, each of the ~800k particles is assigned 9 *addresses*, one per painting:

```
for each painting p in 0..8:
  particle i gets:
    pos_p = (x, y) sampled from p's opaque pixels, with sub-pixel jitter
    col_p = RGB at that pixel, optional light saturation nudge
```

Sample distribution per painting: with-replacement uniform sampling of opaque-alpha pixels. Jitter is `±0.5 * (worldW/imgW, worldH/imgH)`. Each painting is aspect-corrected against world height 2.0; world width follows the image aspect. Camera framing covers the widest painting.

**Decorrelation**: particle i's address in painting p has no spatial correlation with its address in painting p+1 — shuffle per painting. Without this the sandstorm feels too orderly (every grain travels short, parallel paths).

Per-particle extras (single vec4):
- `seedX, seedY` — unit random for noise phases
- `delay` — 0..1, staggers when this particle starts/ends its transition
- `sizeJitter` — 0.3..1.0, multiplies splat size

## Storage: data textures

9 lookups-per-particle is the only place this needs care.

- **Position texture**: `DataTexture(width=9, height=N, format=RG, type=HalfFloat)` — 9 xy pairs per particle.
- **Color texture**: `DataTexture(width=9, height=N, format=RGB, type=HalfFloat)`.

In the vertex node, fetch with `texelFetch(posTex, ivec2(uFromIdx, particleIndex))` and the same for `uToIdx`. Particle index comes from a 1D integer attribute or `instanceIndex` equivalent in TSL.

Memory: 800k × 9 × (2 pos + 3 color) floats at 16-bit ≈ 36 MB. Acceptable.

## Transition: sandstorm over ~12s

Three simultaneous envelopes driven by one `uT ∈ [0,1]`. Direction (from→to) set by `uFromIdx`, `uToIdx` uniforms.

### 1. Target lerp (position)

Each particle moves from its `pos_from` to `pos_to`:

```
posLerp = mix(pos_from, pos_to, smoothstep(0, 1, uT_eased))
```

Ease is a further `smoothstep` around uT so ends are calm, middle is fastest.

### 2. Gust scatter (displacement added on top)

Amplitude envelope: `gust = sin(π · uT)²` — zero at both ends, peaks mid-transition.

Composition:
- Directional bias: leftward wind proportional to gust, ~0.12 world units.
- Low-frequency curl noise: 2D sin/cos pairs over (pos.x, pos.y) with a time-drifting phase. Amplitude ~0.08.
- High-frequency chaos: per-particle seed-driven, amplitude ~0.04.
- Z component small, ±0.06, to give storm volume without wrecking flatness.

### 3. Color dwell (desaturation pulse)

Per-particle:
```
cT = smoothstep(0.4 + delay·0.3, 0.6 + delay·0.3, uT)
colorInterpolated = mix(col_from, col_to, cT)
dustMix = sin(π · uT)^3  // more peaked than gust
finalColor = mix(colorInterpolated, dustyNeutral, dustMix · 0.55)
```

`dustyNeutral` is a warm grey (`vec3(0.55, 0.5, 0.44)`) — not pure grey, tinted like airborne sand. Color drops out as gust peaks and re-emerges as wind dies. This is the perceptual kernel of "sandstorm hits".

### Per-particle stagger

`delay` seed (0..1) shifts each particle's effective uT by `±0.1`. Fronts and trailing grains — not a lockstep sweep.

## Ambient motion (always on)

Keep v4's two-wave ambient drift at gentle amplitudes (~0.015). Ensures the cloud breathes while holding a painting. Reuse the TSL from series-v1's `main.js`.

## Soft-splat look

`PointsNodeMaterial`, `sizeAttenuation: true`, `transparent: true`, `depthWrite: false`, `blending: NormalBlending`.

- `sizeNode`: `uSize * sizeJitter * (1 + small ambient pulse)`. Default `uSize ≈ 6`.
- `colorNode` returns `vec4(finalColor, opacity * alphaFalloff)` where `alphaFalloff` uses the built-in point sprite UV to produce a radial gaussian: `exp(-r² / 0.2)` with `r` = distance from point center.

Overlapping soft splats blend into continuous tone. Particle count can scale down to ~500k if fill-rate is an issue.

## Controls

- **→** — advance to painting `min(i+1, 8)`.
- **←** — advance to painting `max(i-1, 0)`.
- During an active transition, input is ignored. (No queueing — keeps timing predictable.)
- **F** — fullscreen.

No cycling/wrap. No auto-advance.

## GUI (lil-gui)

Only knobs worth tuning live:
- `transitionDuration` — 6..24 s, default 12
- `gustAmplitude` — 0..0.3, default 0.15
- `gustChaos` — 0..0.1, default 0.04
- `dustMixStrength` — 0..1, default 0.55
- `splatScale` — 2..12, default 6
- `opacity` — 0..1, default 0.9
- `ambientWaveAmp` — 0..0.05, default 0.015
- `bgColor` — default `#000`
- Action: "Restart from painting 0"

## Explicitly cut from v4

- Cursor wither / vortex / press
- Rot disintegration
- Spacebar dissolve
- DOF, afterimage trails, vignette
- Dust layer
- Transform-controlled lights, manual light transforms
- Auto-cycle mode
- Hue shift
- Auto-rotate

v5 is a focused piece. Add back if something is actually missing.

## Load & init

1. `await renderer.init()`
2. Load all 9 PNGs in parallel.
3. For each: `getImageData` → opaque pixel index list → sample N particles → store address at column p of position/color data textures.
4. Build one BufferGeometry with `aIndex` (particle row) + `aSeed` attributes.
5. Build single `Points` mesh with soft-splat material.
6. Install keyboard handler and render loop.

Progress text in `#info` during step 3 (it's the slow part — ~9 × getImageData on 3176×2064 images; downsample to max width 1600 like series-v1 does).

## Risks and mitigations

- **VRAM / bandwidth on data-texture lookups** — 9-wide texture is tiny in one dimension; caching is good. Low risk.
- **Fill rate with 800k soft splats at splatScale 6** — monitor. Can drop to 500k or tighten gaussian. Default `devicePixelRatio` clamped to 2.
- **Alpha sampling bias** — paintings with mostly-opaque PNGs get evenly covered; paintings with sparse alpha get denser clusters. That's intentional — trim silhouette drives the image.
- **TSL `texelFetch` availability** — three.js TSL exposes `texture(...)` with integer coordinates via `.uv(ivec2)` or `texelFetch`. If the former is flaky, fall back to normalized UV with `(uFromIdx + 0.5) / 9`.

## Out of scope

- Depth maps / pseudo-3D. Trim images are flat; adding synthetic depth hurts image clarity.
- Any mouse/cursor interaction.
- Auto-advance / cycling.
- Post-processing chain.

## Success criteria

1. 9 paintings load and each reads clearly as itself when held.
2. Advancing with → produces a visible, physical sandstorm — scatter, desaturate, reform — over ~12 s, not an alpha crossfade.
3. ← returns to the previous painting with the same quality transition.
4. Ambient motion is present but doesn't distract.
5. Holds 60 fps on a modern Mac at 1440p with default settings.
