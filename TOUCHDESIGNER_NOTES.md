# Lars Lerin x Refik Anadol — TouchDesigner Particle System

## Overview
Particle visualization of Lars Lerin paintings in TouchDesigner, inspired by Refik Anadol's sculptural data installations. Controlled via TouchDesigner MCP (Model Context Protocol).

## Architecture
- **1.64M particles** (1280x1280 GLSL TOP — max for non-commercial TD license)
- **GPU compute sim** — particle positions stored in 32-bit float texture, updated per frame via feedback loop
- **GLSL MAT instancing** — vertex shader reads position texture via `TDInstanceID()`, assigns painting color from home grid position
- **Sphere SOP** as instance shape (6 pts mesh, lightweight)

## Node Network
| Node | Type | Purpose |
|------|------|---------|
| `lerin_color` | moviefileinTOP | Source painting |
| `lerin_depth` | moviefileinTOP | Depth map |
| `particle_sim` | glslTOP 1280x1280 rgba32float | GPU particle simulation |
| `particle_fb` | feedbackTOP | Maintains state between frames |
| `particles` | geometryCOMP | Instanced geometry (1,638,400 instances) |
| `particle_mat` | glslMAT | Custom vertex + pixel shaders |
| `render1` | renderTOP | 1280x720 render |
| `levels1` | levelTOP | Contrast/gamma post-processing |
| `bloom1` | bloomTOP | Glow |
| `OUT` | nullTOP | Final output |
| `white_box` | geometryCOMP | Gallery frame (5 walls, currently hidden) |
| `light1-4` | lightCOMP | Point lights for scene |
| `cam1` | cameraCOMP | Straight-on view |

## Shader Files (textDATs)
- `particle_sim_pixel` — v10 simulation: anchored XY + dramatic Z sculpture
- `particle_mat_vertex` — instancing, painting color sampling from home UV
- `particle_mat_pixel` — 4-point position-based lighting, depth AO

## Key Design Decisions

### Sim approach (v1→v10 evolution)
1. **v1-v5**: Fragment shader displacement → looked like cloth, not particles
2. **v6**: Free-flowing 3D curl noise → spiral vortex problem (center-pull gravity)
3. **v7-v8**: Anchored to home position → painting visible but too static
4. **v9**: Free-flowing dramatic → lost the painting entirely
5. **v10 (current)**: Anchored XY (painting readable) + dramatic Z (ridged noise terrain). Edge mask pins borders, movement happens inside the frame.

### Lighting
- Per-sphere normals caused repeated shadow pattern on each particle
- Switched to **position-based lighting** — brightness from distance to light positions, not surface normals
- 4 lights: warm key (upper right), warm side (left), cool under, back rim

### Particle color
- Sampled from painting at **home grid UV** (posUV), not current position
- Color is permanently imprinted — particles carry their birth color as they move

### Edge mask
- `edgeMask = edgeX * edgeY` fades all displacement to zero near borders
- Creates Anadol's "frame" effect — still edges, alive interior

## Custom Parameters (on `particles` COMP → "Anadol FX" page)
| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| Flow Speed | 0.5 | 0.01–1.0 | Overall animation speed |
| Z Depth | 2.0 | 0.0–8.0 | Height of sculptural peaks |
| Curl Strength | 0.01 | 0.001–0.05 | XY displacement amount |
| Particle Size | 0.004 | 0.001–0.05 | Individual particle scale |

These drive the sim and mat uniforms via expressions.

## Known Issues / Warnings
- **Cook dependency loop** warning is expected (feedback loop) — not an error
- Non-commercial license caps TOP resolution at 1280x1280 and render at 1280x720
- CPU cook time ~14ms for the sim — could optimize noise functions if needed

## Future Ideas
- **Latent space walks**: Train VAE/StyleGAN on Lerin paintings, feed latent vectors via OSC/CHOP to drive particle morphology (true Anadol approach)
- **White box frame**: Built but hidden (`white_box` geo) — enable and add to render for gallery presentation
- **Camera animation**: Slow orbit or subtle drift for installation mode
- **Multiple paintings**: Crossfade between source images for transitions

## Source Images
- Painting: `shared/images/lerin2.jpg`
- Depth map: `shared/images/lerin2_depth.png`
