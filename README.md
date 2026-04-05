# Elverket × Lars Lerin — Particle Dissolution Prototypes

Three approaches to image → particle → dissolution for an immersive installation.

## Usage

Open any approach's `index.html` in a browser (via local server). Press F for fullscreen.

- `approach-a-webgl/` — Raw WebGL + GLSL shaders
- `approach-b-gsplat/` — Gaussian splatting with instanced quads
- `approach-c-hybrid/` — Splat-aesthetic sprites on Three.js

## Images

Drop still images into `shared/images/`. Supported: jpg, png.
If no images found, a generated gradient placeholder is used.

## Local Server

Each approach needs a local server for ES modules:

```bash
# Approach A
cd approach-a-webgl && python3 -m http.server 8001

# Approach B
cd approach-b-gsplat && python3 -m http.server 8002

# Approach C
cd approach-c-hybrid && python3 -m http.server 8003
```
