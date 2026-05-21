# Elverket × Lars Lerin

Browser-based prototypes for image-to-image transitions, point clouds, and particle systems based on Lars Lerin watercolours. Built for the Elverket installation.

Open `index.html` in a browser through a local server to see the index of prototypes.

## Tools

- `transition-v2/` — A→B image transition, WebGPU
- `transition-v1/` — A→B image transition, WebGL2

## Other prototypes

- `birds-v1/` — particle flock with image-based waypoints
- `series-v1/`, `series-v2/` — image-series transitions
- `pointcloud-v1`–`v6`, `pointcloud/` — point-cloud experiments
- `approach-a-webgl/` through `approach-f-splat/` — earlier rendering experiments

## Images

The transition tools auto-load defaults from each tool's `defaults/` folder and let you upload your own through the UI. The older prototypes read from `shared/images/`.

## Local server

ES-module pages need a local HTTP server. From the repo root:

```bash
python3 -m http.server 8000
```

Then visit `http://localhost:8000/`.
