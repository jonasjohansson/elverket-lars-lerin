// ─── Gaussian Splatting Particle Dissolution ───
// Approach B: Instanced 2D Gaussian ellipsoids for painterly watercolor dissolution

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { alpha: false, premultipliedAlpha: true });
if (!gl) { document.body.textContent = 'WebGL2 required'; throw new Error('No WebGL2'); }

// ─── Constants ───
const NUM_SPLATS = 60000;
const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 4.0;
const BLACK_TIME = 2.0;

// ─── State ───
let state = 'LOADING'; // LOADING -> HOLDING -> DISSOLVING -> BLACK -> HOLDING ...
let stateTime = 0;
let currentImageIndex = 0;
let images = [];
let splatData = null; // Float32Array for instance attributes
let lastTime = 0;

// ─── Shaders ───
const VERT_SRC = /*glsl*/`#version 300 es
precision highp float;

// Quad vertices
in vec2 a_quadPos;

// Per-instance
in vec2 a_position;
in vec3 a_color;
in vec2 a_scale;
in float a_rotation;
in float a_opacity;
in float a_dissolveDelay;
in vec3 a_noiseOffset;

uniform vec2 u_resolution;
uniform float u_dissolve; // 0..1
uniform float u_time;

out vec2 v_uv;
out vec4 v_color;

// ─── Simplex noise 3D (Ashima Arts) ───
vec3 mod289(vec3 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 mod289(vec4 x) { return x - floor(x * (1.0 / 289.0)) * 289.0; }
vec4 permute(vec4 x) { return mod289(((x * 34.0) + 10.0) * x); }
vec4 taylorInvSqrt(vec4 r) { return 1.79284291400159 - 0.85373472095314 * r; }

float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);
    vec3 i = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);
    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;
    i = mod289(i);
    vec4 p = permute(permute(permute(
        i.z + vec4(0.0, i1.z, i2.z, 1.0))
      + i.y + vec4(0.0, i1.y, i2.y, 1.0))
      + i.x + vec4(0.0, i1.x, i2.x, 1.0));
    float n_ = 0.142857142857;
    vec3 ns = n_ * D.wyz - D.xzx;
    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);
    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);
    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);
    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);
    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));
    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;
    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);
    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0),dot(p1,p1),dot(p2,p2),dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;
    vec4 m = max(0.6 - vec4(dot(x0,x0),dot(x1,x1),dot(x2,x2),dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m*m, vec4(dot(p0,x0),dot(p1,x1),dot(p2,x2),dot(p3,x3)));
}

// Curl noise from simplex for swirling motion
vec2 curlNoise(vec3 p) {
    float eps = 0.01;
    float n1 = snoise(p + vec3(eps, 0.0, 0.0));
    float n2 = snoise(p - vec3(eps, 0.0, 0.0));
    float n3 = snoise(p + vec3(0.0, eps, 0.0));
    float n4 = snoise(p - vec3(0.0, eps, 0.0));
    float dx = (n3 - n4) / (2.0 * eps);
    float dy = -(n1 - n2) / (2.0 * eps);
    return vec2(dx, dy);
}

void main() {
    // Per-splat dissolve progress with stagger
    float d = max(0.0, u_dissolve - a_dissolveDelay) / (1.0 - a_dissolveDelay + 0.001);
    d = clamp(d, 0.0, 1.0);

    // Animated position
    vec2 pos = a_position;
    vec2 scale = a_scale;

    if (d > 0.0) {
        // Curl noise drift
        vec3 noisePos = a_noiseOffset + vec3(u_time * 0.3, u_time * 0.2, u_time * 0.1);
        vec2 curl = curlNoise(noisePos) * 120.0 * d;
        pos += curl;

        // Gravity drift
        pos.y += d * d * 60.0;

        // Scale grows (watercolor bleed)
        scale *= 1.0 + d * 2.5;
    }

    // Build rotation + scale matrix
    float c = cos(a_rotation);
    float s = sin(a_rotation);
    mat2 rot = mat2(c, s, -s, c);
    vec2 localPos = rot * (a_quadPos * scale);

    // To clip space
    vec2 screenPos = pos + localPos;
    vec2 ndc = (screenPos / u_resolution) * 2.0 - 1.0;
    ndc.y = -ndc.y; // flip Y
    gl_Position = vec4(ndc, 0.0, 1.0);

    v_uv = a_quadPos; // -1..1

    // Opacity with non-linear fade
    float opacity = a_opacity * (1.0 - pow(d, 1.5));
    v_color = vec4(a_color * opacity, opacity);
}
`;

const FRAG_SRC = /*glsl*/`#version 300 es
precision highp float;

in vec2 v_uv;
in vec4 v_color;

out vec4 fragColor;

void main() {
    // 2D Gaussian kernel
    float r2 = dot(v_uv, v_uv);
    float gauss = exp(-r2 * 2.0);

    // Premultiplied alpha
    float alpha = gauss * v_color.a;
    if (alpha < 0.003) discard;

    fragColor = vec4(v_color.rgb * gauss, alpha);
}
`;

// ─── Compile shaders ───
function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error(gl.getShaderInfoLog(s));
        throw new Error('Shader compile error');
    }
    return s;
}

function createProgram(vs, fs) {
    const p = gl.createProgram();
    gl.attachShader(p, vs);
    gl.attachShader(p, fs);
    gl.linkProgram(p);
    if (!gl.getProgramParameter(p, gl.LINK_STATUS)) {
        console.error(gl.getProgramInfoLog(p));
        throw new Error('Program link error');
    }
    return p;
}

const vs = createShader(gl.VERTEX_SHADER, VERT_SRC);
const fs = createShader(gl.FRAGMENT_SHADER, FRAG_SRC);
const program = createProgram(vs, fs);

// ─── Attribute locations ───
const loc = {
    quadPos: gl.getAttribLocation(program, 'a_quadPos'),
    position: gl.getAttribLocation(program, 'a_position'),
    color: gl.getAttribLocation(program, 'a_color'),
    scale: gl.getAttribLocation(program, 'a_scale'),
    rotation: gl.getAttribLocation(program, 'a_rotation'),
    opacity: gl.getAttribLocation(program, 'a_opacity'),
    dissolveDelay: gl.getAttribLocation(program, 'a_dissolveDelay'),
    noiseOffset: gl.getAttribLocation(program, 'a_noiseOffset'),
};

const uni = {
    resolution: gl.getUniformLocation(program, 'u_resolution'),
    dissolve: gl.getUniformLocation(program, 'u_dissolve'),
    time: gl.getUniformLocation(program, 'u_time'),
};

// ─── VAO setup ───
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

// Quad geometry: 2 triangles, vertices in [-1,1]
const quadVerts = new Float32Array([
    -1, -1,  1, -1,  1, 1,
    -1, -1,  1,  1, -1, 1,
]);
const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, quadVerts, gl.STATIC_DRAW);
gl.enableVertexAttribArray(loc.quadPos);
gl.vertexAttribPointer(loc.quadPos, 2, gl.FLOAT, false, 0, 0);

// Instance buffer — interleaved
// Layout per instance (floats):
// position(2) + color(3) + scale(2) + rotation(1) + opacity(1) + dissolveDelay(1) + noiseOffset(3) = 13 floats
const FLOATS_PER_INSTANCE = 13;
const STRIDE = FLOATS_PER_INSTANCE * 4;

const instanceBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
gl.bufferData(gl.ARRAY_BUFFER, FLOATS_PER_INSTANCE * 4 * NUM_SPLATS, gl.DYNAMIC_DRAW);

function setupInstanceAttrib(location, size, offset) {
    if (location < 0) return;
    gl.enableVertexAttribArray(location);
    gl.vertexAttribPointer(location, size, gl.FLOAT, false, STRIDE, offset * 4);
    gl.vertexAttribDivisor(location, 1);
}

// offset in floats: pos(0), color(2), scale(5), rotation(7), opacity(8), dissolveDelay(9), noiseOffset(10)
setupInstanceAttrib(loc.position, 2, 0);
setupInstanceAttrib(loc.color, 3, 2);
setupInstanceAttrib(loc.scale, 2, 5);
setupInstanceAttrib(loc.rotation, 1, 7);
setupInstanceAttrib(loc.opacity, 1, 8);
setupInstanceAttrib(loc.dissolveDelay, 1, 9);
setupInstanceAttrib(loc.noiseOffset, 3, 10);

gl.bindVertexArray(null);

// ─── Image loading ───

// Generate fallback watercolor-style images on canvas
function generateFallbackImage(palette, index) {
    const w = 1024, h = 768;
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');

    // Dark background wash
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, w, h);

    // Overlapping radial gradients — watercolor blobs
    const blobs = 8 + Math.floor(Math.random() * 6);
    for (let i = 0; i < blobs; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const r = 100 + Math.random() * 300;
        const color = palette.colors[Math.floor(Math.random() * palette.colors.length)];
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        grad.addColorStop(0, color + 'cc');
        grad.addColorStop(0.4, color + '88');
        grad.addColorStop(0.7, color + '33');
        grad.addColorStop(1, color + '00');
        ctx.globalCompositeOperation = 'lighter';
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
    }

    // Soft overlay pass
    ctx.globalCompositeOperation = 'multiply';
    for (let i = 0; i < 3; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const r = 200 + Math.random() * 400;
        const color = palette.overlay;
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        grad.addColorStop(0, color + '66');
        grad.addColorStop(1, color + '00');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
    }

    ctx.globalCompositeOperation = 'source-over';
    return ctx.getImageData(0, 0, w, h);
}

const palettes = [
    { // Arctic blues (Lerin's Svalbard palette)
        bg: '#0a1520',
        colors: ['#4a7fb5', '#6ba3d6', '#2c5a8a', '#89c4e8', '#1a3a5c', '#c8dff0'],
        overlay: '#1a3050'
    },
    { // Earth tones / autumn
        bg: '#1a1208',
        colors: ['#8b6914', '#c49a3c', '#5c4a1e', '#d4a84b', '#3a2e10', '#e8c882'],
        overlay: '#2a1e08'
    },
    { // Muted greens / moss and water
        bg: '#0a1a10',
        colors: ['#3a7a4a', '#5aaa6a', '#2a5a38', '#7aca8a', '#1a3a22', '#a8dab0'],
        overlay: '#0a2a15'
    },
];

async function loadImages() {
    // Try loading from shared/images
    const extensions = ['jpg', 'jpeg', 'png'];
    const names = ['01', '02', '03', '1', '2', '3', 'lerin-1', 'lerin-2', 'lerin-3', 'painting-1', 'painting-2'];
    const loaded = [];

    for (const name of names) {
        for (const ext of extensions) {
            if (loaded.length >= 5) break;
            try {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                const url = `../shared/images/${name}.${ext}`;
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                    img.src = url;
                });
                // Extract pixel data
                const c = document.createElement('canvas');
                c.width = img.width; c.height = img.height;
                const ctx = c.getContext('2d');
                ctx.drawImage(img, 0, 0);
                loaded.push(ctx.getImageData(0, 0, img.width, img.height));
            } catch (e) { /* skip */ }
        }
    }

    if (loaded.length > 0) return loaded;

    // Fallback: generate watercolor images
    console.log('No images found — generating fallback watercolors');
    return palettes.map((p, i) => generateFallbackImage(p, i));
}

// ─── Sample image into splat data ───
function sampleImageToSplats(imageData) {
    const { data, width, height } = imageData;
    const arr = new Float32Array(NUM_SPLATS * FLOATS_PER_INSTANCE);

    // Compute brightness-weighted sampling probability
    const numPixels = width * height;
    const brightness = new Float32Array(numPixels);
    let totalBrightness = 0;
    for (let i = 0; i < numPixels; i++) {
        const off = i * 4;
        const b = (data[off] * 0.299 + data[off + 1] * 0.587 + data[off + 2] * 0.114) / 255;
        // Weight brighter pixels more, but keep some darks
        brightness[i] = 0.05 + b * 0.95;
        totalBrightness += brightness[i];
    }

    // Build CDF for weighted sampling
    const cdf = new Float32Array(numPixels);
    cdf[0] = brightness[0] / totalBrightness;
    for (let i = 1; i < numPixels; i++) {
        cdf[i] = cdf[i - 1] + brightness[i] / totalBrightness;
    }

    // Scale factor to fit image to canvas
    const cw = canvas.width;
    const ch = canvas.height;
    const imgAspect = width / height;
    const canvasAspect = cw / ch;
    let scaleX, scaleY, offsetX, offsetY;
    if (imgAspect > canvasAspect) {
        scaleX = cw / width;
        scaleY = scaleX;
        offsetX = 0;
        offsetY = (ch - height * scaleY) / 2;
    } else {
        scaleY = ch / height;
        scaleX = scaleY;
        offsetX = (cw - width * scaleX) / 2;
        offsetY = 0;
    }

    for (let s = 0; s < NUM_SPLATS; s++) {
        // Weighted random pixel selection via binary search on CDF
        const r = Math.random();
        let lo = 0, hi = numPixels - 1;
        while (lo < hi) {
            const mid = (lo + hi) >> 1;
            if (cdf[mid] < r) lo = mid + 1; else hi = mid;
        }
        const pixIdx = lo;
        const px = pixIdx % width;
        const py = Math.floor(pixIdx / width);
        const off = pixIdx * 4;

        // Color with slight warmth/saturation jitter
        let cr = data[off] / 255;
        let cg = data[off + 1] / 255;
        let cb = data[off + 2] / 255;

        // Warmth jitter
        const warmth = (Math.random() - 0.5) * 0.08;
        cr = Math.min(1, Math.max(0, cr + warmth));
        cg = Math.min(1, Math.max(0, cg + warmth * 0.5));
        cb = Math.min(1, Math.max(0, cb - warmth * 0.3));

        // Saturation jitter
        const lum = cr * 0.299 + cg * 0.587 + cb * 0.114;
        const satJitter = 1.0 + (Math.random() - 0.5) * 0.3;
        cr = lum + (cr - lum) * satJitter;
        cg = lum + (cg - lum) * satJitter;
        cb = lum + (cb - lum) * satJitter;
        cr = Math.min(1, Math.max(0, cr));
        cg = Math.min(1, Math.max(0, cg));
        cb = Math.min(1, Math.max(0, cb));

        // Anisotropic scale for brushstroke feel
        const sx = 3 + Math.random() * 12;  // 3-15px
        const sy = 1.5 + Math.random() * 6.5; // 1.5-8px

        // Rotation with bias toward horizontal/vertical
        let angle;
        const biasRoll = Math.random();
        if (biasRoll < 0.3) {
            angle = (Math.random() - 0.5) * 0.4; // near horizontal
        } else if (biasRoll < 0.5) {
            angle = Math.PI / 2 + (Math.random() - 0.5) * 0.4; // near vertical
        } else {
            angle = Math.random() * Math.PI; // random
        }

        const base = s * FLOATS_PER_INSTANCE;
        arr[base + 0] = px * scaleX + offsetX;  // position x
        arr[base + 1] = py * scaleY + offsetY;  // position y
        arr[base + 2] = cr;   // color r
        arr[base + 3] = cg;   // color g
        arr[base + 4] = cb;   // color b
        arr[base + 5] = sx;   // scale x
        arr[base + 6] = sy;   // scale y
        arr[base + 7] = angle; // rotation
        arr[base + 8] = 0.7 + Math.random() * 0.3; // opacity (0.7-1.0)
        arr[base + 9] = Math.random() * 0.3;        // dissolve delay (0-0.3)
        arr[base + 10] = Math.random() * 100;       // noise offset x
        arr[base + 11] = Math.random() * 100;       // noise offset y
        arr[base + 12] = Math.random() * 100;       // noise offset z
    }

    return arr;
}

// ─── Resize ───
function resize() {
    const dpr = window.devicePixelRatio || 1;
    const w = window.innerWidth;
    const h = window.innerHeight;
    if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
        canvas.width = w * dpr;
        canvas.height = h * dpr;
        gl.viewport(0, 0, canvas.width, canvas.height);
    }
}

// ─── Transition to new image ───
function loadNextImage() {
    const img = images[currentImageIndex % images.length];
    splatData = sampleImageToSplats(img);
    gl.bindBuffer(gl.ARRAY_BUFFER, instanceBuf);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, splatData);
    currentImageIndex++;
}

// ─── Render loop ───
function render(now) {
    requestAnimationFrame(render);
    now *= 0.001; // ms -> sec
    const dt = lastTime ? Math.min(now - lastTime, 0.1) : 0.016;
    lastTime = now;

    if (state === 'LOADING') return;

    resize();

    stateTime += dt;

    let dissolve = 0;

    switch (state) {
        case 'HOLDING':
            dissolve = 0;
            if (stateTime >= HOLD_TIME) {
                state = 'DISSOLVING';
                stateTime = 0;
            }
            break;
        case 'DISSOLVING':
            dissolve = Math.min(stateTime / DISSOLVE_TIME, 1.0);
            if (stateTime >= DISSOLVE_TIME) {
                state = 'BLACK';
                stateTime = 0;
            }
            break;
        case 'BLACK':
            dissolve = 1.0;
            if (stateTime >= BLACK_TIME) {
                loadNextImage();
                state = 'HOLDING';
                stateTime = 0;
            }
            break;
    }

    // Clear
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    if (state === 'BLACK' && stateTime > 0.5) return; // fully faded

    // Blending: premultiplied alpha
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);

    gl.useProgram(program);
    gl.uniform2f(uni.resolution, canvas.width, canvas.height);
    gl.uniform1f(uni.dissolve, dissolve);
    gl.uniform1f(uni.time, now);

    gl.bindVertexArray(vao);
    gl.drawArraysInstanced(gl.TRIANGLES, 0, 6, NUM_SPLATS);
    gl.bindVertexArray(null);

    gl.disable(gl.BLEND);
}

// ─── Fullscreen ───
function toggleFullscreen() {
    if (!document.fullscreenElement) {
        canvas.requestFullscreen().catch(() => {});
    } else {
        document.exitFullscreen();
    }
}

document.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') toggleFullscreen();
});
canvas.addEventListener('click', toggleFullscreen);

// ─── Init ───
async function init() {
    resize();
    images = await loadImages();
    console.log(`Loaded ${images.length} image(s), sampling ${NUM_SPLATS} splats each`);
    loadNextImage();
    state = 'HOLDING';
    stateTime = 0;
    requestAnimationFrame(render);
}

init();
