// ─── Gaussian Blur Dissolution ───
// Approach B: Directional blur that smears the image leftward like watercolor bleeding

const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', { alpha: false, antialias: false });
if (!gl) { document.body.textContent = 'WebGL2 required'; throw new Error('No WebGL2'); }

// ─── Constants ───
const HOLD_TIME = 3.0;
const DISSOLVE_TIME = 6.0;
const BLACK_TIME = 2.0;
const NUM_SAMPLES = 24;

// ─── State ───
let state = 'LOADING';
let stateTime = 0;
let currentImageIndex = 0;
let textures = [];
let lastTime = 0;
let globalTime = 0;

// ─── Shaders ───
const VERT_SRC = /*glsl*/`#version 300 es
precision highp float;

in vec2 a_position;
out vec2 v_uv;

void main() {
    v_uv = a_position * 0.5 + 0.5;
    gl_Position = vec4(a_position, 0.0, 1.0);
}
`;

const FRAG_SRC = /*glsl*/`#version 300 es
precision highp float;

in vec2 v_uv;
out vec4 fragColor;

uniform sampler2D u_image;
uniform float u_dissolve;  // 0..1
uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_imageSize;

// ─── Ashima Arts simplex 3D noise ───
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

    vec4 norm = taylorInvSqrt(vec4(dot(p0,p0), dot(p1,p1), dot(p2,p2), dot(p3,p3)));
    p0 *= norm.x; p1 *= norm.y; p2 *= norm.z; p3 *= norm.w;

    vec4 m = max(0.6 - vec4(dot(x0,x0), dot(x1,x1), dot(x2,x2), dot(x3,x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0,x0), dot(p1,x1), dot(p2,x2), dot(p3,x3)));
}

void main() {
    vec2 uv = v_uv;

    // Compute cover-fit UVs (image fills canvas, cropped to fit)
    float canvasAspect = u_resolution.x / u_resolution.y;
    float imageAspect = u_imageSize.x / u_imageSize.y;
    vec2 coverUV = uv;
    if (canvasAspect > imageAspect) {
        // Canvas is wider — crop top/bottom
        float scale = canvasAspect / imageAspect;
        coverUV.y = (uv.y - 0.5) * scale + 0.5;
    } else {
        // Canvas is taller — crop sides
        float scale = imageAspect / canvasAspect;
        coverUV.x = (uv.x - 0.5) * scale + 0.5;
    }

    // Directional dissolve: left regions dissolve first
    float normalizedX = coverUV.x;
    float xDelay = normalizedX * 0.6; // left=0 delay, right=0.6 delay
    float d = clamp((u_dissolve - xDelay) / (1.0 - xDelay + 0.001), 0.0, 1.0);

    // When fully dissolved or out of image bounds, output black
    if (d >= 1.0 || coverUV.x < 0.0 || coverUV.x > 1.0 || coverUV.y < 0.0 || coverUV.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }

    // When no dissolve, output crisp image
    if (d <= 0.0) {
        fragColor = vec4(texture(u_image, coverUV).rgb, 1.0);
        return;
    }

    // ─── Directional Gaussian blur dissolution ───
    // Blur radius grows quadratically for organic acceleration
    float blurSize = d * d * 0.18;

    vec4 color = vec4(0.0);
    float totalWeight = 0.0;

    for (int i = 0; i < ${NUM_SAMPLES}; i++) {
        float t = float(i) / float(${NUM_SAMPLES} - 1);
        // Gaussian weight — samples closer to origin are heavier
        float weight = exp(-t * t * 3.0);

        // Primary offset: rightward (we sample from the right to smear leftward)
        vec2 offset = vec2(t * blurSize, 0.0);

        // Organic vertical wobble via noise — painterly, not mechanical
        float wobble = snoise(vec3(coverUV * 5.0 + t * 2.0, u_time * 0.2));
        offset.y += wobble * blurSize * 0.15;

        // Additional subtle noise on horizontal offset for organic quality
        float hNoise = snoise(vec3(coverUV * 3.0 + t * 1.5, u_time * 0.15 + 10.0));
        offset.x += hNoise * blurSize * 0.05;

        vec2 sampleUV = coverUV + offset;

        // Clamp to image bounds
        sampleUV = clamp(sampleUV, vec2(0.0), vec2(1.0));

        color += texture(u_image, sampleUV) * weight;
        totalWeight += weight;
    }
    color /= totalWeight;

    // Fade out opacity as dissolve progresses
    float alpha = 1.0 - smoothstep(0.4, 1.0, d);

    // Slight desaturation as it dissolves — watercolor wash effect
    float lum = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    float desatAmount = smoothstep(0.2, 0.8, d) * 0.3;
    vec3 finalColor = mix(color.rgb, vec3(lum), desatAmount);

    fragColor = vec4(finalColor * alpha, 1.0);
}
`;

// ─── Compile shaders ───
function createShader(type, src) {
    const s = gl.createShader(type);
    gl.shaderSource(s, src);
    gl.compileShader(s);
    if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
        console.error('Shader error:', gl.getShaderInfoLog(s));
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
        console.error('Link error:', gl.getProgramInfoLog(p));
        throw new Error('Program link error');
    }
    return p;
}

const vs = createShader(gl.VERTEX_SHADER, VERT_SRC);
const fs = createShader(gl.FRAGMENT_SHADER, FRAG_SRC);
const program = createProgram(vs, fs);

// ─── Locations ───
const aPosition = gl.getAttribLocation(program, 'a_position');
const uImage = gl.getUniformLocation(program, 'u_image');
const uDissolve = gl.getUniformLocation(program, 'u_dissolve');
const uTime = gl.getUniformLocation(program, 'u_time');
const uResolution = gl.getUniformLocation(program, 'u_resolution');
const uImageSize = gl.getUniformLocation(program, 'u_imageSize');

// ─── Fullscreen quad ───
const vao = gl.createVertexArray();
gl.bindVertexArray(vao);

const quadBuf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, quadBuf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1,  1, 1,
    -1, -1,  1,  1, -1, 1,
]), gl.STATIC_DRAW);
gl.enableVertexAttribArray(aPosition);
gl.vertexAttribPointer(aPosition, 2, gl.FLOAT, false, 0, 0);

gl.bindVertexArray(null);

// ─── Image loading ───

function generateFallbackImage(palette) {
    const w = 1920, h = 1080;
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    const ctx = c.getContext('2d');

    // Base wash
    ctx.fillStyle = palette.bg;
    ctx.fillRect(0, 0, w, h);

    // Watercolor blobs
    const blobs = 10 + Math.floor(Math.random() * 8);
    for (let i = 0; i < blobs; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const r = 150 + Math.random() * 400;
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

    // Multiply overlay
    ctx.globalCompositeOperation = 'multiply';
    for (let i = 0; i < 4; i++) {
        const x = Math.random() * w;
        const y = Math.random() * h;
        const r = 200 + Math.random() * 500;
        const grad = ctx.createRadialGradient(x, y, 0, x, y, r);
        grad.addColorStop(0, palette.overlay + '66');
        grad.addColorStop(1, palette.overlay + '00');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, w, h);
    }
    ctx.globalCompositeOperation = 'source-over';

    return { canvas: c, width: w, height: h };
}

const palettes = [
    { bg: '#0a1520', colors: ['#4a7fb5', '#6ba3d6', '#2c5a8a', '#89c4e8', '#1a3a5c', '#c8dff0'], overlay: '#1a3050' },
    { bg: '#1a1208', colors: ['#8b6914', '#c49a3c', '#5c4a1e', '#d4a84b', '#3a2e10', '#e8c882'], overlay: '#2a1e08' },
    { bg: '#0a1a10', colors: ['#3a7a4a', '#5aaa6a', '#2a5a38', '#7aca8a', '#1a3a22', '#a8dab0'], overlay: '#0a2a15' },
];

function createTextureFromImage(img) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, img);
    return tex;
}

function createTextureFromCanvas(cvs) {
    const tex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, cvs);
    return tex;
}

// Image dimensions stored alongside textures
let imageSizes = [];

async function loadImages() {
    const imageNames = ['lerin1', 'lerin2', 'lerin3', 'lerin-1', 'lerin-2', 'lerin-3', '01', '02', '03', '1', '2', '3'];
    const extensions = ['jpg', 'jpeg', 'png'];
    const loaded = [];

    for (const name of imageNames) {
        for (const ext of extensions) {
            if (loaded.length >= 6) break;
            try {
                const img = new Image();
                img.crossOrigin = 'anonymous';
                const url = `../shared/images/${name}.${ext}`;
                await new Promise((resolve, reject) => {
                    img.onload = resolve;
                    img.onerror = reject;
                    img.src = url;
                });
                const tex = createTextureFromImage(img);
                loaded.push({ texture: tex, width: img.naturalWidth, height: img.naturalHeight });
            } catch (e) { /* skip */ }
        }
    }

    if (loaded.length > 0) return loaded;

    // Fallback: generate watercolor images
    console.log('No images found — generating fallback watercolors');
    return palettes.map(p => {
        const result = generateFallbackImage(p);
        const tex = createTextureFromCanvas(result.canvas);
        return { texture: tex, width: result.width, height: result.height };
    });
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

// ─── Render loop ───
function render(now) {
    requestAnimationFrame(render);
    now *= 0.001;
    const dt = lastTime ? Math.min(now - lastTime, 0.1) : 0.016;
    lastTime = now;
    globalTime += dt;

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
            // Ease-in curve for gentle start, accelerating dissolution
            const t = Math.min(stateTime / DISSOLVE_TIME, 1.0);
            dissolve = t * t * (3.0 - 2.0 * t); // smoothstep
            if (stateTime >= DISSOLVE_TIME) {
                state = 'BLACK';
                stateTime = 0;
            }
            break;
        case 'BLACK':
            dissolve = 1.0;
            if (stateTime >= BLACK_TIME) {
                currentImageIndex = (currentImageIndex + 1) % textures.length;
                state = 'HOLDING';
                stateTime = 0;
            }
            break;
    }

    // Clear to black
    gl.clearColor(0, 0, 0, 1);
    gl.clear(gl.COLOR_BUFFER_BIT);

    // Skip drawing when fully black
    if (state === 'BLACK' && stateTime > 0.1) return;

    const img = textures[currentImageIndex];

    gl.useProgram(program);
    gl.uniform1f(uDissolve, dissolve);
    gl.uniform1f(uTime, globalTime);
    gl.uniform2f(uResolution, canvas.width, canvas.height);
    gl.uniform2f(uImageSize, img.width, img.height);

    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_2D, img.texture);
    gl.uniform1i(uImage, 0);

    gl.bindVertexArray(vao);
    gl.drawArrays(gl.TRIANGLES, 0, 6);
    gl.bindVertexArray(null);
}

// ─── Fullscreen ───
document.addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') {
        if (!document.fullscreenElement) {
            canvas.requestFullscreen().catch(() => {});
        } else {
            document.exitFullscreen();
        }
    }
});

// ─── Init ───
async function init() {
    resize();
    textures = await loadImages();
    console.log(`Loaded ${textures.length} image(s) — directional blur dissolution`);
    state = 'HOLDING';
    stateTime = 0;
    requestAnimationFrame(render);
}

init();
