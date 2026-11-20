/**
 * GLSL ES 3.00 shaders for the WebGL2 restoration pipeline.
 *
 * The fragment shader math is a line-for-line mirror of invertPixel() in
 * model.ts: same transmission exponential, same ambient + strobe(t^2) path
 * gain, same veil term and the same min-path-gain floor. A parity test
 * renders a tile on GPU and compares it against the CPU reference.
 *
 * Per-pixel physical factors are precomputed on the CPU into a float texture
 * (see FactorTextures) so the shader stays branch-light and the same shader
 * serves calibrated and blind modes — only the source of the factors differs.
 */

export const FULLSCREEN_VS = `#version 300 es
layout(location=0) in vec2 a_position;
out vec2 v_uv;
void main() {
  // Fullscreen triangle covering clip space with a standard 0..1 UV. Texture
  // blocks are pre-flipped on upload (top-origin image -> bottom-origin GL),
  // and every pass uses this same convention, so no per-pass UV flip is
  // needed — flipping here twice across chained passes would invert them.
  v_uv = a_position * 0.5 + 0.5;
  gl_Position = vec4(a_position, 0.0, 1.0);
}`;

/**
 * Restoration pass.
 *
 * Uniforms:
 *   u_image      RGBA16F, underwater linear RGB (A unused)
 *   u_factors    RGBA16F, per-pixel (tR, tG, tB, strobeG)
 *   u_ambient    vec3 downwelling E_k(z)
 *   u_veil       vec3 veil color (linear)
 *   u_eta        float strobe backscatter coupling
 *   u_minPath    float path-gain floor
 *   u_maxGain    float maximum restoration gain (noise-aware clamp)
 *   u_reg        float Wiener regularization (0 = exact inversion)
 */
export const RESTORE_FS = `#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform sampler2D u_factors;
uniform vec3 u_ambient;
uniform vec3 u_veil;
uniform float u_eta;
uniform float u_minPath;
uniform float u_maxGain;
uniform float u_reg;
in vec2 v_uv;
out vec4 outColor;

void main() {
  vec3 m = texture(u_image, v_uv).rgb;
  vec4 f = texture(u_factors, v_uv);
  vec3 t = f.rgb;
  float G = f.a;

  vec3 veilTerm = u_veil * (1.0 - t) * (1.0 + u_eta * G);
  vec3 physicalGain = u_ambient * t + G * t * t;
  vec3 pathGain = max(physicalGain, max(vec3(u_minPath), vec3(1.0 / u_maxGain)));
  vec3 residual = m - veilTerm;
  vec3 restored = residual / pathGain;
  // Gated Wiener: healthy channels (gain >= 0.25) invert exactly; low-SNR
  // channels (deep red) get a gray-prior shrink to suppress false color.
  if (u_reg > 0.0) {
    vec3 wiener = (pathGain * pathGain) / (pathGain * pathGain + vec3(u_reg));
    vec3 gated = mix(wiener, vec3(1.0), step(vec3(0.25), pathGain));
    restored *= gated;
  }
  outColor = vec4(restored, 1.0);
}`;

/**
 * Skin constraint pass.
 *
 * YCbCr locus detection in encoded sRGB, gamut target in Lab. Lab<->RGB is
 * done via XYZ with the D65 white, matching skin.ts. Weights and the max
 * blend are identical to the CPU routine.
 *
 * Uniforms:
 *   u_restored  RGBA16F restored linear image
 *   u_maxBlend  float
 */
export const SKIN_FS = `#version 300 es
precision highp float;
uniform sampler2D u_restored;
uniform float u_maxBlend;
in vec2 v_uv;
out vec4 outColor;

vec3 srgbEncode(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0/2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

vec3 srgbDecode(vec3 c) {
  vec3 lo = c / 12.92;
  vec3 hi = pow((c + 0.055) / 1.055, vec3(2.4));
  return mix(lo, hi, step(vec3(0.04045), c));
}

// BT.601 YCbCr, studio 0..255 scale.
vec3 rgbToYCbCr(vec3 rgb) {
  float Y = 16.0 + 65.481*rgb.r + 128.553*rgb.g + 24.966*rgb.b;
  float Cb = 128.0 - 37.797*rgb.r - 74.203*rgb.g + 112.0*rgb.b;
  float Cr = 128.0 + 112.0*rgb.r - 93.786*rgb.g - 18.214*rgb.b;
  return vec3(Y, Cb, Cr);
}

vec3 linearRgbToXyz(vec3 rgb) {
  return vec3(
    dot(vec3(0.4124564, 0.3575761, 0.1804375), rgb),
    dot(vec3(0.2126729, 0.7151522, 0.0721750), rgb),
    dot(vec3(0.0193339, 0.1191920, 0.9503041), rgb)
  );
}

vec3 xyzToLab(vec3 xyz) {
  vec3 n = xyz / vec3(0.95047, 1.0, 1.08883);
  float d = 6.0/29.0;
  vec3 f = mix(n / (3.0*d*d) + 4.0/29.0, pow(max(n, vec3(0.0)), vec3(1.0/3.0)),
               step(vec3(d*d*d), n));
  return vec3(116.0*f.y - 16.0, 500.0*(f.x-f.y), 200.0*(f.y-f.z));
}

vec3 labToXyz(vec3 lab) {
  float fy = (lab.x + 16.0) / 116.0;
  float fx = lab.y / 500.0 + fy;
  float fz = fy - lab.z / 200.0;
  float d = 6.0/29.0;
  vec3 f = vec3(fx, fy, fz);
  vec3 finv = mix(3.0*d*d*(f - 4.0/29.0), f*f*f, step(vec3(d), f));
  return finv * vec3(0.95047, 1.0, 1.08883);
}

vec3 xyzToLinearRgb(vec3 xyz) {
  return vec3(
    dot(vec3( 3.2404542, -1.5371385, -0.4985314), xyz),
    dot(vec3(-0.9692660,  1.8760108,  0.0415560), xyz),
    dot(vec3( 0.0556434, -0.2040259,  1.0572252), xyz)
  );
}

float skinWeight(vec3 rgb) {
  vec3 ycc = rgbToYCbCr(rgb);
  if (ycc.x < 45.0 || ycc.x > 232.0) return 0.0;
  float ex = (ycc.y - 104.5) / 24.0;
  float ey = (ycc.z - 154.5) / 20.0;
  float dist = sqrt(ex*ex + ey*ey);
  float inner = 1.0/1.15;
  float outer = 1.15;
  if (dist <= inner) return 1.0;
  if (dist >= outer) return 0.0;
  float t = (dist - inner) / (outer - inner);
  float s = t*t*(3.0 - 2.0*t);
  return 1.0 - s;
}

vec3 projectSkin(vec3 lab) {
  vec3 L = clamp(lab, vec3(28.0, 6.0, 12.0), vec3(90.0, 32.0, 42.0));
  float tt = (L.x - 28.0) / (90.0 - 28.0);
  float aTarget = 12.0 + 6.0*tt;
  float bTarget = 17.0 + 13.0*tt;
  return vec3(L.x, 0.5*L.y + 0.5*aTarget, 0.5*L.z + 0.5*bTarget);
}

void main() {
  vec3 lin = texture(u_restored, v_uv).rgb;
  vec3 encoded = clamp(srgbEncode(lin), 0.0, 1.0);
  float w = skinWeight(encoded);
  if (w <= 0.0) {
    outColor = vec4(lin, 1.0);
    return;
  }
  vec3 lab = xyzToLab(linearRgbToXyz(max(lin, vec3(0.0))));
  vec3 targetLab = projectSkin(lab);
  vec3 targetRgb = xyzToLinearRgb(labToXyz(targetLab));
  float blend = u_maxBlend * w;
  outColor = vec4(mix(lin, targetRgb, blend), 1.0);
}`;

/**
 * Display pass: linear HDR -> Reinhard -> sRGB encode -> 0..1.
 */
export const DISPLAY_FS = `#version 300 es
precision highp float;
uniform sampler2D u_image;
uniform float u_exposure;
in vec2 v_uv;
out vec4 outColor;

vec3 srgbEncode(vec3 c) {
  vec3 lo = c * 12.92;
  vec3 hi = 1.055 * pow(max(c, vec3(0.0)), vec3(1.0/2.4)) - 0.055;
  return mix(lo, hi, step(vec3(0.0031308), c));
}

void main() {
  vec3 x = max(vec3(0.0), texture(u_image, v_uv).rgb * u_exposure);
  vec3 mapped = x / (1.0 + x);
  outColor = vec4(clamp(srgbEncode(mapped), 0.0, 1.0), 1.0);
}`;
