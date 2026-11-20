/**
 * WebGL2 execution pipeline.
 *
 * Design for a 30 MP photo in a browser:
 *   - All working textures are RGBA32F (full float), color-renderable in
 *     WebGL2 via EXT_color_buffer_float. Tiling bounds VRAM regardless of the
 *     total megapixel count, so the image is processed in
 *     TILES with a small ping-pong set of tile-sized textures.
 *   - Per-pixel physical factors (tR,tG,tB,G) are built on the CPU from the
 *     distance map — the same formulas as model.ts — and streamed into the
 *     tile's factor texture. Blind estimation runs entirely on the CPU on a
 *     downscaled image and returns a full-res distance map before GPU work.
 *   - The restore -> skin -> display chain is one render per tile per pass.
 *
 * The class is environment-agnostic: a headless test passes a WebGL2 context
 * from Playwright; the demo passes a canvas context. No DOM code lives here.
 */

import {
  type FloatImage,
  type RestoreParams,
  depthOptics,
  restoreBlind,
  transmission,
} from "./model.js";
import { strobeGeometry } from "./strobe.js";
import {
  DISPLAY_FS,
  FULLSCREEN_VS,
  RESTORE_FS,
  SKIN_FS,
} from "./shaders.js";

/**
 * Pack a Float32 value into an IEEE 754 half-float (uint16 bit pattern).
 * RGBA16F textures require HALF_FLOAT (Uint16Array) uploads; feeding them a
 * Float32Array is a GL INVALID_OPERATION and silently leaves the texture
 * empty, so every float upload goes through this conversion.
 */
export function floatToHalfFloat(value: number): number {
  const x = value;
  if (!Number.isFinite(x)) return x > 0 ? 0x7c00 : 0xfc00; // ±inf / NaN
  const f32 = new Float32Array(1);
  const u32 = new Uint32Array(f32.buffer);
  f32[0] = x;
  const bits = u32[0]!;
  const sign = (bits >>> 16) & 0x8000;
  let mantissa = bits & 0x7fffff;
  let exp = (bits >>> 23) & 0xff;

  if (exp === 0) return sign; // ±0 / subnormal f32 -> ±0 half
  exp -= 127 - 15; // rebase exponent from f32 bias (127) to f16 bias (15)
  if (exp >= 0x1f) return sign | 0x7c00; // overflow -> ±inf
  if (exp <= 0) {
    if (exp < -10) return sign; // underflow -> ±0
    // f16 subnormal: prepend implicit f32 leading 1 then shift.
    mantissa = (mantissa | 0x800000) >>> (1 - exp + 13);
    return sign | mantissa;
  }
  // Round-to-nearest-even with the 13 dropped mantissa bits.
  const rounded = (mantissa + 0x7ff + ((mantissa >>> 13) & 1)) >>> 13;
  return sign | (exp << 10) | (rounded & 0x3ff);
}

/** Pack an entire RGBA Float32 tile into a half-float Uint16Array. */
function packHalfFloatRGBA(src: Float32Array): Uint16Array {
  const out = new Uint16Array(src.length);
  for (let i = 0; i < src.length; i++) out[i] = floatToHalfFloat(src[i]!);
  return out;
}

export interface GpuRestoreOptions {
  params: RestoreParams;
  /** Apply the skin gamut constraint after inversion. */
  applySkin?: boolean;
  maxSkinBlend?: number;
  /** Output encoded sRGB bytes (display-ready) instead of linear float. */
  encodeSrgb?: boolean;
  exposure?: number;
  /** Tile edge length in pixels; smaller = less VRAM, more draw calls. */
  tileSize?: number;
}

export interface GpuRestoreResult {
  /** Linear restored image when encodeSrgb is false. */
  linear?: FloatImage;
  /** Packed sRGB bytes when encodeSrgb is true. */
  srgb?: Uint8ClampedArray;
  width: number;
  height: number;
  strobePower: number;
  gpuMs: number;
}

function compile(gl: WebGL2RenderingContext, type: number, src: string): WebGLShader {
  const sh = gl.createShader(type)!;
  gl.shaderSource(sh, src);
  gl.compileShader(sh);
  if (!gl.getShaderParameter(sh, gl.COMPILE_STATUS)) {
    const log = gl.getShaderInfoLog(sh);
    gl.deleteShader(sh);
    throw new Error("Shader compile error: " + log);
  }
  return sh;
}

function link(gl: WebGL2RenderingContext, vsSrc: string, fsSrc: string): WebGLProgram {
  const prog = gl.createProgram()!;
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, vsSrc));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fsSrc));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) {
    const log = gl.getProgramInfoLog(prog);
    gl.deleteProgram(prog);
    throw new Error("Program link error: " + log);
  }
  return prog;
}

type ProgramSet = {
  restore: WebGLProgram;
  skin: WebGLProgram;
  display: WebGLProgram;
};

/**
 * Build the (tR,tG,tB,G) factor quadruple for every pixel, mirroring the
 * CPU loops in restoreCalibrated/restoreBlind exactly.
 */
export function buildFactorPixels(
  width: number,
  height: number,
  distance: Float32Array,
  params: RestoreParams,
): Float32Array {
  const { c } = depthOptics(params.waterType, params.depthM);
  const out = new Float32Array(width * height * 4);
  for (let y = 0; y < height; y++) {
    const ny = height === 1 ? 0 : 1 - (2 * y) / (height - 1);
    for (let x = 0; x < width; x++) {
      const nx = width === 1 ? 0 : (2 * x) / (width - 1) - 1;
      const i = y * width + x;
      const d = distance[i]!;
      const t = transmission(c, d);
      const G = strobeGeometry(nx, ny, d, params.strobe);
      const q = i * 4;
      out[q] = t[0];
      out[q + 1] = t[1];
      out[q + 2] = t[2];
      out[q + 3] = G;
    }
  }
  return out;
}

export class WebGL2Restorer {
  /** Texture unit reserved for FBO/scratch bindings, never used by samplers. */
  static readonly SCRATCH_UNIT = 7;
  private gl: WebGL2RenderingContext;
  private programs: ProgramSet;
  private vao: WebGLVertexArrayObject;
  private halfFloatExt: object | null;

  constructor(gl: WebGL2RenderingContext) {
    this.gl = gl;
    this.halfFloatExt = gl.getExtension("EXT_color_buffer_float");
    if (!this.halfFloatExt) {
      throw new Error("RGBA32F render targets unsupported (EXT_color_buffer_float missing)");
    }
    this.programs = {
      restore: link(gl, FULLSCREEN_VS, RESTORE_FS),
      skin: link(gl, FULLSCREEN_VS, SKIN_FS),
      display: link(gl, FULLSCREEN_VS, DISPLAY_FS),
    };
    // Fullscreen triangle.
    const vao = gl.createVertexArray()!;
    gl.bindVertexArray(vao);
    const vbo = gl.createBuffer()!;
    gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array([-1, -1, 3, -1, -1, 3]),
      gl.STATIC_DRAW,
    );
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 2, gl.FLOAT, false, 0, 0);
    this.vao = vao;
  }

  private makeFloatTexture(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    // Build on a scratch unit so sampler units 0/1 keep their input bindings.
    // RGBA32F full float: the underwater red channel carries tiny linear
    // values whose half-float quantization would be amplified by the inverse
    // path gain into visible errors, so the working/input textures stay 32F.
    gl.activeTexture(gl.TEXTURE0 + WebGL2Restorer.SCRATCH_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA32F,
      width,
      height,
      0,
      gl.RGBA,
      gl.FLOAT,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private makeOutputTexture(width: number, height: number): WebGLTexture {
    const gl = this.gl;
    const tex = gl.createTexture()!;
    gl.activeTexture(gl.TEXTURE0 + WebGL2Restorer.SCRATCH_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      gl.RGBA8,
      width,
      height,
      0,
      gl.RGBA,
      gl.UNSIGNED_BYTE,
      null,
    );
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    return tex;
  }

  private uploadTile(
    tex: WebGLTexture,
    width: number,
    height: number,
    pixels: Float32Array | Uint8ClampedArray,
    internalFormat: number,
    format: number,
    type: number,
  ): void {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + WebGL2Restorer.SCRATCH_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // Half-float render targets require a Uint16 bit-pattern source.
    const data: Float32Array | Uint8ClampedArray | Uint16Array =
      type === gl.HALF_FLOAT && pixels instanceof Float32Array
        ? packHalfFloatRGBA(pixels)
        : pixels;
    gl.texImage2D(
      gl.TEXTURE_2D,
      0,
      internalFormat,
      width,
      height,
      0,
      format,
      type,
      data,
    );
  }

  /**
   * Full restoration with tiling. Blind mode first estimates a distance map
   * on the CPU (downscaled internally), then GPU inversion runs per tile.
   */
  restore(image: FloatImage, opts: GpuRestoreOptions): GpuRestoreResult {
    const gl = this.gl;
    const { width, height } = image;
    const started = performance.now();

    // Resolve distance + strobe parameters.
    let distance = opts.params.distance;
    let strobePower = opts.params.strobe.power;
    if (!distance || opts.params.estimateDistance) {
      const blind = restoreBlind(image, opts.params);
      distance = blind.distance;
      strobePower = blind.strobePower;
    }
    const effectiveParams: RestoreParams = {
      ...opts.params,
      distance,
      strobe: { ...opts.params.strobe, power: strobePower },
    };
    const { ambient, veil } = depthOptics(
      effectiveParams.waterType,
      effectiveParams.depthM,
    );
    const eta = effectiveParams.strobeBackscatterCoupling ?? 0.55;
    const minPath = effectiveParams.minPathGain ?? 1e-4;
    // GLSL has no Infinity uniform; a huge finite value makes 1/maxGain ~0,
    // i.e. the noise-aware clamp is effectively disabled by default.
    const maxGain = Number.isFinite(effectiveParams.maxGain ?? Infinity)
      ? (effectiveParams.maxGain as number)
      : 1e9;
    const regularization = effectiveParams.regularization ?? 0;
    const tile = opts.tileSize ?? 1024;
    const encode = opts.encodeSrgb ?? true;
    const applySkin = opts.applySkin ?? true;

    const fbo = gl.createFramebuffer()!;

    const factorsFull = buildFactorPixels(width, height, distance!, effectiveParams);
    const finalLinear = encode ? null : new Float32Array(width * height * 3);
    const finalBytes = encode ? new Uint8ClampedArray(width * height * 4) : null;

    const useProgram = (prog: WebGLProgram) => {
      gl.useProgram(prog);
      gl.bindVertexArray(this.vao);
    };
    // Samplers live on fixed units so creating/binding framebuffer textures
    // on a separate scratch unit can never clobber an input binding.
    const UNIT_IMAGE = 0;
    const UNIT_FACTORS = 1;
    const UNIT_SCRATCH = 7;
    const bindInput = (unit: number, tex: WebGLTexture, locName: string) => {
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, tex);
      const loc = gl.getUniformLocation(gl.getParameter(gl.CURRENT_PROGRAM), locName);
      if (loc !== null) gl.uniform1i(loc, unit);
    };

    for (let ty = 0; ty < height; ty += tile) {
      const th = Math.min(tile, height - ty);
      for (let tx = 0; tx < width; tx += tile) {
        const tw = Math.min(tile, width - tx);
        // Per-block textures match the block exactly, so the fullscreen
        // triangle spans the whole texture (v_uv 0..1) with no partial-fill
        // or sub-viewport coordinate ambiguity.
        const imgTex = this.makeFloatTexture(tw, th);
        const facTex = this.makeFloatTexture(tw, th);
        const workTexA = this.makeFloatTexture(tw, th);
        const workTexB = this.makeFloatTexture(tw, th);
        const outTex = encode ? this.makeOutputTexture(tw, th) : workTexB;

        this.uploadBlock(image, imgTex, tw, th, tx, ty);
        this.uploadFactorBlock(factorsFull, width, facTex, tw, th, tx, ty);

        // Pass 1: restore into workTexA.
        useProgram(this.programs.restore);
        bindInput(0, imgTex, "u_image");
        bindInput(1, facTex, "u_factors");
        const rp = gl.getParameter(gl.CURRENT_PROGRAM);
        this.uniform3fv(rp, "u_ambient", ambient);
        this.uniform3fv(rp, "u_veil", veil);
        gl.uniform1f(gl.getUniformLocation(rp, "u_eta"), eta);
        gl.uniform1f(gl.getUniformLocation(rp, "u_minPath"), minPath);
        gl.uniform1f(gl.getUniformLocation(rp, "u_maxGain"), maxGain);
        gl.uniform1f(gl.getUniformLocation(rp, "u_reg"), regularization);
        this.renderTo(fbo, workTexA, tw, th);

        let current = workTexA;
        // Pass 2: skin constraint into workTexB.
        if (applySkin) {
          useProgram(this.programs.skin);
          bindInput(0, current, "u_restored");
          gl.uniform1f(
            gl.getUniformLocation(gl.getParameter(gl.CURRENT_PROGRAM), "u_maxBlend"),
            opts.maxSkinBlend ?? 0.75,
          );
          this.renderTo(fbo, workTexB, tw, th);
          current = workTexB;
        }

        if (encode) {
          // Pass 3: display encode into RGBA8.
          useProgram(this.programs.display);
          bindInput(0, current, "u_image");
          gl.uniform1f(
            gl.getUniformLocation(gl.getParameter(gl.CURRENT_PROGRAM), "u_exposure"),
            opts.exposure ?? 1,
          );
          this.renderTo(fbo, outTex, tw, th);
          this.readbackBlock(finalBytes!, width, height, fbo, outTex, tx, ty, tw, th, true);
        } else {
          this.readbackBlock(finalLinear!, width, height, fbo, current, tx, ty, tw, th, false);
        }

        [imgTex, facTex, workTexA, workTexB].forEach((t) => gl.deleteTexture(t));
        if (encode) gl.deleteTexture(outTex);
      }
    }

    gl.deleteFramebuffer(fbo);

    const gpuMs = performance.now() - started;
    return encode
      ? { srgb: finalBytes!, width, height, strobePower, gpuMs }
      : {
          linear: { width, height, data: finalLinear! },
          width,
          height,
          strobePower,
          gpuMs,
        };
  }

  private uniform3fv(program: WebGLProgram, name: string, v: ArrayLike<number>): void {
    const loc = this.gl.getUniformLocation(program, name);
    if (loc !== null) this.gl.uniform3fv(loc, v as Float32Array);
  }

  private renderTo(
    fbo: WebGLFramebuffer,
    tex: WebGLTexture,
    w: number,
    h: number,
  ): void {
    const gl = this.gl;
    // Bind the render target on the scratch unit; sampler units 0/1 are left
    // untouched and must never reference this texture (no feedback loop).
    gl.activeTexture(gl.TEXTURE0 + WebGL2Restorer.SCRATCH_UNIT);
    gl.bindTexture(gl.TEXTURE_2D, tex);
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(
      gl.FRAMEBUFFER,
      gl.COLOR_ATTACHMENT0,
      gl.TEXTURE_2D,
      tex,
      0,
    );
    gl.viewport(0, 0, w, h);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }

  /** Pack an RGB float block (tw x th) into an exactly-sized RGBA texture. */
  private uploadBlock(
    image: FloatImage,
    tex: WebGLTexture,
    tw: number,
    th: number,
    tx: number,
    ty: number,
  ): void {
    const gl = this.gl;
    const buf = new Float32Array(tw * th * 4);
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const src = ((ty + y) * image.width + (tx + x)) * 3;
        // Flip the top-origin image into GL's bottom-origin texture layout
        // once, at upload. All downstream passes then use a standard 0..1 UV
        // with the same convention, and readback flips back to top-origin.
        const texY = th - 1 - y;
        const dst = (texY * tw + x) * 4;
        buf[dst] = image.data[src]!;
        buf[dst + 1] = image.data[src + 1]!;
        buf[dst + 2] = image.data[src + 2]!;
        buf[dst + 3] = 1;
      }
    }
    this.uploadTile(tex, tw, th, buf, gl.RGBA32F, gl.RGBA, gl.FLOAT);
  }

  /** Pack the (tR,tG,tB,G) factor block into an exactly-sized RGBA texture. */
  private uploadFactorBlock(
    factorsFull: Float32Array,
    fullWidth: number,
    tex: WebGLTexture,
    tw: number,
    th: number,
    tx: number,
    ty: number,
  ): void {
    const gl = this.gl;
    const buf = new Float32Array(tw * th * 4);
    for (let y = 0; y < th; y++) {
      for (let x = 0; x < tw; x++) {
        const src = ((ty + y) * fullWidth + (tx + x)) * 4;
        // Match uploadBlock's bottom-origin flip so per-pixel factors align
        // with the image texel under the standard UV.
        const texY = th - 1 - y;
        const dst = (texY * tw + x) * 4;
        buf[dst] = factorsFull[src]!;
        buf[dst + 1] = factorsFull[src + 1]!;
        buf[dst + 2] = factorsFull[src + 2]!;
        buf[dst + 3] = factorsFull[src + 3]!;
      }
    }
    this.uploadTile(tex, tw, th, buf, gl.RGBA32F, gl.RGBA, gl.FLOAT);
  }

  private readbackBlock(
    target: Uint8ClampedArray | Float32Array,
    fullWidth: number,
    fullHeight: number,
    fbo: WebGLFramebuffer,
    tex: WebGLTexture,
    tx: number,
    ty: number,
    tw: number,
    th: number,
    bytes: boolean,
  ): void {
    const gl = this.gl;
    gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, tex, 0);
    gl.viewport(0, 0, tw, th);
    const channels = 4;
    const buf = bytes
      ? new Uint8Array(tw * th * channels)
      : new Float32Array(tw * th * channels);
    gl.readPixels(0, 0, tw, th, gl.RGBA, bytes ? gl.UNSIGNED_BYTE : gl.FLOAT, buf);
    // readPixels returns bottom-origin rows; flip back to the top-origin
    // image layout while placing into the full-frame target.
    for (let y = 0; y < th; y++) {
      const srcRow = th - 1 - y;
      for (let x = 0; x < tw; x++) {
        const dstX = tx + x;
        const dstY = ty + y;
        const srcIdx = (srcRow * tw + x) * channels;
        if (bytes) {
          const dst = (dstY * fullWidth + dstX) * 4;
          target[dst] = buf[srcIdx]!;
          target[dst + 1] = buf[srcIdx + 1]!;
          target[dst + 2] = buf[srcIdx + 2]!;
          target[dst + 3] = 255;
        } else {
          const dst = (dstY * fullWidth + dstX) * 3;
          target[dst] = buf[srcIdx]!;
          target[dst + 1] = buf[srcIdx + 1]!;
          target[dst + 2] = buf[srcIdx + 2]!;
        }
      }
    }
    void fullHeight;
  }

  dispose(): void {
    const gl = this.gl;
    Object.values(this.programs).forEach((p) => gl.deleteProgram(p));
    gl.deleteVertexArray(this.vao);
  }
}
