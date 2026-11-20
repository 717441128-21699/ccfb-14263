/**
 * High-level restoration facade used by the demo, CLI and benchmarks.
 *
 * Two execution backends share the same parameters and acceptance checks:
 *   - CPU: pure TypeScript reference (also the fallback where WebGL2 is
 *     unavailable, e.g. some headless CI images).
 *   - GPU: WebGL2Restorer tiled pipeline.
 *
 * The blind path downsamples large images for DCP estimation to keep the
 * CPU pre-pass cheap, then restores at full resolution.
 */

import {
  type FloatImage,
  type RestoreParams,
  restoreBlind,
  restoreCalibrated,
  estimateDistance,
  depthOptics,
} from "./model.js";
import { applySkinConstraint } from "./skin.js";
import { toDisplayBytes } from "./tone.js";
import { WebGL2Restorer, type GpuRestoreResult } from "./webgl2.js";

export interface PipelineOptions {
  params: RestoreParams;
  applySkin?: boolean;
  maxSkinBlend?: number;
  /** Downsample factor for the blind DCP pre-pass on large images. */
  blindDownscale?: number;
}

export interface PipelineResult {
  restored: FloatImage;
  strobePower: number;
  distance: Float32Array;
}

function boxDownsample(img: FloatImage, factor: number): FloatImage {
  if (factor <= 1) return img;
  const w = Math.max(1, Math.floor(img.width / factor));
  const h = Math.max(1, Math.floor(img.height / factor));
  const data = new Float32Array(w * h * 3);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const sx = Math.min(img.width - 1, x * factor);
      const sy = Math.min(img.height - 1, y * factor);
      const src = (sy * img.width + sx) * 3;
      const dst = (y * w + x) * 3;
      data[dst] = img.data[src]!;
      data[dst + 1] = img.data[src + 1]!;
      data[dst + 2] = img.data[src + 2]!;
    }
  }
  return { width: w, height: h, data };
}

function upsampleDistance(
  small: Float32Array,
  sw: number,
  sh: number,
  fullW: number,
  fullH: number,
): Float32Array {
  const out = new Float32Array(fullW * fullH);
  for (let y = 0; y < fullH; y++) {
    const sy = Math.min(sh - 1, Math.floor((y * sh) / fullH));
    for (let x = 0; x < fullW; x++) {
      const sx = Math.min(sw - 1, Math.floor((x * sw) / fullW));
      out[y * fullW + x] = small[sy * sw + sx]!;
    }
  }
  return out;
}

/** CPU reference pipeline. */
export function restoreCpu(
  image: FloatImage,
  opts: PipelineOptions,
): PipelineResult {
  const applySkin = opts.applySkin ?? true;
  let params = opts.params;
  let distance = params.distance;
  let strobePower = params.strobe.power;

  if (!distance || params.estimateDistance) {
    // Estimate on a downscaled copy for speed, then upsample the distance.
    const factor =
      opts.blindDownscale ??
      (image.width > 1600 ? Math.ceil(image.width / 1024) : 1);
    const small = boxDownsample(image, factor);
    const { c, veil } = depthOptics(params.waterType, params.depthM);
    const smallDistance = estimateDistance(small, c, veil, 7, 0.95);
    distance =
      factor > 1
        ? upsampleDistance(smallDistance, small.width, small.height, image.width, image.height)
        : smallDistance;
    const blindParams: RestoreParams = { ...params, distance };
    const blind = restoreBlind(image, blindParams);
    strobePower = blind.strobePower;
    params = {
      ...blindParams,
      strobe: { ...params.strobe, power: strobePower },
    };
  }

  const calibrated = restoreCalibrated(image, { ...params, distance });
  const restored = applySkin
    ? applySkinConstraint(calibrated, opts.maxSkinBlend ?? 0.75).image
    : calibrated;

  return { restored, strobePower, distance: distance! };
}

/** GPU pipeline; requires a live WebGL2 context. */
export function restoreGpu(
  gl: WebGL2RenderingContext,
  image: FloatImage,
  opts: PipelineOptions & { exposure?: number; tileSize?: number },
): GpuRestoreResult {
  const restorer = new WebGL2Restorer(gl);
  try {
    return restorer.restore(image, {
      params: opts.params,
      applySkin: opts.applySkin ?? true,
      maxSkinBlend: opts.maxSkinBlend,
      encodeSrgb: true,
      exposure: opts.exposure ?? 1,
      tileSize: opts.tileSize,
    });
  } finally {
    restorer.dispose();
  }
}

export { toDisplayBytes };
