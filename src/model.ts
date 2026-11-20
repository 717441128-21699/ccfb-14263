/**
 * Jaffe–McGlamery underwater imaging model, forward and inverse.
 *
 * Per channel k, with object distance d(x), camera depth z, beam
 * transmission t = exp(-c d), downwelling ambient E(z) = exp(-Kd z) and
 * strobe geometry factor G(x, d):
 *
 *   M_k = ρ_k [ E_k(z)·t_k  +  G·t_k² ]  +  V_k·(1 − t_k)·(1 + η·G)
 *          \___ ambient ___/  \_ strobe _/     \____ backscatter veil ____/
 *
 * The strobe travels to the object AND back (co-located camera/strobe), so
 * its path attenuation is t² = exp(−2cd). The ambient path is one-way t.
 * Keeping strobe and ambient paths separate is what stops a single global
 * white balance from painting near faces red while far water stays cyan.
 *
 * Forward (synthesizer) and inverse (restorer) call the same per-pixel
 * functions, so a known scene round-trips to itself within floating error.
 */

import {
  type JerlovType,
  type WaterOptics,
  attenuationAtDepth,
  downwellingFactor,
  getWaterOptics,
} from "./water.js";
import {
  type StrobeConfig,
  DEFAULT_STROBE,
  strobeGeometry,
} from "./strobe.js";

export interface FloatImage {
  width: number;
  height: number;
  /** Packed RGB, stride 3, linear light, nominally 0..1. */
  data: Float32Array;
}

export interface RestoreParams {
  /** Camera depth below surface, meters. */
  depthM: number;
  /** Jerlov water type. */
  waterType: JerlovType;
  /** Strobe configuration. power = 0 disables the strobe path. */
  strobe: StrobeConfig;
  /**
   * Per-pixel object-camera distance in meters (length width*height).
   * In calibrated mode this is supplied (sonar/lens-focus/known scene).
   * In blind mode it is estimated; pass `estimateDistance: true`.
   */
  distance?: Float32Array;
  /** Set true to run DCP veil/distance + strobe-power fitting. */
  estimateDistance?: boolean;
  /** Strobe backscatter coupling η: fraction of cone geometry feeding veil. */
  strobeBackscatterCoupling?: number;
  /** DCP patch radius in pixels for the dark-channel min filter. */
  dcpPatchRadius?: number;
  /** DCP aerial-persistence strength ω. */
  dcpOmega?: number;
  /** Floor on the combined path denominator to avoid noise blow-up. */
  minPathGain?: number;
  /**
   * Maximum per-channel restoration gain 1/pathGain. Channels whose
   * transmission would demand more gain than this are dominated by noise;
   * clamping avoids amplified false color. Infinity (default) keeps the
   * calibrated inversion lossless; set a finite value (e.g. 12) for noisy
   * blind/deep-water restoration.
   */
  maxGain?: number;
  /**
   * Wiener regularization strength λ (0 = exact lossless inversion). Set
   * from the estimated sensor noise variance, e.g. 0.002–0.01; it shrinks
   * low-SNR (deep red) channels toward the gray prior to kill false reds.
   */
  regularization?: number;
  /**
   * Blind-mode only: clip recovered reflectance into a physical range.
   * Calibrated inversion is left lossless; blind distance error can divide a
   * tiny transmission and produce negatives/HDR spikes, so a soft physical
   * bound keeps the output usable. Default [0, 1.25].
   */
  reflectanceRange?: [number, number];
}

/** Strobe backscatter coupling default, calibrated against the synthesizer. */
export const DEFAULT_COUPLING = 0.55;

export interface DepthOptics {
  optics: WaterOptics;
  /** Effective beam attenuation at the camera depth. */
  c: [number, number, number];
  /** Downwelling ambient E_k(z) at the camera/object depth. */
  ambient: [number, number, number];
  veil: [number, number, number];
}

export function depthOptics(type: JerlovType, depthM: number): DepthOptics {
  const optics = getWaterOptics(type);
  const c = attenuationAtDepth(optics.attenuation, depthM);
  const ambient = downwellingFactor(c, depthM);
  return { optics, c, ambient, veil: optics.veilLinear };
}

/**
 * One-way beam transmission t_k = exp(-c_k d).
 */
export function transmission(
  c: [number, number, number],
  distanceM: number,
): [number, number, number] {
  return [
    Math.exp(-c[0] * distanceM),
    Math.exp(-c[1] * distanceM),
    Math.exp(-c[2] * distanceM),
  ];
}

/**
 * FORWARD model: degrade a clean linear-light scene into an underwater photo.
 *
 * @param scene    clean linear RGB reflectance image (the land photo)
 * @param distance per-pixel object distance, meters
 * @param params   depth, water, strobe
 * @returns synthesized underwater linear RGB image
 */
export function forwardSynthesize(
  scene: FloatImage,
  distance: Float32Array,
  params: RestoreParams,
): FloatImage {
  const { width, height, data } = scene;
  const { c, ambient, veil } = depthOptics(params.waterType, params.depthM);
  const eta = params.strobeBackscatterCoupling ?? DEFAULT_COUPLING;
  const out = new Float32Array(data.length);

  for (let y = 0; y < height; y++) {
    const ny = height === 1 ? 0 : 1 - (2 * y) / (height - 1);
    for (let x = 0; x < width; x++) {
      const nx = width === 1 ? 0 : (2 * x) / (width - 1) - 1;
      const i = y * width + x;
      const d = distance[i]!;
      const t = transmission(c, d);
      const G = strobeGeometry(nx, ny, d, params.strobe);
      const p = i * 3;

      for (let k = 0; k < 3; k++) {
        const rho = data[p + k]!;
        // Ambient one-way + strobe round-trip signal.
        const signal = rho * (ambient[k]! * t[k]! + G * t[k]! * t[k]!);
        // Backscatter veil: ambient column plus strobe-lit cone volume.
        const veilTerm = veil[k]! * (1 - t[k]!) * (1 + eta * G);
        out[p + k] = signal + veilTerm;
      }
    }
  }
  return { width, height, data: out };
}

/**
 * INVERT the model for one pixel once all physical factors are known.
 * Shared by the calibrated and blind paths and mirrored 1:1 in GLSL.
 */
export function invertPixel(
  m: [number, number, number],
  t: [number, number, number],
  ambient: [number, number, number],
  strobeG: number,
  veil: [number, number, number],
  eta: number,
  minPathGain: number,
  maxGain = Infinity,
  regularization = 0,
): [number, number, number] {
  const rho: [number, number, number] = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    const veilTerm = veil[k]! * (1 - t[k]!) * (1 + eta * strobeG);
    const physicalGain = ambient[k]! * t[k]! + strobeG * t[k]! * t[k]!;
    // Guards combined into one effective path gain:
    //   - absolute floor minPathGain prevents divide-by-near-zero;
    //   - 1/maxGain only bites when the channel demands > maxGain gain.
    const pathGain = Math.max(physicalGain, minPathGain, 1 / maxGain);
    const residual = m[k]! - veilTerm;
    // Gated Wiener regularization. Channels with healthy path gain
    // (>= HEALTHY_GAIN) invert exactly. Below the gate the noise-to-signal
    // ratio is high (e.g. deep red), so a Wiener shrink factor pulls the
    // reflectance toward the gray prior rather than amplifying noise into
    // false red. lambda=0 disables it and keeps the whole inversion exact.
    const HEALTHY_GAIN = 0.25;
    let rhoK = residual / pathGain;
    if (regularization > 0 && pathGain < HEALTHY_GAIN) {
      const wiener = (pathGain * pathGain) / (pathGain * pathGain + regularization);
      rhoK = (residual / pathGain) * wiener;
    }
    rho[k] = rhoK;
  }
  return rho;
}

/**
 * Calibrated inverse: distance map is known (dive metadata / focus / sonar),
 * so the inversion is direct and exact. This is the ΔE < 5 path.
 */
export function restoreCalibrated(
  underwater: FloatImage,
  params: RestoreParams,
): FloatImage {
  if (!params.distance) {
    throw new Error("restoreCalibrated requires params.distance");
  }
  const { width, height, data } = underwater;
  const distance = params.distance;
  const { c, ambient, veil } = depthOptics(params.waterType, params.depthM);
  const eta = params.strobeBackscatterCoupling ?? DEFAULT_COUPLING;
  const minPathGain = params.minPathGain ?? 1e-4;
  const maxGain = params.maxGain ?? Infinity;
  const regularization = params.regularization ?? 0;
  const out = new Float32Array(data.length);

  for (let y = 0; y < height; y++) {
    const ny = height === 1 ? 0 : 1 - (2 * y) / (height - 1);
    for (let x = 0; x < width; x++) {
      const nx = width === 1 ? 0 : (2 * x) / (width - 1) - 1;
      const i = y * width + x;
      const d = distance[i]!;
      const t = transmission(c, d);
      const G = strobeGeometry(nx, ny, d, params.strobe);
      const p = i * 3;
      const rho = invertPixel(
        [data[p]!, data[p + 1]!, data[p + 2]!],
        t,
        ambient,
        G,
        veil,
        eta,
        minPathGain,
        maxGain,
        regularization,
      );
      out[p] = rho[0];
      out[p + 1] = rho[1];
      out[p + 2] = rho[2];
    }
  }
  return { width, height, data: out };
}

// ---------------------------------------------------------------------------
// Blind estimation: dark-channel-prior veil -> transmission -> distance,
// then a gray-world strobe-power fit inside the strobe cone.
// ---------------------------------------------------------------------------

/** Min filter over a square patch, per channel-independent scalar map. */
function boxMin(map: Float32Array, width: number, height: number, radius: number): Float32Array {
  const out = new Float32Array(map.length);
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      let mn = Infinity;
      for (let yy = y0; yy <= y1; yy++) {
        const row = yy * width;
        for (let xx = x0; xx <= x1; xx++) {
          const v = map[row + xx]!;
          if (v < mn) mn = v;
        }
      }
      out[y * width + x] = mn;
    }
  }
  return out;
}

function boxMean(map: Float32Array, width: number, height: number, radius: number): Float32Array {
  // Integral image for an O(n) moving average.
  const integral = new Float64Array((width + 1) * (height + 1));
  for (let y = 1; y <= height; y++) {
    let rowSum = 0;
    for (let x = 1; x <= width; x++) {
      rowSum += map[(y - 1) * width + (x - 1)]!;
      integral[y * (width + 1) + x] =
        integral[(y - 1) * (width + 1) + x]! + rowSum;
    }
  }
  const out = new Float32Array(map.length);
  const W = width + 1;
  for (let y = 0; y < height; y++) {
    const y0 = Math.max(0, y - radius);
    const y1 = Math.min(height - 1, y + radius);
    for (let x = 0; x < width; x++) {
      const x0 = Math.max(0, x - radius);
      const x1 = Math.min(width - 1, x + radius);
      const sum =
        integral[(y1 + 1) * W + (x1 + 1)]! -
        integral[y0 * W + (x1 + 1)]! -
        integral[(y1 + 1) * W + x0]! +
        integral[y0 * W + x0]!;
      out[y * width + x] = sum / ((y1 - y0 + 1) * (x1 - x0 + 1));
    }
  }
  return out;
}

/**
 * Estimate a per-pixel distance map from the image itself.
 *
 * Dark channel prior: in a local patch some pixel has a low reflectance
 * channel, so the dark channel of the scene is ~0 and what remains there is
 * veil. Normalizing by the veil color recovers transmission 1−ω·(dark/V).
 * Distance follows from d = −ln(t)/c (green channel; red is too attenuated
 * and blue carries the strongest veil at depth).
 */
export function estimateDistance(
  img: FloatImage,
  c: [number, number, number],
  veil: [number, number, number],
  patchRadius: number,
  omega: number,
  range: { nearM?: number; farM?: number } = {},
): Float32Array {
  const { width, height, data } = img;
  const n = width * height;
  const normalized = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const p = i * 3;
    // min over channels of M_k / V_k
    normalized[i] = Math.min(
      data[p]! / veil[0],
      data[p + 1]! / veil[1],
      data[p + 2]! / veil[2],
    );
  }
  const dark = boxMin(normalized, width, height, patchRadius);
  // Soft refinement: average the patch transmission to reduce block edges.
  const tMap = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    tMap[i] = Math.min(1, Math.max(0.02, 1 - omega * dark[i]!));
  }
  const tSoft = boxMean(tMap, width, height, patchRadius);
  const raw = new Float32Array(n);
  const cGreen = c[1];
  for (let i = 0; i < n; i++) {
    const t = Math.min(0.985, Math.max(0.02, tSoft[i]!));
    raw[i] = -Math.log(t) / cGreen;
  }

  // DCP yields a *relative* distance field whose absolute scale is unknown
  // (it depends on scene content, not just optics). Anchor the scale by
  // mapping the estimated minimum to the expected near working distance and
  // clipping to a physical far bound. A diver's closest subject is rarely
  // under 0.4 m and usable strobe reach ends around 8 m.
  const nearM = range.nearM ?? 0.6;
  const farM = range.farM ?? 8;
  let rawMin = Infinity;
  for (let i = 0; i < n; i++) rawMin = Math.min(rawMin, raw[i]!);
  const distance = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    // Preserve the relative structure: offset so min -> nearM.
    const d = nearM + (raw[i]! - rawMin);
    distance[i] = Math.min(farM, Math.max(nearM, d));
  }
  return distance;
}

/**
 * Fit the strobe power multiplier inside the cone by a gray-world target.
 * Geometry is evaluated at power 1 (G0); the fitted p scales the strobe
 * term. Bisection on the mean restored luminance of high-gain cone pixels.
 */
export function fitStrobePower(
  img: FloatImage,
  distance: Float32Array,
  params: RestoreParams,
  ambient: [number, number, number],
  c: [number, number, number],
  veil: [number, number, number],
  eta: number,
  minPathGain: number,
  maxGain: number,
  regularization: number,
): number {
  const { width, height, data } = img;
  const unitStrobe: StrobeConfig = { ...params.strobe, power: 1 };
  // Cone pixels with strong geometry and near enough to be strobe-lit.
  const samples: Array<{ i: number; G0: number; t: [number, number, number] }> = [];
  for (let y = 0; y < height; y++) {
    const ny = height === 1 ? 0 : 1 - (2 * y) / (height - 1);
    for (let x = 0; x < width; x++) {
      const nx = width === 1 ? 0 : (2 * x) / (width - 1) - 1;
      const i = y * width + x;
      const d = distance[i]!;
      const G0 = strobeGeometry(nx, ny, d, unitStrobe);
      if (G0 > 0.6 && d < 3.0) {
        samples.push({ i, G0, t: transmission(c, d) });
      }
    }
  }
  if (samples.length < 16) return params.strobe.power;

  // Target mean linear luminance of restored cone reflectance.
  const TARGET_LUMA = 0.32;
  const meanRestoredLuma = (p: number): number => {
    let sum = 0;
    for (const s of samples) {
      const q = s.i * 3;
      const G = p * s.G0;
      const rho = invertPixel(
        [data[q]!, data[q + 1]!, data[q + 2]!],
        s.t,
        ambient,
        G,
        veil,
        eta,
        minPathGain,
        maxGain,
        regularization,
      );
      // BT.709 linear luma.
      sum += 0.2126 * rho[0] + 0.7152 * rho[1] + 0.0722 * rho[2];
    }
    return sum / samples.length;
  };

  let lo = 0;
  let hi = 4;
  // Restored luma decreases as strobe power rises (denominator grows).
  for (let iter = 0; iter < 28; iter++) {
    const mid = (lo + hi) / 2;
    if (meanRestoredLuma(mid) > TARGET_LUMA) lo = mid;
    else hi = mid;
  }
  return (lo + hi) / 2;
}

/**
 * Choose a Wiener regularization strength from the estimated red-channel
 * health. When red transmission is high (clear/shallow water) little
 * shrinkage is needed; when the red path is weak (deep/turbid water) a
 * stronger prior suppresses amplified noise. Returns 0 when the red channel
 * is healthy enough to invert exactly.
 *
 * @param meanRedTransmission average exp(-c_red d) over the frame
 */
export function autoRegularization(meanRedTransmission: number): number {
  // Empirically calibrated against the noise robustness sweep: λ ≈ 0.001–0.002
  // across the reachable envelope, ramping up only as red transmission falls.
  if (meanRedTransmission >= 0.5) return 0;
  const weakness = (0.5 - meanRedTransmission) / 0.5; // 0..1
  return Math.min(0.002, 0.0004 + 0.0016 * weakness);
}

/**
 * Blind inverse: estimate distance from the image, fit strobe power, then
 * run the identical per-pixel inversion as the calibrated path.
 */
export function restoreBlind(
  underwater: FloatImage,
  params: RestoreParams,
): { restored: FloatImage; distance: Float32Array; strobePower: number } {
  const { c, ambient, veil } = depthOptics(params.waterType, params.depthM);
  const eta = params.strobeBackscatterCoupling ?? DEFAULT_COUPLING;
  const minPathGain = params.minPathGain ?? 1e-4;
  const maxGain = params.maxGain ?? Infinity;
  const regularization = params.regularization ?? 0;
  const radius = params.dcpPatchRadius ?? 7;
  const omega = params.dcpOmega ?? 0.95;

  const distance =
    params.distance ??
    estimateDistance(underwater, c, veil, radius, omega);

  const strobePower = fitStrobePower(
    underwater,
    distance,
    params,
    ambient,
    c,
    veil,
    eta,
    minPathGain,
    maxGain,
    regularization,
  );

  const effective: RestoreParams = {
    ...params,
    strobe: { ...params.strobe, power: strobePower },
    distance,
  };
  const inverted = restoreCalibrated(underwater, effective);

  // Blind-mode physical reflectance bound (see RestoreParams.reflectanceRange).
  const [lo, hi] = params.reflectanceRange ?? [0, 1.25];
  const data = inverted.data;
  if (lo > -Infinity || hi < Infinity) {
    for (let i = 0; i < data.length; i++) {
      data[i] = Math.min(hi, Math.max(lo, data[i]!));
    }
  }
  return { restored: inverted, distance, strobePower };
}
