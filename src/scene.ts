/**
 * Procedural LAND scene generator and underwater test-image synthesizer.
 *
 * The acceptance requirement is that test images be synthesized from real
 * land photos through the physical water attenuation model. Two land sources
 * are supported:
 *   1. A supplied land photo (decodeSrgbBytes -> forwardSynthesize).
 *   2. A deterministic procedural scene containing the ColorChecker chart,
 *      skin-tone discs standing in for diver faces, coral-like structures
 *      and a blue-sky gradient — so the full pipeline is testable with no
 *      external image files.
 *
 * Every synthetic underwater image carries its ground-truth distance map,
 * depth and water parameters alongside it, which is what makes the ΔE and
 * skin-gamut acceptance checks possible.
 */

import { COLOR_CHECKER } from "./colorchart.js";
import { srgbDecode } from "./color.js";
import {
  type FloatImage,
  type RestoreParams,
  forwardSynthesize,
} from "./model.js";
import { DEFAULT_STROBE } from "./strobe.js";
import type { JerlovType } from "./water.js";

/** Deterministic value-noise PRNG (mulberry32) so fixtures are reproducible. */
export function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

export interface LandScene {
  image: FloatImage;
  /** Per-pixel object-camera distance, meters. */
  distance: Float32Array;
  /** Label of the dominant object at each pixel (for region assertions). */
  labels: Uint8Array;
}

export const LABEL = {
  BACKGROUND: 0,
  CHART: 1,
  SKIN: 2,
  CORAL: 3,
  SAND: 4,
} as const;

/**
 * Build a procedural land scene.
 *
 * Layout (normalized coords): ColorChecker board left-of-center at 1.5–2.5 m,
 * two skin-tone discs at 1.2–1.8 m (near "diver faces"), branching coral on
 * the right at 1.0–2.2 m, sandy floor in the lower third, open water/sky
 * gradient behind. Distance increases toward the top/back to ~6 m.
 */
export function generateLandScene(width: number, height: number, seed = 7): LandScene {
  const rng = makeRng(seed);
  const data = new Float32Array(width * height * 3);
  const distance = new Float32Array(width * height);
  const labels = new Uint8Array(width * height);

  // Pre-generate coral branch centers and skin disc parameters.
  const corals = Array.from({ length: 9 }, () => ({
    x: 0.62 + rng() * 0.32,
    y: 0.3 + rng() * 0.45,
    r: 0.02 + rng() * 0.035,
    warm: rng() > 0.4,
  }));
  const faces = [
    { x: 0.27, y: 0.62, r: 0.085, patch: 0 }, // dark skin patch
    { x: 0.4, y: 0.7, r: 0.075, patch: 1 }, // light skin patch
  ];

  // Chart rect in normalized coords.
  const chart = { x0: 0.06, y0: 0.18, x1: 0.34, y1: 0.52 };

  const chartLinear = COLOR_CHECKER.map((p) =>
    p.srgb.map((v) => srgbDecode(v / 255)),
  ) as Array<[number, number, number]>;

  for (let y = 0; y < height; y++) {
    const ny = 1 - y / (height - 1); // 0 bottom -> 1 top
    for (let x = 0; x < width; x++) {
      const nx = x / (width - 1);
      const i = y * width + x;
      const p = i * 3;

      // Default: sky/background gradient (pale blue top to sandy haze).
      const skyT = ny;
      let r = 0.55 + 0.25 * skyT;
      let g = 0.66 + 0.2 * skyT;
      let b = 0.78 + 0.18 * skyT;
      let d = 3.5 + 2.5 * ny; // farther toward the top
      let label: number = LABEL.BACKGROUND;

      // Sandy floor lower third.
      if (ny < 0.28) {
        const grain = (rng() - 0.5) * 0.06;
        r = 0.76 + grain;
        g = 0.7 + grain;
        b = 0.52 + grain * 0.6;
        d = 2.2 + (0.28 - ny) * 2.0;
        label = LABEL.SAND;
      }

      // Coral branches (drawn before chart so the board stays clean).
      for (const c of corals) {
        const dist = Math.hypot(nx - c.x, ny - c.y);
        if (dist < c.r) {
          const shade = 1 - 0.3 * (dist / c.r);
          if (c.warm) {
            r = 0.85 * shade;
            g = 0.32 * shade;
            b = 0.22 * shade;
          } else {
            r = 0.95 * shade;
            g = 0.55 * shade;
            b = 0.3 * shade;
          }
          d = 1.2 + c.y * 1.2;
          label = LABEL.CORAL;
        }
      }

      // ColorChecker board (6 cols x 4 rows).
      if (nx >= chart.x0 && nx <= chart.x1 && ny >= chart.y0 && ny <= chart.y1) {
        const col = Math.min(5, Math.floor(((nx - chart.x0) / (chart.x1 - chart.x0)) * 6));
        const row = Math.min(3, Math.floor(((ny - chart.y0) / (chart.y1 - chart.y0)) * 4));
        const color = chartLinear[row * 6 + col]!;
        r = color[0];
        g = color[1];
        b = color[2];
        // Flat board facing the camera; slight depth tilt.
        d = 1.6 + (nx - chart.x0) * 0.4;
        label = LABEL.CHART;
      }

      // Skin discs (diver faces), drawn last so nothing occludes them.
      for (const f of faces) {
        const dist = Math.hypot(nx - f.x, ny - f.y);
        if (dist < f.r) {
          const color = chartLinear[f.patch]!;
          // Subtle radial shading like a sphere.
          const shade = 1 - 0.25 * (dist / f.r);
          r = color[0] * shade;
          g = color[1] * shade;
          b = color[2] * shade;
          d = 1.3 + (f.r - dist) * 0.4;
          label = LABEL.SKIN;
        }
      }

      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
      distance[i] = d;
      labels[i] = label;
    }
  }

  return { image: { width, height, data }, distance, labels };
}

export interface SyntheticUnderwater {
  /** The degraded underwater photo (linear light). */
  underwater: FloatImage;
  /** The original clean land scene. */
  land: LandScene;
  /** Exact parameters used, for round-trip restoration. */
  params: RestoreParams;
}

export interface SynthesizeOptions {
  width: number;
  height: number;
  depthM: number;
  waterType: JerlovType;
  strobePower?: number;
  seed?: number;
}

/**
 * Full fixture: procedural land scene -> Jaffe-McGlamery underwater photo.
 */
export function synthesizeUnderwater(opts: SynthesizeOptions): SyntheticUnderwater {
  const land = generateLandScene(opts.width, opts.height, opts.seed ?? 7);
  const params: RestoreParams = {
    depthM: opts.depthM,
    waterType: opts.waterType,
    strobe: { ...DEFAULT_STROBE, power: opts.strobePower ?? 1 },
    distance: land.distance,
  };
  const underwater = forwardSynthesize(land.image, land.distance, params);
  return { underwater, land, params };
}

/**
 * Synthesize from a real land photo instead of the procedural scene.
 * A distance map must be supplied (or a simple gradient is assumed).
 */
export function synthesizeFromLandPhoto(
  land: FloatImage,
  distance: Float32Array | null,
  params: RestoreParams,
): SyntheticUnderwater {
  const d =
    distance ??
    (() => {
      const n = land.width * land.height;
      const out = new Float32Array(n);
      for (let y = 0; y < land.height; y++) {
        const ny = 1 - y / (land.height - 1);
        for (let x = 0; x < land.width; x++) {
          out[y * land.width + x] = 1.5 + 3.5 * ny;
        }
      }
      return out;
    })();
  const withDistance: RestoreParams = { ...params, distance: d };
  const underwater = forwardSynthesize(land, d, withDistance);
  return {
    underwater,
    land: { image: land, distance: d, labels: new Uint8Array(land.width * land.height) },
    params: withDistance,
  };
}
