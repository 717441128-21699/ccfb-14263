/**
 * Robustness acceptance: the ΔE < 5 requirement must hold with realistic
 * camera sensor noise and distance-map estimation error, not only on a
 * noise-free exact round trip. These tests characterize the reachable
 * operating envelope and verify the Wiener regularization keeps skin/chart
 * colors natural instead of amplifying deep-water red noise into false reds.
 */
import { describe, expect, it } from "vitest";
import { colorCheckerLinear } from "../src/colorchart.js";
import { deltaE2000, linearRgbToLab, type RGB } from "../src/color.js";
import type { JerlovType } from "../src/water.js";
import {
  autoRegularization,
  forwardSynthesize,
  restoreCalibrated,
  transmission,
  depthOptics,
  type FloatImage,
  type RestoreParams,
} from "../src/model.js";
import { DEFAULT_STROBE } from "../src/strobe.js";

function displayMap(c: RGB): RGB {
  return c.map((v) => Math.max(0, v) / (1 + Math.max(0, v))) as RGB;
}

function makeRng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/**
 * Mean ColorChecker ΔE after degrade -> add noise -> restore with a slightly
 * wrong distance map and the auto-selected regularization.
 */
function noisyChartMeanDeltaE(
  depthM: number,
  type: JerlovType,
  sigma: number,
  distanceErrPct: number,
): number {
  const rng = makeRng(42);
  const { c } = depthOptics(type, depthM);
  const distanceM = 2;
  const meanRedT = transmission(c, distanceM)[0]!;
  const reg = autoRegularization(meanRedT);
  const errors: number[] = [];

  for (const rgb of colorCheckerLinear()) {
    const scene: FloatImage = { width: 1, height: 1, data: Float32Array.from(rgb) };
    const cleanParams: RestoreParams = {
      depthM,
      waterType: type,
      strobe: { ...DEFAULT_STROBE, power: 1 },
      distance: new Float32Array([distanceM]),
    };
    const uw = forwardSynthesize(scene, new Float32Array([distanceM]), cleanParams);
    const noisy = new Float32Array(3);
    for (let i = 0; i < 3; i++) {
      const g = (rng() + rng() + rng() - 1.5) * 1.4142;
      noisy[i] = Math.max(0, uw.data[i]! + sigma * g);
    }
    const restoreParams: RestoreParams = {
      ...cleanParams,
      distance: new Float32Array([distanceM * (1 + distanceErrPct)]),
      regularization: reg,
    };
    const out = restoreCalibrated({ ...uw, data: noisy }, restoreParams);
    const got: RGB = [out.data[0]!, out.data[1]!, out.data[2]!];
    errors.push(
      deltaE2000(linearRgbToLab(displayMap(rgb)), linearRgbToLab(displayMap(got))),
    );
  }
  return errors.reduce((a, b) => a + b, 0) / errors.length;
}

describe("ROBUSTNESS: ΔE < 5 under realistic noise + distance error", () => {
  // Realistic consumer camera: linear-light noise σ≈0.004, ±5% distance
  // estimation error. Envelope spans clear shallow to moderately deep water.
  const envelope: Array<{ type: JerlovType; depth: number }> = [
    { type: "I", depth: 5 },
    { type: "IA", depth: 8 },
    { type: "IB", depth: 10 },
    { type: "II", depth: 12 },
    { type: "III", depth: 15 },
  ];

  for (const { type, depth } of envelope) {
    it(`${type} @ ${depth} m keeps mean chart ΔE under 5`, () => {
      const de = noisyChartMeanDeltaE(depth, type, 0.004, 0.05);
      expect(de).toBeLessThan(5);
    });
  }

  it("regularization never fires on healthy red channels (exact inversion preserved)", () => {
    // High red transmission -> autoRegularization returns 0.
    expect(autoRegularization(0.8)).toBe(0);
    expect(autoRegularization(0.5)).toBe(0);
  });

  it("regularization grows as red transmission weakens", () => {
    const shallow = autoRegularization(0.4);
    const deep = autoRegularization(0.05);
    expect(deep).toBeGreaterThan(shallow);
    expect(deep).toBeLessThanOrEqual(0.002);
  });

  it("noise-free exact round trip stays lossless when regularization is off", () => {
    const rgb: RGB = [0.5, 0.4, 0.3];
    const scene: FloatImage = { width: 1, height: 1, data: Float32Array.from(rgb) };
    const params: RestoreParams = {
      depthM: 10,
      waterType: "IB",
      strobe: { ...DEFAULT_STROBE, power: 1 },
      distance: new Float32Array([2]),
      regularization: 0,
    };
    const uw = forwardSynthesize(scene, new Float32Array([2]), params);
    const out = restoreCalibrated(uw, params);
    for (let k = 0; k < 3; k++) {
      // Float32 single precision gives ~1e-7 round-trip tolerance.
      expect(Math.abs(out.data[k]! - rgb[k]!)).toBeLessThan(1e-6);
    }
  });
});
