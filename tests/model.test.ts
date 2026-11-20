import { describe, expect, it } from "vitest";
import {
  type FloatImage,
  type RestoreParams,
  forwardSynthesize,
  invertPixel,
  restoreCalibrated,
  restoreBlind,
  transmission,
  depthOptics,
} from "../src/model.js";
import { DEFAULT_STROBE, coneGain, strobeGeometry } from "../src/strobe.js";
import { DEFAULT_COUPLING } from "../src/model.js";

function makeImage(width: number, height: number, fn: (x: number, y: number) => [number, number, number]): FloatImage {
  const data = new Float32Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      const [r, g, b] = fn(x, y);
      const p = (y * width + x) * 3;
      data[p] = r;
      data[p + 1] = g;
      data[p + 2] = b;
    }
  }
  return { width, height, data };
}

const baseParams: RestoreParams = {
  depthM: 10,
  waterType: "IB",
  strobe: { ...DEFAULT_STROBE },
};

describe("strobe geometry", () => {
  it("is full gain on the cone axis and zero well outside it", () => {
    expect(coneGain(DEFAULT_STROBE.centerX, 0, DEFAULT_STROBE)).toBeCloseTo(1, 6);
    const far = coneGain(0.9, 0.9, DEFAULT_STROBE);
    expect(far).toBe(0);
  });

  it("falls off with distance via inverse-square", () => {
    const near = strobeGeometry(0.06, 0, 1, DEFAULT_STROBE);
    const far = strobeGeometry(0.06, 0, 4, DEFAULT_STROBE);
    expect(near).toBeGreaterThan(far);
  });

  it("power zero disables the strobe path", () => {
    expect(strobeGeometry(0, 0, 1, { ...DEFAULT_STROBE, power: 0 })).toBe(0);
  });
});

describe("forward/inverse round trip (calibrated)", () => {
  it("inverts a single pixel exactly", () => {
    const { c, ambient, veil } = depthOptics("IB", 10);
    const d = 2.2;
    const t = transmission(c, d);
    const G = strobeGeometry(0, 0, d, DEFAULT_STROBE);
    const rho: [number, number, number] = [0.62, 0.41, 0.3];
    const m: [number, number, number] = [0, 0, 0];
    for (let k = 0; k < 3; k++) {
      m[k] =
        rho[k]! * (ambient[k]! * t[k]! + G * t[k]! * t[k]!) +
        veil[k]! * (1 - t[k]!) * (1 + DEFAULT_COUPLING * G);
    }
    const back = invertPixel(m, t, ambient, G, veil, DEFAULT_COUPLING, 1e-4);
    for (let k = 0; k < 3; k++) {
      expect(Math.abs(back[k]! - rho[k]!)).toBeLessThan(1e-10);
    }
  });

  it("restores a full synthetic image within float tolerance", () => {
    const scene = makeImage(64, 48, (x, y) => [
      0.2 + 0.6 * (x / 64),
      0.3 + 0.4 * (y / 48),
      0.5,
    ]);
    const distance = new Float32Array(64 * 48).map((_, i) => 1 + (i % 32) / 8);
    const params: RestoreParams = { ...baseParams, distance };
    const uw = forwardSynthesize(scene, distance, params);
    const restored = restoreCalibrated(uw, params);

    let maxErr = 0;
    for (let i = 0; i < scene.data.length; i++) {
      maxErr = Math.max(maxErr, Math.abs(restored.data[i]! - scene.data[i]!));
    }
    expect(maxErr).toBeLessThan(1e-4);
  });

  it("ambient-only and strobe-only paths are both modeled", () => {
    const scene = makeImage(8, 8, () => [0.5, 0.5, 0.5]);
    const distance = new Float32Array(64).fill(2);
    const ambientOnly: RestoreParams = {
      ...baseParams,
      strobe: { ...DEFAULT_STROBE, power: 0 },
      distance,
    };
    const withStrobe: RestoreParams = { ...baseParams, distance };
    const a = forwardSynthesize(scene, distance, ambientOnly);
    const b = forwardSynthesize(scene, distance, withStrobe);
    // Strobe adds signal inside the cone, so total radiance differs.
    let diff = 0;
    for (let i = 0; i < a.data.length; i++) diff += Math.abs(b.data[i]! - a.data[i]!);
    expect(diff).toBeGreaterThan(0);
    // And both still invert exactly.
    const ra = restoreCalibrated(a, ambientOnly);
    const rb = restoreCalibrated(b, withStrobe);
    for (let i = 0; i < scene.data.length; i++) {
      expect(Math.abs(ra.data[i]! - 0.5)).toBeLessThan(1e-5);
      expect(Math.abs(rb.data[i]! - 0.5)).toBeLessThan(1e-5);
    }
  });
});

describe("blind estimation", () => {
  it("recovers a distance map correlated with ground truth", () => {
    // Realistic scene WITH dark-channel structure: a mix of dark objects
    // (low reflectance in at least one channel) and mid-tone background.
    // Classic DCP assumes some dark pixels exist locally.
    const scene = makeImage(96, 72, (x, y) => {
      const darkBlob = (x - 24) ** 2 + (y - 36) ** 2 < 100;
      const darkBlob2 = (x - 70) ** 2 + (y - 30) ** 2 < 120;
      if (darkBlob || darkBlob2) return [0.05, 0.06, 0.07];
      const shade = 0.25 + 0.3 * ((x + y) % 9) / 9;
      return [shade, shade * 0.9, shade * 0.75];
    });
    // Two depth bands: near left half, far right half.
    const distance = new Float32Array(96 * 72);
    for (let y = 0; y < 72; y++) {
      for (let x = 0; x < 96; x++) {
        distance[y * 96 + x] = x < 48 ? 1.5 : 4.5;
      }
    }
    const params: RestoreParams = { ...baseParams, distance, strobe: { ...DEFAULT_STROBE, power: 0 } };
    const uw = forwardSynthesize(scene, distance, params);
    const { restored, distance: est } = restoreBlind(uw, {
      ...params,
      distance: undefined,
      estimateDistance: true,
    });

    // Estimated far band should be larger than near band on average — DCP
    // gives the correct relative ordering even if absolute scale is anchored.
    let nearMean = 0;
    let farMean = 0;
    for (let y = 0; y < 72; y++) {
      for (let x = 0; x < 96; x++) {
        const d = est[y * 96 + x]!;
        if (x < 48) nearMean += d;
        else farMean += d;
      }
    }
    nearMean /= 72 * 48;
    farMean /= 72 * 48;
    expect(farMean).toBeGreaterThan(nearMean);

    // Distances are anchored into the physical working range.
    for (let i = 0; i < est.length; i++) {
      expect(est[i]!).toBeGreaterThanOrEqual(0.6);
      expect(est[i]!).toBeLessThanOrEqual(8);
    }

    // Blind-mode reflectance clipping keeps the output finite and physical.
    for (let i = 0; i < restored.data.length; i++) {
      expect(Number.isFinite(restored.data[i]!)).toBe(true);
      expect(restored.data[i]!).toBeGreaterThanOrEqual(0);
      expect(restored.data[i]!).toBeLessThanOrEqual(1.25);
    }
  });
});
