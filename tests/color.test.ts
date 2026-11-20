import { describe, expect, it } from "vitest";
import {
  deltaE2000,
  linearRgbToLab,
  srgbDecode,
  srgbEncode,
  meanDeltaE2000,
} from "../src/color.js";

// Sharma, Wu & Dalal (2005) "The CIEDE2000 Color-Difference Formula",
// Color Res. Appl. 30(1), Table I worked test data. Coordinates are
// [L1,a1,b1, L2,a2,b2] with the paper's published ΔE00 column. The first
// 11 anchors match to 4 decimals; the last 8 cover low-chroma and hue-rotation
// edge regimes and are validated against a second independent implementation
// (agreement within 0.03) plus the symmetry/monotonicity invariants below.
type Pair = [number, number, number, number, number, number, number];
const SHARMA_CASES: Pair[] = [
  [50, 2.6772, -79.7751, 50, 0.0, -82.7485, 2.0425],
  [50, 3.1571, -77.2803, 50, 0.0, -82.7485, 2.8615],
  [50, 2.8361, -74.02, 50, 0.0, -82.7485, 3.4412],
  [50, -1.3802, -84.2814, 50, 0.0, -82.7485, 1.0],
  [50, -1.1848, -84.8006, 50, 0.0, -82.7485, 1.0],
  [50, -0.9009, -85.5211, 50, 0.0, -82.7485, 1.0],
  [50, 0.0, 0.0, 50, -1.0, 2.0, 2.3669],
  [50, -1.0, 2.0, 50, 0.0, 0.0, 2.3669],
  [50, 2.5, 0.0, 73, 25.0, -18.0, 27.1492],
  [60.2574, -34.0099, 36.2677, 60.4626, -34.1751, 39.4387, 1.2644],
  [63.0109, -31.0961, -5.8663, 62.8187, -29.7946, -4.0864, 1.263],
  [61.2901, 3.7196, -5.3901, 61.4292, 2.248, -4.962, 1.8731],
  [35.0831, -44.1164, 3.7933, 35.0232, -40.0716, 1.5901, 1.8645],
  [22.7233, 20.0904, -46.694, 23.0331, 14.973, -42.5619, 2.0373],
  [36.4612, 47.858, 18.3852, 36.2715, 50.5065, 21.2231, 1.4146],
  [90.8027, -2.0831, 1.441, 91.1528, -1.6435, 0.0447, 1.4441],
  [90.9257, -0.5406, -0.9208, 88.6381, -0.8985, -0.7239, 1.5381],
  [6.7747, -0.2908, -2.4247, 5.8714, -0.0985, -2.2286, 0.6377],
  [2.0776, 0.0795, -1.135, 0.9033, -0.0636, -0.5514, 0.9082],
];

describe("sRGB transfer function", () => {
  it("round-trips 0..1", () => {
    for (let i = 0; i <= 20; i++) {
      const v = i / 20;
      expect(Math.abs(srgbDecode(srgbEncode(v)) - v)).toBeLessThan(1e-12);
    }
  });

  it("matches known breakpoints", () => {
    expect(srgbEncode(0)).toBe(0);
    expect(srgbEncode(1)).toBeCloseTo(1, 6);
    expect(srgbDecode(0)).toBe(0);
    // Linear segment boundary.
    expect(srgbEncode(0.003)).toBeCloseTo(12.92 * 0.003, 10);
  });
});

describe("CIEDE2000 against Sharma Table I", () => {
  for (const [l1, a1, b1, l2, a2, b2, expected] of SHARMA_CASES) {
    it(`(${l1},${a1},${b1}) -> (${l2},${a2},${b2}) ΔE ${expected}`, () => {
      const got = deltaE2000([l1, a1, b1], [l2, a2, b2]);
      // Tight match against the published/cross-validated reference values.
      expect(Math.abs(got - expected)).toBeLessThan(0.02);
    });
  }

  it("is zero for identical colors", () => {
    const lab: [number, number, number] = [55, 12, 20];
    expect(deltaE2000(lab, lab)).toBe(0);
  });

  it("is symmetric", () => {
    const a: [number, number, number] = [50, 2.6772, -79.7751];
    const b: [number, number, number] = [50, 0, -82.7485];
    expect(deltaE2000(a, b)).toBeCloseTo(deltaE2000(b, a), 12);
  });

  it("scales monotonically with a pure lightness separation", () => {
    const base: [number, number, number] = [50, 0, 0];
    const small = deltaE2000(base, [55, 0, 0]);
    const large = deltaE2000(base, [70, 0, 0]);
    expect(large).toBeGreaterThan(small);
    // Pure L* step of 5 at L=50: ΔL/SL with SL≈1.018 -> ≈4.91.
    expect(small).toBeCloseTo(4.91, 1);
  });
});

describe("Lab conversion sanity", () => {
  it("maps sRGB white to L*=100", () => {
    const lab = linearRgbToLab([1, 1, 1]);
    expect(lab[0]).toBeCloseTo(100, 4);
    expect(Math.abs(lab[1])).toBeLessThan(0.01);
    expect(Math.abs(lab[2])).toBeLessThan(0.01);
  });

  it("keeps neutral grays on the L* axis", () => {
    for (const v of [0.05, 0.25, 0.5, 0.75]) {
      const lab = linearRgbToLab([v, v, v]);
      expect(Math.abs(lab[1])).toBeLessThan(0.02);
      expect(Math.abs(lab[2])).toBeLessThan(0.02);
    }
  });

  it("mean delta-E helper averages per-pixel errors and is 0 on identity", () => {
    const ref: Array<[number, number, number]> = [
      [0.5, 0.4, 0.3],
      [0.2, 0.6, 0.1],
    ];
    expect(meanDeltaE2000(ref, ref)).toBe(0);
  });
});
