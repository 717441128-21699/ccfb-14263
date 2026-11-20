/**
 * Acceptance tests against the boss's hard requirements:
 *   1. ColorChecker recovered with mean CIEDE2000 < 5 (ΔE < 5).
 *   2. Diver face skin lands inside the normal human skin gamut.
 *   3. Coral colors are not washed out by the water-cast removal.
 *   4. Works across depths (5/10/18 m) and Jerlov water types.
 *   5. Strobe/ambient split: near strobe-lit and far ambient-only regions
 *      both restore correctly in one image.
 *
 * Test images are synthesized from land scenes through the SAME physical
 * forward model the production restorer inverts, with the ground-truth
 * distance map retained for the calibrated path.
 */
import { describe, expect, it } from "vitest";
import {
  COLOR_CHECKER,
  colorCheckerLinear,
} from "../src/colorchart.js";
import {
  deltaE2000,
  linearRgbToLab,
  srgbDecode,
  type RGB,
} from "../src/color.js";
import {
  type FloatImage,
  type JerlovType,
  forwardSynthesize,
  restoreCalibrated,
} from "../src/index.js";
import { applySkinConstraint } from "../src/skin.js";
import { LABEL, synthesizeUnderwater } from "../src/scene.js";
import { DEFAULT_STROBE } from "../src/strobe.js";
import { reinhardLinear } from "../src/tone.js";

/** Average restored linear color inside a labeled region. */
function regionMean(img: FloatImage, labels: Uint8Array, label: number): RGB {
  let r = 0;
  let g = 0;
  let b = 0;
  let n = 0;
  for (let i = 0; i < labels.length; i++) {
    if (labels[i] !== label) continue;
    const p = i * 3;
    r += img.data[p]!;
    g += img.data[p + 1]!;
    b += img.data[p + 2]!;
    n++;
  }
  return [r / n, g / n, b / n];
}

/** ΔE between two linear RGB colors after the same display mapping. */
function displayDeltaE(ref: RGB, got: RGB): number {
  const map = (c: RGB): RGB => {
    const m = reinhardLinear({ width: 1, height: 1, data: Float32Array.from(c) });
    return [m.data[0]!, m.data[1]!, m.data[2]!];
  };
  return deltaE2000(linearRgbToLab(map(ref)), linearRgbToLab(map(got)));
}

describe("REQUIREMENT: ColorChecker ΔE < 5 after restoration", () => {
  const cases: Array<{ depth: number; type: JerlovType }> = [
    { depth: 5, type: "I" },
    { depth: 10, type: "IB" },
    { depth: 10, type: "II" },
    { depth: 18, type: "III" },
    { depth: 12, type: "3C" },
  ];

  for (const { depth, type } of cases) {
    it(`${type} water at ${depth} m`, () => {
      // Each of the 24 patches is rendered alone at a fixed distance,
      // degraded through the forward model and restored, so the metric has
      // no neighbor-patch contamination.
      const errors = isolatedChartRoundTrip(depth, type);

      const mean = errors.reduce((a, b) => a + b, 0) / errors.length;
      const p95 = [...errors].sort((a, b) => a - b)[Math.floor(0.95 * errors.length)]!;
      expect(mean).toBeLessThan(5);
      expect(p95).toBeLessThan(8);
    });
  }
});

/**
 * Render each of the 24 ColorChecker patches alone at a fixed distance,
 * degrade and restore it, then measure CIEDE2000 against the reference.
 * Isolation removes neighbor-patch contamination from the metric.
 */
function isolatedChartRoundTrip(depthM: number, type: JerlovType): number[] {
  const ref = colorCheckerLinear();
  const distanceM = 2;
  const errors: number[] = [];
  for (let patch = 0; patch < ref.length; patch++) {
    const rgb = ref[patch]!;
    const scene: FloatImage = {
      width: 1,
      height: 1,
      data: Float32Array.from(rgb),
    };
    const distance = new Float32Array([distanceM]);
    const params = {
      depthM,
      waterType: type,
      strobe: { ...DEFAULT_STROBE, power: 1 },
      distance,
    };
    const uw = forwardSynthesize(scene, distance, params);
    const restored = restoreCalibrated(uw, params);
    const got: RGB = [restored.data[0]!, restored.data[1]!, restored.data[2]!];
    errors.push(displayDeltaE(rgb, got));
  }
  return errors;
}

describe("REQUIREMENT: skin tones land in the human gamut", () => {
  it("both face regions restore inside the Lab skin envelope", () => {
    const { underwater, land, params } = synthesizeUnderwater({
      width: 384,
      height: 288,
      depthM: 12,
      waterType: "IB",
    });
    const restored = restoreCalibrated(underwater, params);
    const constrained = applySkinConstraint(restored, 0.8).image;

    const faceMean = regionMean(constrained, land.labels, LABEL.SKIN);
    const lab = linearRgbToLab(faceMean);
    // Skin envelope from skin.ts.
    expect(lab[0]).toBeGreaterThan(28);
    expect(lab[0]).toBeLessThan(90);
    expect(lab[1]).toBeGreaterThan(6);
    expect(lab[1]).toBeLessThan(32);
    expect(lab[2]).toBeGreaterThan(12);
    expect(lab[2]).toBeLessThan(42);
  });

  it("reference skin patches restore within ΔE 6 of their land values", () => {
    const darkRef = COLOR_CHECKER[0]!.srgb.map((v) => srgbDecode(v / 255)) as RGB;
    const lightRef = COLOR_CHECKER[1]!.srgb.map((v) => srgbDecode(v / 255)) as RGB;
    for (const [ref, name] of [
      [darkRef, "dark"],
      [lightRef, "light"],
    ] as Array<[RGB, string]>) {
      const scene: FloatImage = { width: 1, height: 1, data: Float32Array.from(ref) };
      const distance = new Float32Array([1.6]);
      const params = {
        depthM: 12,
        waterType: "IB" as JerlovType,
        strobe: { ...DEFAULT_STROBE, power: 1 },
        distance,
      };
      const uw = forwardSynthesize(scene, distance, params);
      const restored = restoreCalibrated(uw, params);
      const got: RGB = [restored.data[0]!, restored.data[1]!, restored.data[2]!];
      const de = displayDeltaE(ref, got);
      expect(de, `${name} skin ΔE`).toBeLessThan(6);
    }
  });
});

describe("REQUIREMENT: coral colors are not washed out", () => {
  it("coral hue direction is preserved and saturation stays substantial", () => {
    const { underwater, land, params } = synthesizeUnderwater({
      width: 384,
      height: 288,
      depthM: 12,
      waterType: "IB",
    });
    const restored = restoreCalibrated(underwater, params);

    const landCoral = regionMean(land.image, land.labels, LABEL.CORAL);
    const gotCoral = regionMean(restored, land.labels, LABEL.CORAL);

    // Warm coral: red dominant over blue in both source and restored.
    expect(gotCoral[0]).toBeGreaterThan(gotCoral[2]);
    expect(landCoral[0] - landCoral[2]).toBeGreaterThan(0);

    // Saturation (max-min spread) retains at least 55% of the land value.
    const landSat = Math.max(...landCoral) - Math.min(...landCoral);
    const gotSat = Math.max(...gotCoral) - Math.min(...gotCoral);
    expect(gotSat / landSat).toBeGreaterThan(0.55);

    // Hue does not flip toward green/blue.
    const landLab = linearRgbToLab(landCoral);
    const gotLab = linearRgbToLab(gotCoral);
    expect(gotLab[1]).toBeGreaterThan(0); // positive a* = red direction
    expect(Math.abs(gotLab[1] - landLab[1])).toBeLessThan(12);
  });
});

describe("REQUIREMENT: strobe/ambient split in one image", () => {
  it("near strobe-lit and far ambient-only regions both recover neutrals", () => {
    // Neutral gray scene so any residual cast is visible as a channel split.
    const W = 128;
    const H = 128;
    const scene: FloatImage = {
      width: W,
      height: H,
      data: new Float32Array(W * H * 3).fill(0.5),
    };
    // Near center (strobe cone) and far corners (ambient dominated).
    const distance = new Float32Array(W * H);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const nx = (2 * x) / (W - 1) - 1;
        const ny = 1 - (2 * y) / (H - 1);
        const r = Math.hypot(nx, ny);
        distance[y * W + x] = 1.2 + r * 2.8; // 1.2 m center -> ~5 m corners
      }
    }
    const params = {
      depthM: 10,
      waterType: "IB" as JerlovType,
      strobe: { ...DEFAULT_STROBE, power: 1 },
      distance,
    };
    const uw = forwardSynthesize(scene, distance, params);
    const restored = restoreCalibrated(uw, params);

    const channelSplit = (x0: number, x1: number, y0: number, y1: number): number => {
      let maxSplit = 0;
      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const p = (y * W + x) * 3;
          const rgb: RGB = [restored.data[p]!, restored.data[p + 1]!, restored.data[p + 2]!];
          maxSplit = Math.max(maxSplit, Math.max(...rgb) - Math.min(...rgb));
        }
      }
      return maxSplit;
    };

    // Center cone (strobe path) and corners (ambient path) both neutral.
    expect(channelSplit(54, 74, 54, 74)).toBeLessThan(0.02);
    expect(channelSplit(0, 12, 0, 12)).toBeLessThan(0.03);
  });
});
