import { describe, expect, it } from "vitest";
import {
  JERLOV_TYPES,
  attenuationAtDepth,
  downwellingFactor,
  getWaterOptics,
} from "../src/water.js";

describe("Jerlov water optics", () => {
  it("red always attenuates faster than blue", () => {
    for (const type of JERLOV_TYPES) {
      const c = getWaterOptics(type).attenuation;
      expect(c[0]).toBeGreaterThan(c[1]);
      expect(c[1]).toBeGreaterThan(c[2]);
    }
  });

  it("coastal water attenuates more than clear oceanic", () => {
    expect(getWaterOptics("9C").attenuation[0]).toBeGreaterThan(
      getWaterOptics("I").attenuation[0],
    );
    expect(getWaterOptics("9C").attenuation[2]).toBeGreaterThan(
      getWaterOptics("I").attenuation[2],
    );
  });

  it("veil colors are water-like (blue oceanic, greener coastal)", () => {
    const ocean = getWaterOptics("I").veilLinear;
    expect(ocean[2]).toBeGreaterThan(ocean[0]); // blue > red
    const coastal = getWaterOptics("9C").veilLinear;
    expect(coastal[1]).toBeGreaterThan(ocean[1]); // more green near shore
  });
});

describe("depth dependence", () => {
  const base = getWaterOptics("I").attenuation;

  it("near-surface attenuation exceeds deep-water baseline and relaxes with depth", () => {
    const shallow = attenuationAtDepth(base, 0);
    const deep = attenuationAtDepth(base, 30);
    expect(shallow[0]).toBeGreaterThan(deep[0]);
    // At 30 m the exponential excess (scale depth 12 m) retains ~8% of the
    // 0.12 near-surface term -> within ~3% of the deep-water baseline.
    expect((deep[0] - base[0]) / base[0]).toBeLessThan(0.03);
    // At 60 m the profile has effectively converged to the baseline
    // (e^-5 ≈ 0.007 residual on the 0.12 excess ≈ 0.0008 absolute).
    const veryDeep = attenuationAtDepth(base, 60);
    expect(veryDeep[0]).toBeCloseTo(base[0], 2);
  });

  it("downwelling factor decreases monotonically with depth, red first", () => {
    const c = attenuationAtDepth(base, 10);
    const e5 = downwellingFactor(c, 5);
    const e20 = downwellingFactor(c, 20);
    for (let k = 0; k < 3; k++) {
      expect(e20[k]!).toBeLessThan(e5[k]!);
    }
    // At 20 m the red ambient is nearly gone while blue survives.
    expect(e20[0]!).toBeLessThan(0.02);
    expect(e20[2]!).toBeGreaterThan(e20[0]!);
  });
});
