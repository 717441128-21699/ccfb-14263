import { describe, expect, it } from "vitest";
import { applySkinConstraint, skinWeight, SKIN_GAMUT } from "../src/skin.js";
import { linearRgbToLab, srgbDecode, srgbEncode } from "../src/color.js";
import type { FloatImage } from "../src/model.js";

function imageFromLinear(rgb: Array<[number, number, number]>): FloatImage {
  const data = new Float32Array(rgb.length * 3);
  rgb.forEach(([r, g, b], i) => {
    data[i * 3] = r;
    data[i * 3 + 1] = g;
    data[i * 3 + 2] = b;
  });
  return { width: rgb.length, height: 1, data };
}

/** Convert an encoded-sRGB triple (0..1) to linear light. */
function lin(r: number, g: number, b: number): [number, number, number] {
  return [srgbDecode(r), srgbDecode(g), srgbDecode(b)];
}

describe("skin detection (operates on encoded sRGB)", () => {
  it("accepts reference skin tones", () => {
    // ColorChecker dark/light skin, encoded sRGB normalized to 0..1.
    expect(skinWeight(115 / 255, 82 / 255, 68 / 255)).toBeGreaterThan(0.5);
    expect(skinWeight(194 / 255, 150 / 255, 130 / 255)).toBeGreaterThan(0.5);
  });

  it("rejects saturated coral reds and open water blues", () => {
    // Strong orange-red coral: too saturated for the skin ellipse.
    expect(skinWeight(0.95, 0.2, 0.1)).toBe(0);
    // Water blue.
    expect(skinWeight(0.05, 0.25, 0.5)).toBe(0);
    // Foliage green.
    expect(skinWeight(0.2, 0.45, 0.15)).toBe(0);
  });

  it("rejects out-of-gamut luma extremes", () => {
    expect(skinWeight(0.01, 0.01, 0.01)).toBe(0);
    expect(skinWeight(1, 1, 1)).toBe(0);
  });
});

describe("skin gamut constraint (operates on linear light)", () => {
  it("pulls a false-red face pixel back into the Lab skin gamut", () => {
    // Encoded [0.706,0.337,0.235] is inside the YCbCr skin locus but its
    // linear Lab a* ≈ 36.2 exceeds the gamut cap of 32 (red-channel
    // overshoot). The constraint must bring a* back inside.
    const before = lin(0.706, 0.337, 0.235);
    expect(linearRgbToLab(before)[1]).toBeGreaterThan(SKIN_GAMUT.aMax);

    const { image, stats } = applySkinConstraint(imageFromLinear([before]), 0.9);
    expect(stats.skinPixels).toBe(1);
    const lab = linearRgbToLab([
      image.data[0]!,
      image.data[1]!,
      image.data[2]!,
    ]);
    expect(lab[1]).toBeLessThanOrEqual(SKIN_GAMUT.aMax + 0.5);
    expect(lab[2]).toBeGreaterThanOrEqual(SKIN_GAMUT.bMin - 0.5);
    expect(lab[2]).toBeLessThanOrEqual(SKIN_GAMUT.bMax + 0.5);
    // Lightness is preserved, only chroma is reined in.
    expect(lab[0]).toBeGreaterThan(30);
  });

  it("leaves non-skin pixels bit-identical (coral protection)", () => {
    // Linear-light coral; its encoded form is outside the skin locus.
    const coral = lin(0.9, 0.28, 0.18);
    const img = imageFromLinear([coral]);
    const before = new Float32Array(img.data);
    const { image, stats } = applySkinConstraint(img);
    expect(stats.skinPixels).toBe(0);
    for (let i = 0; i < before.length; i++) {
      expect(image.data[i]).toBe(before[i]);
    }
  });

  it("keeps dark skin dark: lightness is never flattened to a single value", () => {
    const dark = lin(115 / 255, 82 / 255, 68 / 255);
    const light = lin(194 / 255, 150 / 255, 130 / 255);
    const { image } = applySkinConstraint(imageFromLinear([dark, light]), 0.9);
    const lDark = linearRgbToLab([image.data[0]!, image.data[1]!, image.data[2]!])[0]!;
    const lLight = linearRgbToLab([image.data[3]!, image.data[4]!, image.data[5]!])[0]!;
    expect(lLight).toBeGreaterThan(lDark + 8);
  });

  it("encodes back into a valid sRGB range", () => {
    const img = imageFromLinear([lin(0.6, 0.35, 0.28), lin(0.9, 0.2, 0.1)]);
    const { image } = applySkinConstraint(img);
    for (let i = 0; i < image.data.length; i++) {
      const v = srgbEncode(Math.max(0, image.data[i]!));
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });
});
