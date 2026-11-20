/**
 * X-Rite ColorChecker Classic (24-patch) reference colors.
 *
 * Reference sRGB values are the widely published chart specifications
 * (BabelColor averages of measured charts), stored as encoded sRGB 0..255.
 * The synthesizer embeds these as linear-light reflectances; the acceptance
 * test compares restored values against the same references with CIEDE2000.
 */

import { type RGB, srgbDecode } from "./color.js";

export interface ColorPatch {
  name: string;
  /** Encoded sRGB reference, 0..255. */
  srgb: [number, number, number];
}

export const COLOR_CHECKER: ColorPatch[] = [
  { name: "dark-skin", srgb: [115, 82, 68] },
  { name: "light-skin", srgb: [194, 150, 130] },
  { name: "blue-sky", srgb: [92, 122, 157] },
  { name: "foliage", srgb: [87, 108, 67] },
  { name: "blue-flower", srgb: [133, 128, 177] },
  { name: "bluish-green", srgb: [103, 189, 170] },
  { name: "orange", srgb: [239, 163, 54] },
  { name: "purplish-blue", srgb: [71, 91, 166] },
  { name: "moderate-red", srgb: [197, 90, 102] },
  { name: "purple", srgb: [94, 60, 108] },
  { name: "yellow-green", srgb: [157, 188, 64] },
  { name: "orange-yellow", srgb: [224, 162, 39] },
  { name: "blue", srgb: [50, 79, 145] },
  { name: "green", srgb: [79, 138, 73] },
  { name: "red", srgb: [177, 55, 65] },
  { name: "yellow", srgb: [224, 202, 44] },
  { name: "magenta", srgb: [187, 86, 149] },
  { name: "cyan", srgb: [18, 132, 158] },
  { name: "white", srgb: [243, 243, 242] },
  { name: "neutral-8", srgb: [200, 200, 201] },
  { name: "neutral-6.5", srgb: [160, 161, 161] },
  { name: "neutral-5", srgb: [122, 122, 121] },
  { name: "neutral-3.5", srgb: [85, 85, 85] },
  { name: "black", srgb: [52, 52, 52] },
];

/** Linear-light reflectance of each patch (0..1). */
export function colorCheckerLinear(): RGB[] {
  return COLOR_CHECKER.map((p) =>
    p.srgb.map((v) => srgbDecode(v / 255)) as RGB,
  );
}

/**
 * Render a 6x4 ColorChecker into an RGB FloatImage. Each patch is a solid
 * color; a thin neutral border separates them. The chart occupies the frame
 * with a configurable margin so it can be embedded into a larger scene.
 */
export function renderColorChecker(
  cols: number,
  rows: number,
  margin = 0,
): { image: { width: number; height: number; data: Float32Array }; patchCenters: Array<{ x: number; y: number; index: number }> } {
  const linear = colorCheckerLinear();
  const innerW = cols - 2 * margin;
  const innerH = rows - 2 * margin;
  const patchW = innerW / 6;
  const patchH = innerH / 4;
  const data = new Float32Array(cols * rows * 3);
  const border: RGB = [0.18, 0.18, 0.18];
  const patchCenters: Array<{ x: number; y: number; index: number }> = [];

  for (let y = 0; y < rows; y++) {
    for (let x = 0; x < cols; x++) {
      const gx = x - margin;
      const gy = y - margin;
      const p = (y * cols + x) * 3;
      if (gx < 0 || gy < 0 || gx >= innerW || gy >= innerH) {
        data[p] = border[0];
        data[p + 1] = border[1];
        data[p + 2] = border[2];
        continue;
      }
      const col = Math.min(5, Math.floor(gx / patchW));
      const row = Math.min(3, Math.floor(gy / patchH));
      const patchIndex = row * 6 + col;
      const color = linear[patchIndex]!;
      data[p] = color[0];
      data[p + 1] = color[1];
      data[p + 2] = color[2];

      // Record center once per patch.
      const localX = gx - col * patchW;
      const localY = gy - row * patchH;
      if (
        localX === Math.floor(patchW / 2) &&
        localY === Math.floor(patchH / 2)
      ) {
        patchCenters.push({ x, y, index: patchIndex });
      }
    }
  }
  return { image: { width: cols, height: rows, data }, patchCenters };
}
