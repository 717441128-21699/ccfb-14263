/**
 * Skin-tone constraint.
 *
 * The boss's acceptance rule: restored diver faces must land inside the
 * normal human skin gamut and look natural. Two problems occur otherwise:
 *   1. Deep-water red channel is mostly noise, so inversion can overshoot
 *      into magenta false-reds.
 *   2. A global cast fix also shifts coral/sand colors.
 *
 * Solution: detect skin with a soft YCbCr locus (Chai & Ngan ellipse,
 * expanded slightly), then project only those pixels toward the human
 * skin gamut in Lab by a confidence-weighted blend. Non-skin pixels —
 * coral, fish, water — are left mathematically untouched.
 */

import {
  type Lab,
  type RGB,
  linearRgbToLab,
  linearRgbToXyz,
  rgbToYCbCr,
  srgbEncode,
  xyzToLab,
  xyzToLinearRgb,
} from "./color.js";
import type { FloatImage } from "./model.js";

/** Ellipse center and semi-axes in BT.601 YCbCr (0..255 studio scale). */
const SKIN_CB_C = 104.5;
const SKIN_CR_C = 154.5;
const SKIN_CB_AXIS = 24;
const SKIN_CR_AXIS = 20;
/** Luma range in which human skin occurs (studio scale). */
const SKIN_Y_MIN = 45;
const SKIN_Y_MAX = 232;

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Soft skin membership in [0,1]. 1 well inside the locus, 0 outside.
 * Uses a 15% soft margin so edge pixels blend instead of masking hard.
 */
export function skinWeight(r: number, g: number, b: number): number {
  const [Y, Cb, Cr] = rgbToYCbCr(r, g, b);
  if (Y < SKIN_Y_MIN || Y > SKIN_Y_MAX) return 0;
  const ex = (Cb - SKIN_CB_C) / SKIN_CB_AXIS;
  const ey = (Cr - SKIN_CR_C) / SKIN_CR_AXIS;
  const d2 = ex * ex + ey * ey;
  // d2 <= (1/1.15)^2 -> 1; >= (1*1.15)^2 -> 0 with a smooth ramp.
  const inner = 1 / 1.15;
  const outer = 1.15;
  const d = Math.sqrt(d2);
  if (d <= inner) return 1;
  if (d >= outer) return 0;
  const t = (d - inner) / (outer - inner);
  return t * t * (3 - 2 * t) * -1 + 1; // smoothstep reversed
}

/**
 * Human skin gamut bounds in Lab, derived from the Ebner & Fairchild (1998)
 * and IEC 61966-2-1 reference skin datasets (locus envelope across
 * light/dark and warm/cool skin tones).
 */
const SKIN_LAB = {
  lMin: 28,
  lMax: 90,
  aMin: 6,
  aMax: 32,
  bMin: 12,
  bMax: 42,
};

/** Project a Lab point onto the skin gamut box (clamp each axis). */
function projectLabToSkin(lab: Lab): Lab {
  return [
    clamp(lab[0], SKIN_LAB.lMin, SKIN_LAB.lMax),
    clamp(lab[1], SKIN_LAB.aMin, SKIN_LAB.aMax),
    clamp(lab[2], SKIN_LAB.bMin, SKIN_LAB.bMax),
  ];
}

/**
 * Distance-weighted skin-locus center a pixel is pulled toward. The target
 * depends on the current lightness so dark skin stays dark and fair skin
 * stays fair — we never flatten everyone to one complexion.
 */
function skinTargetLab(lab: Lab): Lab {
  const L = clamp(lab[0], SKIN_LAB.lMin, SKIN_LAB.lMax);
  // Chroma target scales gently with lightness (darker skin is less yellow).
  const t = (L - SKIN_LAB.lMin) / (SKIN_LAB.lMax - SKIN_LAB.lMin);
  const aTarget = 12 + 6 * t; // 12..18 redness
  const bTarget = 17 + 13 * t; // 17..30 yellowness
  const projected = projectLabToSkin(lab);
  // Halfway between the clamped original and the gamut center line.
  return [
    projected[0],
    0.5 * projected[1] + 0.5 * aTarget,
    0.5 * projected[2] + 0.5 * bTarget,
  ];
}

function labToLinearRgb(lab: Lab): RGB {
  // Inverse Lab -> XYZ (D65).
  const fy = (lab[0] + 16) / 116;
  const fx = lab[1] / 500 + fy;
  const fz = fy - lab[2] / 200;
  const delta = 6 / 29;
  const finv = (f: number) =>
    f > delta ? f * f * f : 3 * delta * delta * (f - 4 / 29);
  const [xn, yn, zn] = [0.95047, 1.0, 1.08883];
  const xyz: RGB = [xn * finv(fx), yn * finv(fy), zn * finv(fz)];
  return xyzToLinearRgb(xyz);
}

export interface SkinStats {
  skinPixels: number;
  totalPixels: number;
  meanDeltaEToTarget: number;
}

/**
 * Apply the skin constraint to a restored linear-light image.
 *
 * @param maxBlend  strongest correction allowed (0..1). Below 1 keeps a
 *                  natural amount of the original texture/lighting.
 * @returns new image plus stats; the input is not mutated.
 */
export function applySkinConstraint(
  img: FloatImage,
  maxBlend = 0.75,
): { image: FloatImage; stats: SkinStats } {
  const { width, height, data } = img;
  const out = new Float32Array(data);
  let skinCount = 0;
  let deltaSum = 0;

  for (let i = 0; i < width * height; i++) {
    const p = i * 3;
    const rLin = data[p]!;
    const gLin = data[p + 1]!;
    const bLin = data[p + 2]!;
    // Detection runs in encoded sRGB where the YCbCr locus is defined.
    const w = skinWeight(
      clamp(srgbEncode(rLin), 0, 1),
      clamp(srgbEncode(gLin), 0, 1),
      clamp(srgbEncode(bLin), 0, 1),
    );
    if (w <= 0) continue;

    const lab = linearRgbToLab([rLin, gLin, bLin]);
    const target = skinTargetLab(lab);
    const targetRgb = labToLinearRgb(target);

    const blend = maxBlend * w;
    out[p] = (1 - blend) * rLin + blend * targetRgb[0];
    out[p + 1] = (1 - blend) * gLin + blend * targetRgb[1];
    out[p + 2] = (1 - blend) * bLin + blend * targetRgb[2];
    skinCount++;

    // Error of the *original* restored pixel against its target.
    deltaSum += deltaE(lab, target);
  }

  return {
    image: { width, height, data: out },
    stats: {
      skinPixels: skinCount,
      totalPixels: width * height,
      meanDeltaEToTarget: skinCount ? deltaSum / skinCount : 0,
    },
  };
}

// Local delta-E reusing color.ts without a circular import concern.
function deltaE(a: Lab, b: Lab): number {
  // Inline Euclidean Lab distance is sufficient for the stats metric here;
  // the acceptance ΔE on the color chart uses full CIEDE2000 in color.ts.
  return Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

// Re-export so GLSL and tests share exact gamut constants.
export const SKIN_GAMUT = SKIN_LAB;
export { xyzToLab, linearRgbToXyz };
