/**
 * Color science primitives.
 *
 * All image math in this project runs in linear-light sRGB primaries (D65).
 * Lab conversions use the D65 reference white so delta-E is measured the same
 * way a color-corrected land photo would be judged.
 */

export type RGB = [number, number, number];
export type Lab = [number, number, number];

/** sRGB transfer function: encode a linear-light value to gamma sRGB (0..1). */
export function srgbEncode(c: number): number {
  return c <= 0.0031308 ? 12.92 * c : 1.055 * Math.pow(c, 1 / 2.4) - 0.055;
}

/** sRGB transfer function inverse: gamma sRGB (0..1) -> linear light. */
export function srgbDecode(c: number): number {
  return c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

/** Linear sRGB (0..1, D65) -> CIE XYZ. */
export function linearRgbToXyz(rgb: RGB): RGB {
  const [r, g, b] = rgb;
  return [
    0.4124564 * r + 0.3575761 * g + 0.1804375 * b,
    0.2126729 * r + 0.7151522 * g + 0.072175 * b,
    0.0193339 * r + 0.119192 * g + 0.9503041 * b,
  ];
}

/** CIE XYZ -> linear sRGB (0..1, D65). May leave values slightly outside 0..1. */
export function xyzToLinearRgb(xyz: RGB): RGB {
  const [x, y, z] = xyz;
  return [
    3.2404542 * x - 1.5371385 * y - 0.4985314 * z,
    -0.969266 * x + 1.8760108 * y + 0.041556 * z,
    0.0556434 * x - 0.2040259 * y + 1.0572252 * z,
  ];
}

/** D65 reference white in XYZ. */
export const D65: RGB = [0.95047, 1.0, 1.08883];

function labF(t: number): number {
  const delta = 6 / 29;
  return t > delta * delta * delta ? Math.cbrt(t) : t / (3 * delta * delta) + 4 / 29;
}

/** CIE XYZ (D65) -> CIE L*a*b*. */
export function xyzToLab(xyz: RGB): Lab {
  const [x, y, z] = xyz;
  const [xn, yn, zn] = D65;
  const fx = labF(x / xn);
  const fy = labF(y / yn);
  const fz = labF(z / zn);
  return [116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)];
}

/** Linear sRGB -> Lab convenience wrapper. */
export function linearRgbToLab(rgb: RGB): Lab {
  return xyzToLab(linearRgbToXyz(rgb));
}

/**
 * CIEDE2000 color difference (Sharma, Wu & Dalal 2005 corrected formulation).
 * Parametric weights kL = kC = kH = 1.
 */
export function deltaE2000(lab1: Lab, lab2: Lab): number {
  const [L1, a1, b1] = lab1;
  const [L2, a2, b2] = lab2;

  const C1 = Math.hypot(a1, b1);
  const C2 = Math.hypot(a2, b2);
  const Cbar = (C1 + C2) / 2;

  const G =
    0.5 * (1 - Math.sqrt(Math.pow(Cbar, 7) / (Math.pow(Cbar, 7) + Math.pow(25, 7))));
  const ap1 = (1 + G) * a1;
  const ap2 = (1 + G) * a2;
  const Cp1 = Math.hypot(ap1, b1);
  const Cp2 = Math.hypot(ap2, b2);

  const hp1 = Cp1 === 0 ? 0 : hueAngle(b1, ap1);
  const hp2 = Cp2 === 0 ? 0 : hueAngle(b2, ap2);

  const dLp = L2 - L1;
  const dCp = Cp2 - Cp1;

  let dhp: number;
  if (Cp1 === 0 || Cp2 === 0) {
    dhp = 0;
  } else {
    const diff = hp2 - hp1;
    if (Math.abs(diff) <= 180) dhp = diff;
    else if (diff > 180) dhp = diff - 360;
    else dhp = diff + 360;
  }
  const dHp = 2 * Math.sqrt(Cp1 * Cp2) * Math.sin((dhp * Math.PI) / 360);

  const Lbp = (L1 + L2) / 2;
  const Cbp = (Cp1 + Cp2) / 2;

  let hbp: number;
  if (Cp1 === 0 && Cp2 === 0) {
    hbp = 0;
  } else if (Math.abs(hp1 - hp2) <= 180) {
    hbp = (hp1 + hp2) / 2;
  } else if (hp1 + hp2 < 360) {
    hbp = (hp1 + hp2 + 360) / 2;
  } else {
    hbp = (hp1 + hp2 - 360) / 2;
  }

  const T =
    1 -
    0.17 * Math.cos(deg2rad(hbp - 30)) +
    0.24 * Math.cos(deg2rad(2 * hbp)) +
    0.32 * Math.cos(deg2rad(3 * hbp + 6)) -
    0.2 * Math.cos(deg2rad(4 * hbp - 63));

  const dTheta = 30 * Math.exp(-Math.pow((hbp - 275) / 25, 2));
  const RC =
    2 * Math.sqrt(Math.pow(Cbp, 7) / (Math.pow(Cbp, 7) + Math.pow(25, 7)));
  const SL =
    1 + (0.015 * Math.pow(Lbp - 50, 2)) / Math.sqrt(20 + Math.pow(Lbp - 50, 2));
  const SC = 1 + 0.045 * Cbp;
  const SH = 1 + 0.015 * Cbp * T;
  const RT = -Math.sin(deg2rad(2 * dTheta)) * RC;

  const kL = 1;
  const kC = 1;
  const kH = 1;
  return Math.sqrt(
    Math.pow(dLp / (kL * SL), 2) +
      Math.pow(dCp / (kC * SC), 2) +
      Math.pow(dHp / (kH * SH), 2) +
      RT * (dCp / (kC * SC)) * (dHp / (kH * SH)),
  );
}

function hueAngle(b: number, ap: number): number {
  const h = (Math.atan2(b, ap) * 180) / Math.PI;
  return h >= 0 ? h : h + 360;
}

function deg2rad(d: number): number {
  return (d * Math.PI) / 180;
}

/**
 * BT.601 YCbCr for 0..1 sRGB-gamma inputs, output in the 0..255 studio range
 * (Y 16..235, Cb/Cr 16..240). Skin ellipse thresholds are expressed in this
 * scale because the classic skin-locus literature uses it.
 */
export function rgbToYCbCr(r: number, g: number, b: number): [number, number, number] {
  const Y = 16 + 65.481 * r + 128.553 * g + 24.966 * b;
  const Cb = 128 - 37.797 * r - 74.203 * g + 112.0 * b;
  const Cr = 128 + 112.0 * r - 93.786 * g - 18.214 * b;
  return [Y, Cb, Cr];
}

/** Mean delta-E of paired pixel lists (each pixel given as linear RGB triples). */
export function meanDeltaE2000(reference: RGB[], restored: RGB[]): number {
  if (reference.length !== restored.length) {
    throw new Error("meanDeltaE2000: paired lists must have equal length");
  }
  let sum = 0;
  for (let i = 0; i < reference.length; i++) {
    const a = reference[i]!;
    const b = restored[i]!;
    sum += deltaE2000(linearRgbToLab(a), linearRgbToLab(b));
  }
  return sum / reference.length;
}

/** 95th-percentile delta-E: worst-case color error without letting noise dominate. */
export function percentileDeltaE2000(
  reference: RGB[],
  restored: RGB[],
  percentile = 95,
): number {
  if (reference.length !== restored.length) {
    throw new Error("percentileDeltaE2000: paired lists must have equal length");
  }
  const values: number[] = [];
  for (let i = 0; i < reference.length; i++) {
    values.push(
      deltaE2000(linearRgbToLab(reference[i]!), linearRgbToLab(restored[i]!)),
    );
  }
  values.sort((a, b) => a - b);
  const idx = Math.min(values.length - 1, Math.floor((percentile / 100) * values.length));
  return values[idx]!;
}
