/**
 * Underwater optics: Jerlov water types and depth-dependent attenuation.
 *
 * Beam attenuation c(lambda) = absorption a(lambda) + scattering b(lambda).
 * Values are RGB-channel approximations (center wavelengths ~620/540/460 nm)
 * derived from published Jerlov / Mobley tables, in inverse meters.
 *
 * Red attenuates an order of magnitude faster than blue in open water, which
 * is exactly why naive curve-pulling produces false reds: the red channel at
 * depth carries almost no scene signal, only noise.
 */

export type JerlovType = "I" | "IA" | "IB" | "II" | "III" | "1C" | "3C" | "5C" | "7C" | "9C";

export interface WaterOptics {
  /** Jerlov type label. */
  type: JerlovType;
  /** Beam attenuation coefficient c per RGB channel [r,g,b], 1/m. */
  attenuation: [number, number, number];
  /**
   * Backscatter coefficient b_b per RGB channel, 1/m. Together with the
   * phase-function factor it sets the veiling-light build-up rate.
   */
  backscatter: [number, number, number];
  /**
   * Open-ocean veiling-light color in LINEAR sRGB (the "water color" an
   * infinite water column converges to). Coastal types carry more green.
   */
  veilLinear: [number, number, number];
}

/**
 * Beam attenuation c [1/m] at RGB center wavelengths.
 * Sources: Solonenko & Mobley (2015) Jerlov table compression;
 * Smith & Baker (1981) diffuse/beam attenuation cross-checks.
 */
const ATTENUATION: Record<JerlovType, [number, number, number]> = {
  // Oceanic types: extremely clear to moderately productive.
  I: [0.355, 0.078, 0.052],
  IA: [0.379, 0.091, 0.062],
  IB: [0.407, 0.108, 0.076],
  II: [0.449, 0.142, 0.103],
  III: [0.558, 0.211, 0.169],
  // Coastal types: particles + gelbstoff shift attenuation toward blue loss.
  "1C": [0.502, 0.168, 0.125],
  "3C": [0.613, 0.272, 0.224],
  "5C": [0.705, 0.363, 0.318],
  "7C": [0.812, 0.461, 0.425],
  "9C": [0.933, 0.572, 0.559],
};

/**
 * Backscatter coefficient b_b [1/m]. In oceanic water b_b is dominated by
 * molecular scattering (~0.003 at 500 nm); coastal water adds particles.
 */
const BACKSCATTER: Record<JerlovType, [number, number, number]> = {
  I: [0.0024, 0.0038, 0.0055],
  IA: [0.0028, 0.0044, 0.0064],
  IB: [0.0033, 0.0052, 0.0075],
  II: [0.0042, 0.0066, 0.0094],
  III: [0.0061, 0.0095, 0.0132],
  "1C": [0.0075, 0.0118, 0.0162],
  "3C": [0.0135, 0.0205, 0.0268],
  "5C": [0.0195, 0.0288, 0.0362],
  "7C": [0.0268, 0.0382, 0.0465],
  "9C": [0.0355, 0.0492, 0.0585],
};

/**
 * Veiling-light color (linear sRGB) per water type — the asymptotic radiance
 * of an infinitely thick water column under D65 surface light.
 */
const VEIL: Record<JerlovType, [number, number, number]> = {
  I: [0.012, 0.061, 0.105],
  IA: [0.014, 0.066, 0.108],
  IB: [0.016, 0.072, 0.112],
  II: [0.02, 0.085, 0.118],
  III: [0.028, 0.105, 0.125],
  "1C": [0.03, 0.115, 0.115],
  "3C": [0.045, 0.135, 0.105],
  "5C": [0.058, 0.148, 0.098],
  "7C": [0.07, 0.158, 0.092],
  "9C": [0.082, 0.168, 0.086],
};

export const JERLOV_TYPES: JerlovType[] = [
  "I",
  "IA",
  "IB",
  "II",
  "III",
  "1C",
  "3C",
  "5C",
  "7C",
  "9C",
];

export function getWaterOptics(type: JerlovType): WaterOptics {
  return {
    type,
    attenuation: [...ATTENUATION[type]] as [number, number, number],
    backscatter: [...BACKSCATTER[type]] as [number, number, number],
    veilLinear: [...VEIL[type]] as [number, number, number],
  };
}

/**
 * Depth correction for the attenuation coefficient.
 *
 * Near-surface waters carry a dissolved-particle load; the beam attenuation
 * profile relaxes toward the open-water Jerlov baseline with a 12 m scale
 * depth. This is a compact empirical fit (c(z) = c_deep + (c0-c_deep)e^-z/12)
 * used by the synthesizer so test images carry realistic depth structure
 * rather than one flat coefficient.
 *
 * @param base   Jerlov attenuation at depth
 * @param depthM camera depth below surface in meters
 * @returns effective attenuation at that depth
 */
export function attenuationAtDepth(
  base: [number, number, number],
  depthM: number,
): [number, number, number] {
  const SCALE_DEPTH = 12;
  // Near-surface excess attenuation, strongest in red (gelbstoff/particles).
  const excess: [number, number, number] = [0.12, 0.045, 0.02];
  const f = Math.exp(-Math.max(0, depthM) / SCALE_DEPTH);
  return [
    base[0] + excess[0] * f,
    base[1] + excess[1] * f,
    base[2] + excess[2] * f,
  ];
}

/**
 * Spectral downwelling irradiance loss from the surface to the camera depth.
 * E(z) = E(0) exp(-K_d z), using the diffuse attenuation K_d approximated as
 * 0.82 c (Mobley relationship for Jerlov oceanic waters).
 */
export function downwellingFactor(
  attenuation: [number, number, number],
  depthM: number,
): [number, number, number] {
  const Kd = attenuation.map((c) => 0.82 * c) as [number, number, number];
  return [
    Math.exp(-Kd[0] * depthM),
    Math.exp(-Kd[1] * depthM),
    Math.exp(-Kd[2] * depthM),
  ];
}
