/**
 * Strobe (underwater flash) geometry.
 *
 * Divers mount the strobe close to the camera, so the strobe-to-object and
 * object-to-camera path lengths are both ~d. Its angular support is a cone:
 * full gain inside `innerAngle`, smooth falloff to zero at `outerAngle`.
 * The smoothstep edge avoids a visible cone seam.
 */

export interface StrobeConfig {
  /**
   * Strobe radiant power relative to a reference unit. 1.0 = a typical
   * guide-number setup; 0 disables the strobe path entirely.
   */
  power: number;
  /** Cone half-angle (degrees) inside which gain is 1. */
  innerAngleDeg: number;
  /** Cone half-angle (degrees) at which gain reaches 0. */
  outerAngleDeg: number;
  /**
   * Cone center in normalized image coordinates [-1,1] (y up). Slightly
   * off-center models a strobe on a flex arm rather than the hot shoe.
   */
  centerX: number;
  centerY: number;
  /** Reference distance (m) at which power is normalized; drives 1/d^2 law. */
  referenceDistanceM: number;
  /** Small distance offset keeping the inverse-square term finite. */
  distanceOffsetM: number;
}

export const DEFAULT_STROBE: StrobeConfig = {
  power: 1.0,
  innerAngleDeg: 18,
  outerAngleDeg: 32,
  centerX: 0.06,
  centerY: 0.0,
  referenceDistanceM: 1.5,
  distanceOffsetM: 0.25,
};

function smoothstep(edge0: number, edge1: number, x: number): number {
  const t = Math.min(1, Math.max(0, (x - edge0) / (edge1 - edge0)));
  return t * t * (3 - 2 * t);
}

/**
 * Angular cone gain g(x) for a pixel.
 * @param nx normalized image x in [-1,1]
 * @param ny normalized image y in [-1,1]
 */
export function coneGain(nx: number, ny: number, strobe: StrobeConfig): number {
  const dx = nx - strobe.centerX;
  const dy = ny - strobe.centerY;
  const angleDeg = (Math.atan2(Math.hypot(dx, dy), 1) * 180) / Math.PI;
  return 1 - smoothstep(strobe.innerAngleDeg, strobe.outerAngleDeg, angleDeg);
}

/**
 * Full strobe illumination geometry factor at a pixel: cone gain times the
 * inverse-square distance law, normalized at the reference distance.
 */
export function strobeGeometry(
  nx: number,
  ny: number,
  distanceM: number,
  strobe: StrobeConfig,
): number {
  const g = coneGain(nx, ny, strobe);
  const d = Math.max(0.05, distanceM);
  const inverseSquare =
    Math.pow(strobe.referenceDistanceM + strobe.distanceOffsetM, 2) /
    Math.pow(d + strobe.distanceOffsetM, 2);
  return strobe.power * g * inverseSquare;
}
