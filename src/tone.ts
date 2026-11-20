/**
 * Tone mapping and image (de)serialization.
 *
 * The physics pipeline works in linear light where values may leave 0..1
 * after inversion (deep red channel divides a tiny transmission). A soft
 * Reinhard rolloff brings HDR back into range without clipping highlights
 * hard, and only the final display step applies the sRGB transfer curve.
 */

import { srgbDecode, srgbEncode } from "./color.js";
import type { FloatImage } from "./model.js";

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}

/**
 * Linear HDR -> linear LDR with an exposure gain and Reinhard rolloff.
 * Kept separate from sRGB encoding so the GPU stage can choose when to
 * apply the transfer function.
 */
export function reinhardLinear(img: FloatImage, exposure = 1): FloatImage {
  const out = new Float32Array(img.data.length);
  for (let i = 0; i < img.data.length; i++) {
    const x = Math.max(0, img.data[i]! * exposure);
    out[i] = x / (1 + x);
  }
  return { width: img.width, height: img.height, data: out };
}

/** Encode a linear-light image to 8-bit sRGB bytes (packed RGB). */
export function encodeSrgbBytes(img: FloatImage): Uint8ClampedArray {
  const bytes = new Uint8ClampedArray(img.data.length);
  for (let i = 0; i < img.data.length; i++) {
    bytes[i] = Math.round(clamp(srgbEncode(img.data[i]!), 0, 1) * 255);
  }
  return bytes;
}

/** Decode 8-bit sRGB bytes (packed RGB) to a linear-light FloatImage. */
export function decodeSrgbBytes(
  bytes: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
): FloatImage {
  const data = new Float32Array(width * height * 3);
  for (let i = 0; i < data.length; i++) {
    data[i] = srgbDecode(bytes[i]! / 255);
  }
  return { width, height, data };
}

/** Full display conversion: linear HDR -> sRGB bytes in one call. */
export function toDisplayBytes(img: FloatImage, exposure = 1): Uint8ClampedArray {
  return encodeSrgbBytes(reinhardLinear(img, exposure));
}
