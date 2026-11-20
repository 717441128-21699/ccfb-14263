/**
 * Browser-side harness entry. Bundled by Vite (see webgl2.test.ts) and loaded
 * by gl-harness.html. Exposes runParity / runPerf on window for Playwright.
 */
import { synthesizeUnderwater } from "../src/scene.js";
import { restoreCpu } from "../src/pipeline.js";
import { WebGL2Restorer } from "../src/webgl2.js";
import { toDisplayBytes } from "../src/tone.js";

declare global {
  interface Window {
    HARNESS_READY: boolean;
    runParity: (w: number, h: number) => Promise<unknown>;
    runPerf: (w: number, h: number, tile: number) => Promise<unknown>;
  }
}

function makeContext(): WebGL2RenderingContext {
  const canvas = document.createElement("canvas");
  const gl = canvas.getContext("webgl2", {
    antialias: false,
    preserveDrawingBuffer: false,
  });
  if (!gl) throw new Error("WebGL2 unavailable");
  return gl;
}

window.runParity = async (w: number, h: number) => {
  const { underwater, params } = synthesizeUnderwater({
    width: w,
    height: h,
    depthM: 10,
    waterType: "IB",
    strobePower: 1,
  });

  // CPU reference: calibrated (distance known), skin on, linear output.
  const cpu = restoreCpu(underwater, { params, applySkin: true });

  // GPU path: same params, linear float output so values are directly
  // comparable without an sRGB round trip.
  const gl = makeContext();
  const restorer = new WebGL2Restorer(gl);
  let gpuLinear;
  try {
    const result = restorer.restore(underwater, {
      params,
      applySkin: true,
      encodeSrgb: false,
    });
    gpuLinear = result.linear!;
  } finally {
    restorer.dispose();
  }

  let maxAbsDiff = 0;
  let sumAbsDiff = 0;
  let mismatches = 0;
  const n = cpu.restored.data.length;
  for (let i = 0; i < n; i++) {
    const a = cpu.restored.data[i]!;
    const b = gpuLinear.data[i]!;
    const d = Math.abs(a - b);
    maxAbsDiff = Math.max(maxAbsDiff, d);
    sumAbsDiff += d;
    // Half-float relative+absolute mismatch criterion.
    const tol = 0.02 + 0.02 * Math.abs(a);
    if (d > tol) mismatches++;
  }
  const summary = {
    maxAbsDiff,
    meanAbsDiff: sumAbsDiff / n,
    mismatchRate: mismatches / (n / 3),
    samples: n / 3,
  };
  return { webgl2Supported: true, summary };
};

window.runPerf = async (w: number, h: number, tile: number) => {
  const { underwater, params } = synthesizeUnderwater({
    width: w,
    height: h,
    depthM: 12,
    waterType: "II",
    strobePower: 1,
  });
  const gl = makeContext();
  const restorer = new WebGL2Restorer(gl);
  let gpuMs = 0;
  try {
    // Two runs; report the second (warmer) time.
    restorer.restore(underwater, { params, applySkin: true, encodeSrgb: true, tileSize: tile });
    const r2 = restorer.restore(underwater, {
      params,
      applySkin: true,
      encodeSrgb: true,
      tileSize: tile,
    });
    gpuMs = r2.gpuMs;
  } finally {
    restorer.dispose();
  }
  // Touch the display encoder so the import stays exercised.
  void toDisplayBytes;
  return { gpuMs, pixels: w * h, tile, width: w, height: h };
};

window.HARNESS_READY = true;
