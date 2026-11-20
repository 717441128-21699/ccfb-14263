/**
 * Demo driver: synthesize an underwater photo from a procedural land scene and
 * restore it on the WebGL2 pipeline, drawing before/after to two canvases.
 */
import { synthesizeUnderwater } from "../src/scene.js";
import { WebGL2Restorer } from "../src/webgl2.js";
import { restoreCpu } from "../src/pipeline.js";
import { toDisplayBytes } from "../src/tone.js";
import { DEFAULT_STROBE } from "../src/strobe.js";
import type { JerlovType } from "../src/water.js";
import type { FloatImage } from "../src/model.js";

const $ = <T extends HTMLElement>(id: string): T => document.getElementById(id) as T;

const depthEl = $<HTMLInputElement>("depth");
const strobeEl = $<HTMLInputElement>("strobe");
const depthLabel = $<HTMLSpanElement>("depthLabel");
const strobeLabel = $<HTMLSpanElement>("strobeLabel");
const statusEl = $<HTMLDivElement>("status");
const downloadBtn = $<HTMLButtonElement>("download");

depthEl.addEventListener("input", () => (depthLabel.textContent = depthEl.value));
strobeEl.addEventListener("input", () => (strobeLabel.textContent = strobeEl.value));

function drawBytes(
  canvas: HTMLCanvasElement,
  bytes: Uint8ClampedArray,
  width: number,
  height: number,
  channels: 3 | 4,
): void {
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  if (channels === 4) {
    img.data.set(bytes);
  } else {
    for (let i = 0, j = 0; i < bytes.length; i += 3, j += 4) {
      img.data[j] = bytes[i]!;
      img.data[j + 1] = bytes[i + 1]!;
      img.data[j + 2] = bytes[i + 2]!;
      img.data[j + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
}

let lastRestored: { bytes: Uint8ClampedArray; width: number; height: number } | null = null;

async function run(): Promise<void> {
  const width = Number($<HTMLInputElement>("width").value);
  const height = Number($<HTMLInputElement>("height").value);
  const depthM = Number(depthEl.value);
  const waterType = $<HTMLSelectElement>("waterType").value as JerlovType;
  const strobePower = Number(strobeEl.value);
  const applySkin = $<HTMLInputElement>("skin").checked;
  const blind = $<HTMLInputElement>("blind").checked;

  statusEl.textContent = "正在合成水下照片…";
  await new Promise((r) => setTimeout(r, 0));

  const t0 = performance.now();
  const { underwater, params } = synthesizeUnderwater({
    width,
    height,
    depthM,
    waterType,
    strobePower,
  });

  // Draw the synthesized underwater input.
  drawBytes(
    $<HTMLCanvasElement>("underwaterCanvas"),
    toDisplayBytes(underwater, 1),
    width,
    height,
    3,
  );

  statusEl.textContent = "WebGL2 恢复中…";
  await new Promise((r) => setTimeout(r, 0));

  const runParams = {
    ...params,
    strobe: { ...DEFAULT_STROBE, power: strobePower },
    estimateDistance: blind,
    distance: blind ? undefined : params.distance,
  };

  // WebGL must render to its own offscreen canvas: a canvas can hold only one
  // context type, and the visible restoredCanvas needs a 2D context to blit.
  const glCanvas = document.createElement("canvas");
  const gl = glCanvas.getContext("webgl2", { antialias: false });
  let bytes: Uint8ClampedArray;
  let gpuMs = 0;
  let backend = "WebGL2";

  if (gl && gl.getExtension("EXT_color_buffer_float")) {
    try {
      const restorer = new WebGL2Restorer(gl);
      const result = restorer.restore(underwater, {
        params: runParams,
        applySkin,
        encodeSrgb: true,
        tileSize: 1024,
      });
      restorer.dispose();
      bytes = result.srgb!;
      gpuMs = result.gpuMs;
    } catch (err) {
      console.warn("WebGL2 path failed, falling back to CPU", err);
      const cpu = restoreCpu(underwater, { params: runParams, applySkin });
      bytes = toDisplayBytes(cpu.restored, 1);
      backend = "CPU (回退)";
    }
  } else {
    const cpu = restoreCpu(underwater, { params: runParams, applySkin });
    bytes = toDisplayBytes(cpu.restored, 1);
    backend = "CPU (无 WebGL2)";
  }

  drawBytes($<HTMLCanvasElement>("restoredCanvas"), bytes, width, height, 4);
  const total = performance.now() - t0;
  statusEl.textContent =
    `${backend} 完成 · 分辨率 ${width}×${height} (${(width * height / 1e6).toFixed(2)} MP) · ` +
    `GPU 计算 ${gpuMs.toFixed(1)} ms · 总耗时 ${total.toFixed(1)} ms` +
    (blind ? " · 盲估计" : " · 标定距离");

  lastRestored = { bytes, width, height };
  downloadBtn.disabled = false;
}

downloadBtn.addEventListener("click", () => {
  if (!lastRestored) return;
  const { bytes, width, height } = lastRestored;
  const canvas = document.createElement("canvas");
  canvas.width = width;
  canvas.height = height;
  const ctx = canvas.getContext("2d")!;
  const img = ctx.createImageData(width, height);
  img.data.set(bytes);
  ctx.putImageData(img, 0, 0);
  const a = document.createElement("a");
  a.download = "restored.png";
  a.href = canvas.toDataURL("image/png");
  a.click();
});

$<HTMLButtonElement>("run").addEventListener("click", () => {
  run().catch((err) => {
    console.error(err);
    statusEl.textContent = "出错: " + (err as Error).message;
  });
});

// Auto-run once on load.
void run();

// Keep the FloatImage type referenced for type-checking clarity.
export type { FloatImage };
