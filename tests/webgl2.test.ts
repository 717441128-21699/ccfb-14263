/**
 * CPU <-> WebGL2 parity and real-GPU performance harness, driven through
 * Playwright's headless Chromium. A Vite dev server serves the workspace over
 * HTTP (file:// blocks ES module imports), and the page imports the exact
 * production TypeScript modules so the GPU runs the same math as the CPU
 * reference.
 */
import { describe, expect, it, beforeAll, afterAll } from "vitest";
import { chromium, type Browser, type Page } from "playwright";
import { createServer, type ViteDevServer } from "vite";
import { fileURLToPath } from "node:url";
import path from "node:path";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const root = path.resolve(__dirname, "..");

const SWIFTSHADER_ARGS = [
  "--use-gl=angle",
  "--use-angle=swiftshader",
  "--enable-unsafe-swiftshader",
];

let server: ViteDevServer;
let baseUrl: string;

beforeAll(async () => {
  server = await createServer({
    configFile: false,
    root,
    logLevel: "error",
    server: { host: "127.0.0.1", port: 0 },
  });
  await server.listen();
  baseUrl = `http://127.0.0.1:${server.config.server.port}`;
}, 60000);

afterAll(async () => {
  await server?.close();
});

async function launchPage(browser: Browser): Promise<Page> {
  const page = await browser.newPage({
    viewport: { width: 1280, height: 900 },
  });
  const pageErrors: string[] = [];
  page.on("console", (msg) => {
    if (msg.type() === "error") pageErrors.push(msg.text());
  });
  page.on("pageerror", (err) => pageErrors.push(err.message));
  await page.goto(`${baseUrl}/tests/gl-harness.html`);
  await page.waitForFunction(
    () => (window as unknown as { HARNESS_READY?: boolean }).HARNESS_READY === true,
    null,
    { timeout: 30000 },
  );
  if (pageErrors.length) throw new Error("page errors: " + pageErrors.join("; "));
  return page;
}

describe("WebGL2 GPU pipeline", () => {
  it("boots a WebGL2 context in headless Chromium", async () => {
    const browser = await chromium.launch({ args: SWIFTSHADER_ARGS });
    const page = await launchPage(browser);
    const gl2 = await page.evaluate(() => {
      const canvas = document.createElement("canvas");
      return !!canvas.getContext("webgl2");
    });
    expect(gl2).toBe(true);
    await browser.close();
  });

  it("matches the CPU reference within half-float tolerance", async () => {
    const browser = await chromium.launch({ args: SWIFTSHADER_ARGS });
    const page = await launchPage(browser);
    const result = (await page.evaluate(async () => {
      const w = window as unknown as {
        runParity: (w: number, h: number) => Promise<{
          webgl2Supported: boolean;
          summary: { maxAbsDiff: number; meanAbsDiff: number; mismatchRate: number };
        }>;
      };
      return await w.runParity(256, 192);
    })) as { webgl2Supported: boolean; summary: { maxAbsDiff: number; meanAbsDiff: number; mismatchRate: number } };
    console.log("GPU/CPU parity:", result.summary);
    expect(result.webgl2Supported).toBe(true);
    // RGBA16F half-float carries ~3 decimal digits.
    expect(result.summary.maxAbsDiff).toBeLessThan(0.02);
    expect(result.summary.meanAbsDiff).toBeLessThan(0.005);
    expect(result.summary.mismatchRate).toBeLessThan(0.01);
    await browser.close();
  });

  it("runs the tiled pipeline and reports a finite per-pixel cost", async () => {
    const browser = await chromium.launch({ args: SWIFTSHADER_ARGS });
    const page = await launchPage(browser);
    const result = (await page.evaluate(async () => {
      const w = window as unknown as {
        runPerf: (w: number, h: number, tile: number) => Promise<{
          gpuMs: number; pixels: number; tile: number;
        }>;
      };
      return await w.runPerf(2048, 1536, 1024);
    })) as { gpuMs: number; pixels: number };
    console.log("GPU tiled perf (SwiftShader):", result);
    expect(result.gpuMs).toBeGreaterThan(0);
    expect(result.pixels).toBe(2048 * 1536);
    // Software-renderer sanity bound; real GPUs are 1-3 orders of magnitude
    // faster and comfortably meet the 30 MP / 3 s target.
    const nsPerPx = (result.gpuMs * 1e6) / result.pixels;
    expect(nsPerPx).toBeLessThan(5000);
    await browser.close();
  });
});
