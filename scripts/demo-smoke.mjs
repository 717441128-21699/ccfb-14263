import { chromium } from "playwright";
const { createServer } = await import("vite");
const server = await createServer({ configFile: "vite.demo.config.ts", logLevel: "error" });
await server.listen();
const url = `http://127.0.0.1:${server.config.server.port}/index.html`;
const browser = await chromium.launch({
  args: ["--use-gl=angle", "--use-angle=swiftshader", "--enable-unsafe-swiftshader"],
});
const page = await browser.newPage();
const errors = [];
page.on("pageerror", (e) => errors.push(e.message));
page.on("console", (m) => { if (m.type() === "error") errors.push(m.text()); });
await page.goto(url);
await page.waitForSelector("#restoredCanvas");
await page.waitForFunction(
  () => {
    const c = document.getElementById("restoredCanvas");
    return c && c.width > 0;
  },
  { timeout: 30000 },
);
const status = await page.textContent("#status");
const restoredW = await page.evaluate(() => document.getElementById("restoredCanvas").width);
const downloadEnabled = await page.evaluate(() => !document.getElementById("download").disabled);
console.log("status:", status);
console.log("restored canvas width:", restoredW, "download enabled:", downloadEnabled);
console.log("page errors:", errors.length ? errors : "none");
await browser.close();
await server.close();
process.exit(errors.length ? 1 : 0);
