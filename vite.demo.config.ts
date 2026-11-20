import { defineConfig } from "vite";

/**
 * Vite config for the browser demo. The demo entry lives in demo/ but imports
 * the shared TypeScript engine from ../src, so the project root is the
 * workspace root while the dev server opens the demo page.
 */
export default defineConfig({
  root: "demo",
  publicDir: false,
  build: {
    outDir: "../dist-demo",
    emptyOutDir: true,
    target: "es2022",
  },
  server: {
    host: "127.0.0.1",
    open: "/index.html",
  },
});
