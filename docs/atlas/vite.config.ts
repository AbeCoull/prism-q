import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  build: { chunkSizeWarningLimit: 900 },
  test: { include: ["tests/**/*.test.ts"] },
});
