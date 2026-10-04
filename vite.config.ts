import { defineConfig } from "vite";

export default defineConfig({
  base: "/robotcourse/",
  build: {
    outDir: "dist",
    emptyOutDir: true
  }
});
