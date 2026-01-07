import react from "@vitejs/plugin-react";
import path from "path";
import { defineConfig } from "vite";

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  server: {
    proxy: {
      "/api": {
        changeOrigin: true,
        target: "http://localhost:3000",
      },
      "/flowcraft_proto.v1.FlowService": {
        changeOrigin: true,
        target: "http://localhost:3000",
      },
      "/uploads": {
        changeOrigin: true,
        target: "http://localhost:3000",
      },
    },
  },
  // @ts-expect-error Vitest config is not recognized by Vite's type definitions but works at runtime
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: "./src/test/setup.ts",
  },
});
