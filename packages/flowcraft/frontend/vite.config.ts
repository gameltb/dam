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
        target: "http://localhost:3001",
      },
      "/flowcraft_proto.v1.FlowService": {
        changeOrigin: true,
        target: "http://localhost:3001",
      },
      "/spacetime": {
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/spacetime/, ""),
        target: "http://127.0.0.1:3000",
        ws: true,
      },
      "/uploads": {
        changeOrigin: true,
        target: "http://localhost:3001",
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
