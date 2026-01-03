import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "path";

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
      "/flowcraft_proto.v1.FlowService": {
        target: "http://localhost:3000",
        changeOrigin: true,
      },
    },
  },
  // @ts-expect-error Vitest config is not recognized by Vite's type definitions but works at runtime
  test: {
    globals: true,
    environment: "jsdom",
    setupFiles: "./src/test/setup.ts",
  },
});
