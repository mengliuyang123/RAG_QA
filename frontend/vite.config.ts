/// <reference types="vitest/config" />
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    // 5173 落在 Windows 常见的保留端口段（5078–5177）内，会报 EACCES。
    port: 15173,
    proxy: {
      "/api": {
        target: process.env.VITE_BACKEND_URL ?? "http://127.0.0.1:8000",
        // 保留浏览器的 Host：后端用它和 Origin 比对做 CSRF 来源校验，
        // 改写成后端地址会让开发环境的写请求被当成跨站请求拒绝。
        changeOrigin: false,
      },
    },
  },
  test: {
    environment: "jsdom",
    globals: false,
  },
});
