import { createLiveApi } from "./live";
import { createMockApi } from "./mock";
import type { Api } from "./types";

export { ApiError } from "./mock";

// `vite --mode live` 或 VITE_API_MODE=live 时连接真实后端，默认使用演示数据。
const live = import.meta.env.MODE === "live" || import.meta.env.VITE_API_MODE === "live";

export const api: Api = live ? createLiveApi() : createMockApi();
