# syntax=docker/dockerfile:1
# 前端镜像：构建 React 静态文件，由 nginx 提供并把 /api 反代到后端。
# 国内构建可换 npm 镜像:
#   docker build --build-arg NPM_REGISTRY=https://registry.npmmirror.com -t rag-qa-web ./frontend

FROM node:22-alpine AS builder

ARG NPM_REGISTRY=https://registry.npmjs.org

WORKDIR /web

COPY package.json package-lock.json ./
RUN npm ci --registry="$NPM_REGISTRY"

COPY . .
# 不设置时前端默认使用演示数据（mock），部署必须连接真实后端
ENV VITE_API_MODE=live
RUN npm run build


FROM nginx:1.27-alpine

COPY nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=builder /web/dist /usr/share/nginx/html

EXPOSE 80
