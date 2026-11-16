# 部署到阿里云 ECS（Docker Compose，公网 IP 访问）

适用机器：2 vCPU / 2 GiB / Alibaba Cloud Linux 3 / 3 Mbps 固定带宽。
部署后有三个容器：`db`（PostgreSQL 18 + pgvector）、`app`（FastAPI 后端）、`web`（nginx，提供前端并反代 `/api`）。
浏览器访问 `http://<公网IP>/`。

## 1. 安全组

在 ECS 控制台的「安全组 → 入方向」放行：

| 端口 | 用途 |
| --- | --- |
| 22 | SSH（建议限制为自己的 IP） |
| 80 | 网站 |

5433（数据库）和 8000（后端）只绑定在 127.0.0.1，不要放行。

## 2. 加 swap（2 GiB 内存建议必做）

```bash
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

## 3. 安装 Docker

```bash
sudo dnf config-manager --add-repo https://mirrors.aliyun.com/docker-ce/linux/centos/docker-ce.repo
sudo sed -i 's/$releasever/8/g' /etc/yum.repos.d/docker-ce.repo
sudo dnf install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
sudo systemctl enable --now docker
```

## 4. 配置镜像加速

Docker Hub 在大陆基本无法直接拉取。在阿里云控制台「容器镜像服务 ACR → 镜像工具 → 镜像加速器」复制你自己的加速地址，然后：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<'EOF'
{
  "registry-mirrors": ["https://lwh9subo.mirror.aliyuncs.com"]
}
EOF
sudo systemctl restart docker
```

加速器拉不到所需镜像时，改用第 7 节的「本机构建后上传」。

## 5. 上传代码

服务器上 `git clone` 仓库，或在本机打包上传（不要带 `.env`、`.venv`、`node_modules`、`data`）：

```bash
git clone <仓库地址> /opt/rag-qa
```

## 6. 配置并启动

```bash
cd /opt/rag-qa
cp .env.example .env
vi .env
```

`.env` 至少需要改这些：

```ini
# 以及 LLM/Embedding 等模型配置
OPENAI_API_KEY=<your-api-key>
POSTGRES_PASSWORD=<一个强密码>
PUBLIC_HOST=47.102.252.147
AUTH_SECURE_COOKIE=false      # 纯 HTTP 访问必须关，否则登录后 Cookie 不会被浏览器保存
PIP_INDEX_URL=https://mirrors.aliyun.com/pypi/simple/
NPM_REGISTRY=https://registry.npmmirror.com
```

`.env` 里的 `DATABASE_URL` 不用改，compose 会把它替换成容器内的数据库地址。

`.env` 是以文件形式挂进后端容器的。管理后台「模型配置」页保存时会直接改写这个文件，所以重建容器后修改依然有效。容器里的后端以 uid 1000 运行，需要让它能写这个文件：

```bash
sudo chown 1000:1000 .env
chmod 600 .env
```

不改的话，管理后台保存模型配置会失败，其他功能不受影响。

启动：

```bash
docker compose up -d --build
docker compose ps
docker compose logs -f app
```

每次 `up` 都会先运行一次性的 `migrate` 容器执行数据库迁移（可重复执行），成功退出后 `app` 才启动。`docker compose ps -a` 里看到 `migrate  Exited (0)` 是正常的；如果不是 0，用 `docker compose logs migrate` 查看原因。

创建第一个管理员（全新数据库没有默认账号）：

```bash
docker compose exec app python -m app.interfaces.cli.users create --username admin --name 管理员 --role admin
```

## 7. 备选：本机构建后上传

服务器拉镜像或构建太慢（前端打包在 2 GiB 内存上也可能不够用）时，在本机构建好再传过去：

```bash
docker compose build
docker pull pgvector/pgvector:pg18
docker save rag_qa-app rag_qa-web pgvector/pgvector:pg18 | gzip > rag-qa-images.tar.gz
scp rag-qa-images.tar.gz root@47.102.252.147:/opt/rag-qa/
```

compose 固定了项目名 `rag_qa`，所以两边的镜像名一致。服务器上：

```bash
cd /opt/rag-qa
gunzip -c rag-qa-images.tar.gz | docker load
docker compose up -d --no-build
```

## 8. 日常运维

更新代码后重新构建：

```bash
git pull
docker compose up -d --build
```

备份数据库：

```bash
docker compose exec -T db pg_dump -U rag_user rag_qa | gzip > backup-$(date +%F).sql.gz
```

上传的原始文件在 `app_data` 卷里，数据库在 `pg_data` 卷里。`docker compose down` 会保留它们，`down -v` 会删除它们。

## 安全说明

- 纯 HTTP 下登录 Cookie 和密码都以明文传输。长期使用建议绑定备案域名并上 HTTPS，然后把 `AUTH_SECURE_COOKIE` 改回 `true`。
- 不要把数据库端口 5433 和后端端口 8000 放行到公网。
