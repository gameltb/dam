import "dotenv/config";
import { fastify } from "fastify";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import path from "path";
import { Assets } from "./assets";
import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import { FlowService } from "../generated/flowcraft/v1/core/service_pb";
import { loadFromDisk } from "./db";
import { FlowServiceImpl } from "./service";
import "./templates"; // 触发所有节点和动作的注册

const app = fastify();

// 1. 注册核心插件
const storageDir =
  process.env.FLOWCRAFT_STORAGE_DIR || path.join(process.cwd(), "storage");
const assetsDir = path.join(storageDir, "assets");

await app.register(multipart);
await app.register(fastifyStatic, {
  root: assetsDir,
  prefix: "/uploads/",
});

// 2. 加载持久化数据
loadFromDisk();

// 3. 注册 Connect 服务
await app.register(fastifyConnectPlugin, {
  routes: (router) => {
    router.service(FlowService, FlowServiceImpl);
  },
});

// 4. 添加标准 HTTP 路由
app.post("/api/upload", async (req, reply) => {
  try {
    const data = await req.file();
    if (!data) return reply.code(400).send({ error: "No file uploaded" });

    const buffer = await data.toBuffer();
    const asset = await Assets.saveAsset({
      name: data.filename,
      mimeType: data.mimetype,
      buffer,
    });

    return asset;
  } catch (err: unknown) {
    console.error("[Upload Error]", err);
    return reply.code(500).send({ error: (err as Error).message });
  }
});

// 5. 启动
app.listen({ port: 3000, host: "0.0.0.0" }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log("[Server] Ready with assets and Connect service.");
});
