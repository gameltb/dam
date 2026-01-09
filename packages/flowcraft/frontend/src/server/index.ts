import "dotenv/config";
import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import { fastify } from "fastify";

import { FlowService } from "@/generated/flowcraft/v1/core/service_pb";

import { SERVER_CONFIG } from "./config";
import { AssetService } from "./services/AssetService";
import { FlowServiceImpl } from "./services/FlowService";
import { loadFromDisk } from "./services/PersistenceService";
import "./templates"; // 触发所有节点和动作的注册

const app = fastify();

// 1. 注册核心插件
await app.register(multipart);
await app.register(fastifyStatic, {
  prefix: "/uploads/",
  root: SERVER_CONFIG.assetsDir,
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
    if (!data) return await reply.code(400).send({ error: "No file uploaded" });

    const buffer = await data.toBuffer();
    const asset = AssetService.saveAsset({
      buffer,
      mimeType: data.mimetype,
      name: data.filename,
    });

    return asset;
  } catch (err: unknown) {
    console.error("[Upload Error]", err);
    return await reply.code(500).send({ error: (err as Error).message });
  }
});

// 5. 启动
app.listen({ host: SERVER_CONFIG.host, port: SERVER_CONFIG.port }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(
    `[Server] Ready at http://${SERVER_CONFIG.host}:${SERVER_CONFIG.port.toString()}`,
  );
});
