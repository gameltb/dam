import "dotenv/config";

import { initGlobal } from "../utils/initGlobal";
initGlobal();

import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import { fastify } from "fastify";

import { FlowService } from "@/generated/flowcraft/v1/core/service_pb";

import { wrapReducers } from "../utils/pb-client";
import { SERVER_CONFIG } from "./config";
import { AssetService } from "./services/AssetService";
import { initConfigSync } from "./services/ConfigSyncService";
import { DurableWorkflowService } from "./services/DurableWorkflowService";
import { FlowServiceImpl } from "./services/FlowService";
import { loadFromDisk } from "./services/PersistenceService";
import { initTaskWatcher } from "./services/TaskService";
import "./templates"; // Triggers registration of all nodes and actions
import { initSpacetime, onSpacetimeConnect } from "./spacetimeClient";
import { ChatWorker } from "./workers/ChatWorker";
import { McpWorker } from "./workers/McpWorker";

const app = fastify();

// 1. Initialize SpacetimeDB
initSpacetime();
initTaskWatcher();
initConfigSync();
DurableWorkflowService.start();

onSpacetimeConnect((conn) => {
  const pbConn = wrapReducers(conn as any);

  const chatWorker = new ChatWorker(pbConn);
  void chatWorker.start();

  const mcpWorker = new McpWorker(pbConn);
  void mcpWorker.start();
});

// 2. Register core plugins
await app.register(multipart);
await app.register(fastifyStatic, {
  prefix: "/uploads/",
  root: SERVER_CONFIG.assetsDir,
});

// 2. Load persistent data
loadFromDisk();

// 3. Register Connect services
await app.register(fastifyConnectPlugin, {
  routes: (router) => {
    router.service(FlowService, FlowServiceImpl);
  },
});

// 4. Add standard HTTP routes
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

// 5. Start
app.listen({ host: SERVER_CONFIG.host, port: SERVER_CONFIG.port }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log(`[Server] Ready at http://${SERVER_CONFIG.host}:${SERVER_CONFIG.port.toString()}`);
});
