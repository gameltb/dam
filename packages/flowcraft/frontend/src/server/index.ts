import "dotenv/config";
import { fastify } from "fastify";
import multipart from "@fastify/multipart";
import fastifyStatic from "@fastify/static";
import path from "path";
import { Assets } from "./assets";
import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import { FlowService } from "../generated/flowcraft/v1/core/service_pb";
import { create } from "@bufbuild/protobuf";
import {
  FlowMessageSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  TemplateDiscoveryResponseSchema,
  type FlowMessage,
} from "../generated/flowcraft/v1/core/service_pb";
import {
  ActionDiscoveryResponseSchema,
  ActionTemplateSchema,
} from "../generated/flowcraft/v1/core/action_pb";
import { MutationSource } from "../generated/flowcraft/v1/core/base_pb";
import { v4 as uuidv4 } from "uuid";
import {
  serverGraph,
  serverVersion,
  incrementVersion,
  eventBus,
  loadFromDisk,
} from "./db";
import { NodeRegistry } from "./registry";
import "./templates"; // 触发注册
import "./actions/chat";
import {
  toProtoNode,
  toProtoEdge,
  fromProtoNode,
  fromProtoNodeData,
} from "../utils/protoAdapter";

import { isDynamicNode, type DynamicNodeData } from "../types";

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
    router.service(FlowService, {
      async *watchGraph(_req, ctx) {
        console.log("[Server] Client connected");

        try {
          const snapshot = create(FlowMessageSchema, {
            messageId: uuidv4(),
            timestamp: BigInt(Date.now()),
            payload: {
              case: "snapshot",
              value: create(GraphSnapshotSchema, {
                nodes: serverGraph.nodes.map(toProtoNode) as any,
                edges: serverGraph.edges.map(toProtoEdge) as any,
                version: BigInt(serverVersion),
              }),
            },
          });
          yield snapshot;
        } catch (err) {
          console.error("[Server] Error generating snapshot:", err);
          // Don't rethrow, let the loop handle it or send an error if we had an ErrorMessage case
        }

        const queue: FlowMessage[] = [];
        let waitingPromise: {
          resolve: () => void;
          promise: Promise<void>;
        } | null = null;

        const pushToQueue = (type: string, payload: any) => {
          queue.push(
            create(FlowMessageSchema, {
              messageId: uuidv4(),
              timestamp: BigInt(Date.now()),
              payload: { case: type as any, value: payload },
            }),
          );
          if (waitingPromise) {
            waitingPromise.resolve();
            waitingPromise = null;
          }
        };

        const onMutations = (m: any) => pushToQueue("mutations", m);
        const onTaskUpdate = (t: any) => pushToQueue("taskUpdate", t);
        const onStreamChunk = (c: any) => pushToQueue("streamChunk", c);
        const onWidgetSignal = (s: any) => pushToQueue("widgetSignal", s);

        eventBus.on("mutations", onMutations);
        eventBus.on("taskUpdate", onTaskUpdate);
        eventBus.on("streamChunk", onStreamChunk);
        eventBus.on("widgetSignal", onWidgetSignal);

        try {
          while (!ctx.signal.aborted) {
            while (queue.length > 0) {
              yield queue.shift()!;
            }

            if (!waitingPromise) {
              let resolve: () => void = () => {};
              const promise = new Promise<void>((r) => {
                resolve = r;
              });
              waitingPromise = { resolve, promise };
            }

            // Wait for next message or a heartbeat/timeout
            await Promise.race([
              waitingPromise.promise,
              new Promise((r) => setTimeout(r, 10000)), // 10s heartbeat timeout
            ]);
          }
        } finally {
          eventBus.off("mutations", onMutations);
          eventBus.off("taskUpdate", onTaskUpdate);
          eventBus.off("streamChunk", onStreamChunk);
          eventBus.off("widgetSignal", onWidgetSignal);
          console.log("[Server] Client disconnected, cleaned up listeners");
        }
      },

      async applyMutations(req) {
        const { mutations, source } = req;
        mutations.forEach((mut) => {
          const op = mut.operation;
          if (!op.case) return;

          switch (op.case) {
            case "addNode":
              if (op.value.node) {
                const node = fromProtoNode(op.value.node);
                const templateId = (node.data as any).typeId as string;
                const template = NodeRegistry.getDefinition(
                  templateId || "",
                )?.template;

                if (template && template.defaultState) {
                  const defaultData = fromProtoNodeData(template.defaultState);
                  const nodeData = node.data as DynamicNodeData;
                  const templateData = defaultData;

                  // 合并逻辑
                  node.data = {
                    ...templateData,
                    ...nodeData,
                    activeMode: nodeData.activeMode || templateData.activeMode,
                    modes:
                      nodeData.modes && nodeData.modes.length > 0
                        ? nodeData.modes
                        : templateData.modes,
                    widgetsValues: {
                      ...(templateData.widgetsValues || {}),
                      ...(nodeData.widgetsValues || {}),
                    },
                  };

                  if (
                    node.data.label === "Loading..." &&
                    template.displayName
                  ) {
                    node.data.label = template.displayName;
                  }
                }

                serverGraph.nodes.push(node);
                op.value.node = toProtoNode(node);
              }
              break;
            case "updateNode": {
              const val = op.value;
              const node = serverGraph.nodes.find((n) => n.id === val.id);
              if (node) {
                if (val.presentation) {
                  const pres = val.presentation;
                  if (pres.position)
                    node.position = { x: pres.position.x, y: pres.position.y };
                  if (pres.width || pres.height) {
                    node.measured = {
                      width: pres.width || node.measured?.width || 0,
                      height: pres.height || node.measured?.height || 0,
                    };
                  }
                  node.parentId =
                    pres.parentId === "" ? undefined : pres.parentId;
                }
                if (val.data) {
                  const appData = fromProtoNodeData(val.data);
                  node.data = { ...node.data, ...appData };
                }
              }
              break;
            }
            case "removeNode":
              serverGraph.nodes = serverGraph.nodes.filter(
                (n) => n.id !== op.value.id,
              );
              break;
            case "addEdge":
              if (op.value.edge) {
                const e = op.value.edge;
                serverGraph.edges.push({
                  id: e.edgeId,
                  source: e.sourceNodeId,
                  target: e.targetNodeId,
                  sourceHandle: e.sourceHandle || undefined,
                  targetHandle: e.targetHandle || undefined,
                  data: (e.metadata as any) || {},
                });
              }
              break;
            case "removeEdge":
              serverGraph.edges = serverGraph.edges.filter(
                (e) => e.id !== op.value.id,
              );
              break;
            case "clearGraph":
              serverGraph.nodes = [];
              serverGraph.edges = [];
              break;
          }
        });

        incrementVersion();
        eventBus.emit(
          "mutations",
          create(MutationListSchema, {
            mutations,
            sequenceNumber: BigInt(serverVersion),
            source: source || MutationSource.SOURCE_USER,
          }),
        );
        return {};
      },

      async discoverTemplates() {
        return create(TemplateDiscoveryResponseSchema, {
          templates: NodeRegistry.getTemplates(),
        });
      },

      async discoverActions(req) {
        const actions: any[] = [];
        actions.push(...NodeRegistry.getGlobalActions());

        const node = serverGraph.nodes.find((n) => n.id === req.nodeId);
        if (node) {
          const templateId = (node.data as any).typeId;
          const nodeActions = NodeRegistry.getActionsForNode(templateId);
          actions.push(...nodeActions);
        }

        if (req.selectedNodeIds.length > 1) {
          actions.push(
            create(ActionTemplateSchema, {
              id: "batch-clear",
              label: "Clear All Selected",
              strategy: 1,
            }),
          );
        }

        return create(ActionDiscoveryResponseSchema, { actions });
      },

      async executeAction(req) {
        const { actionId, sourceNodeId, paramsJson: _paramsJson } = req;
        const taskId = uuidv4();
        const params = _paramsJson ? JSON.parse(_paramsJson) : {};

        // 1. 尝试作为全局 Action 处理
        const handler = NodeRegistry.getActionHandler(actionId);
        if (handler) {
          void handler({
            ...req,
            selectedNodeIds: req.contextNodeIds || [],
            params,
            taskId,
            emitTaskUpdate: (t: any) => eventBus.emit("taskUpdate", t),
            emitMutation: (m: any) => eventBus.emit("mutations", m),
            emitStreamChunk: (c: any) => eventBus.emit("streamChunk", c),
          });
          return {};
        }

        // 2. 尝试作为节点特定的逻辑处理 (包括 node:execute 约定)
        const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);
        if (node) {
          const templateId = (node.data as any).typeId;
          const executor = NodeRegistry.getExecutor(templateId);
          if (executor) {
            void executor({
              node,
              params,
              taskId,
              emitTaskUpdate: (t: any) => eventBus.emit("taskUpdate", t),
              emitMutation: (m: any) => eventBus.emit("mutations", m),
              emitStreamChunk: (c: any) => eventBus.emit("streamChunk", c),
            });
          }
        }
        return {};
      },

      async updateWidget(req) {
        const { nodeId, widgetId, valueJson } = req;
        const node = serverGraph.nodes.find((n) => n.id === nodeId);
        if (node && isDynamicNode(node) && node.data.widgets) {
          const widget = node.data.widgets.find((w: any) => w.id === widgetId);
          if (widget) {
            widget.value = JSON.parse(valueJson);
            incrementVersion();
            const protoNode = toProtoNode(node);
            eventBus.emit(
              "mutations",
              create(MutationListSchema, {
                mutations: [
                  {
                    operation: {
                      case: "updateNode",
                      value: {
                        id: nodeId,
                        data: protoNode.state,
                        presentation: protoNode.presentation,
                      },
                    },
                  },
                ],
                sequenceNumber: BigInt(serverVersion),
                source: MutationSource.SOURCE_USER,
              }),
            );
          }
        }
        return {};
      },

      async updateNode(req) {
        const { nodeId, data, presentation } = req;
        const node = serverGraph.nodes.find((n) => n.id === nodeId);
        if (node) {
          if (data) {
            const appData = fromProtoNodeData(data);
            node.data = { ...node.data, ...appData };
          }
          if (presentation) {
            if (presentation.position)
              node.position = {
                x: presentation.position.x,
                y: presentation.position.y,
              };
            if (presentation.width || presentation.height) {
              node.measured = {
                width: presentation.width || node.measured?.width || 0,
                height: presentation.height || node.measured?.height || 0,
              };
            }
            node.parentId =
              presentation.parentId === "" ? undefined : presentation.parentId;
          }
          incrementVersion();
          const protoNode = toProtoNode(node);
          eventBus.emit(
            "mutations",
            create(MutationListSchema, {
              mutations: [
                {
                  operation: {
                    case: "updateNode",
                    value: {
                      id: nodeId,
                      data: protoNode.state,
                      presentation: protoNode.presentation,
                    },
                  },
                },
              ],
              sequenceNumber: BigInt(serverVersion),
              source: MutationSource.SOURCE_USER,
            }),
          );
        }
        return {};
      },

      async sendWidgetSignal(req) {
        eventBus.emit("widgetSignal", req);
        return {};
      },

      async updateViewport(_req) {
        return {};
      },
      async cancelTask(req) {
        console.log(`[Server] Cancel task requested: ${req.taskId}`);
        return {};
      },
    });
  },
});

// 4. 添加标准 HTTP 路由 (必须在插件之后)
app.post("/api/upload", async (req: any, reply) => {
  try {
    const data = await req.file();
    if (!data) return reply.code(400).send({ error: "No file uploaded" });

    const buffer = await data.toBuffer();
    const asset = await Assets.saveAsset({
      name: data.filename,
      mimeType: data.mimetype,
      buffer,
    });

    return reply.send(asset);
  } catch (err: any) {
    console.error("[Upload Error]", err);
    return reply.code(500).send({ error: err.message });
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
