import { fastify } from "fastify";
import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import { FlowService } from "../generated/flowcraft/v1/core/service_pb";
import { create } from "@bufbuild/protobuf";
import {
  FlowMessageSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  TemplateDiscoveryResponseSchema,
  StreamChunkSchema,
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
import {
  NodeSchema,
  TaskUpdateSchema,
} from "../generated/flowcraft/v1/core/node_pb";
import {
  fromProtoNode,
  fromProtoNodeData,
  toProtoNode,
  toProtoEdge,
} from "../utils/protoAdapter";

import { isDynamicNode } from "../types";
import { OpenAI } from "openai";

// --- OpenAI Configuration ---
const AI_CONFIG = {
  apiKey: process.env.OPENAI_API_KEY || "your-key-here",
  baseURL: process.env.OPENAI_BASE_URL || "https://api.openai.com/v1",
  model: process.env.OPENAI_MODEL || "gpt-3.5-turbo",
};

const openai = new OpenAI({
  apiKey: AI_CONFIG.apiKey,
  baseURL: AI_CONFIG.baseURL,
});

const app = fastify();

// 加载持久化数据
loadFromDisk();

await app.register(fastifyConnectPlugin, {
  routes: (router) => {
    router.service(FlowService, {
      async *watchGraph(_req, ctx) {
        console.log("[Server] Client connected");

        yield create(FlowMessageSchema, {
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

        const queue: FlowMessage[] = [];
        let resolveQueue: (() => void) | null = null;

        const handler = (type: string, payload: any) => {
          queue.push(
            create(FlowMessageSchema, {
              messageId: uuidv4(),
              timestamp: BigInt(Date.now()),
              payload: { case: type as any, value: payload },
            }),
          );
          if (resolveQueue) resolveQueue();
        };

        const onMutations = (m: any) => {
          handler("mutations", m);
        };
        const onTaskUpdate = (t: any) => {
          handler("taskUpdate", t);
        };
        const onStreamChunk = (c: any) => {
          handler("streamChunk", c);
        };
        const onWidgetSignal = (s: any) => {
          handler("widgetSignal", s);
        };

        eventBus.on("mutations", onMutations);
        eventBus.on("taskUpdate", onTaskUpdate);
        eventBus.on("streamChunk", onStreamChunk);
        eventBus.on("widgetSignal", onWidgetSignal);

        try {
          while (!ctx.signal.aborted) {
            if (queue.length > 0) {
              yield queue.shift()!;
            } else {
              await new Promise<void>((r) => {
                resolveQueue = r;
              });
              resolveQueue = null;
            }
          }
        } finally {
          eventBus.off("mutations", onMutations);
          eventBus.off("taskUpdate", onTaskUpdate);
          eventBus.off("streamChunk", onStreamChunk);
          eventBus.off("widgetSignal", onWidgetSignal);
        }
      },

      async applyMutations(req) {
        const { mutations, source } = req;
        mutations.forEach((mut) => {
          const op = mut.operation;
          if (!op.case) return;

          switch (op.case) {
            case "addNode":
              if (op.value.node)
                serverGraph.nodes.push(fromProtoNode(op.value.node));
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

        // Add hierarchical mock actions (Frontend will look up schemas by ID)
        actions.push(
          create(ActionTemplateSchema, {
            id: "ai-enhance",
            label: "Enhance",
            path: ["AI Tools"],
            strategy: 1,
          }),
          create(ActionTemplateSchema, {
            id: "prompt-gen",
            label: "Generate from Prompt",
            path: ["AI Tools"],
            strategy: 1,
          }),
          create(ActionTemplateSchema, {
            id: "ai-transform",
            label: "AI Generate (Context Aware)",
            path: ["AI Tools"],
            strategy: 1,
          }),
        );

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

        void (async () => {
          try {
            // 1. Context Collection
            const contextNodeIds = req.contextNodeIds || [];
            const targetNodes = serverGraph.nodes.filter(
              (n) => contextNodeIds.includes(n.id) || n.id === sourceNodeId,
            );
            const contextText = targetNodes
              .map((n) => `[Node ${n.id}]: ${n.data.label}`)
              .join("\n");

            // 2. Action Routing
            if (
              actionId === "ai-transform" ||
              actionId === "ai-enhance" ||
              actionId === "prompt-gen"
            ) {
              const sourceNode = serverGraph.nodes.find(
                (n) => n.id === sourceNodeId,
              );
              if (!sourceNode) return;

              // Determine placement
              const maxY = Math.max(
                ...targetNodes.map(
                  (n) => n.position.y + (n.measured?.height || 200),
                ),
                0,
              );
              const avgX =
                targetNodes.reduce((acc, n) => acc + n.position.x, 0) /
                (targetNodes.length || 1);

              // Initial Status
              eventBus.emit(
                "taskUpdate",
                create(TaskUpdateSchema, {
                  taskId,
                  status: 1,
                  progress: 10,
                  message: "Requesting AI...",
                }),
              );

              // Create placeholder node
              const newNodeId = `ai-${uuidv4().slice(0, 8)}`;
              const newNode = fromProtoNode(
                create(NodeSchema, {
                  nodeId: newNodeId,
                  nodeKind: 1,
                  templateId: "tpl-stream-node",
                  presentation: {
                    position: { x: avgX, y: maxY + 50 },
                    width: 400,
                    height: 300,
                    isInitialized: true,
                  },
                  state: {
                    displayName: `AI Result (${actionId})`,
                    widgetsValuesJson: JSON.stringify({
                      agent_name: "OpenAI",
                      logs: "",
                    }),
                  },
                } as any),
              );

              serverGraph.nodes.push(newNode);
              incrementVersion();
              eventBus.emit(
                "mutations",
                create(MutationListSchema, {
                  mutations: [
                    {
                      operation: {
                        case: "addNode",
                        value: { node: toProtoNode(newNode) },
                      },
                    },
                  ],
                  sequenceNumber: BigInt(serverVersion),
                  source: 2,
                }),
              );

              // 3. OpenAI Streaming
              const stream = await openai.chat.completions.create({
                model: AI_CONFIG.model,
                messages: [
                  {
                    role: "system",
                    content:
                      "You are a Flowcraft assistant. Use the graph context provided to fulfill instructions.",
                  },
                  {
                    role: "user",
                    content: `Graph Context:\n${contextText}\n\nInstruction: ${params.instruction || params.prompt || "Generate insights"}`,
                  },
                ],
                stream: true,
              });

              for await (const chunk of stream) {
                const content = chunk.choices[0]?.delta?.content || "";
                if (content) {
                  eventBus.emit(
                    "streamChunk",
                    create(StreamChunkSchema, {
                      nodeId: newNodeId,
                      widgetId: "logs",
                      chunkData: content,
                      isDone: false,
                    } as any),
                  );
                }
              }

              eventBus.emit(
                "taskUpdate",
                create(TaskUpdateSchema, {
                  taskId,
                  status: 2,
                  progress: 100,
                  message: "AI complete",
                }),
              );
            }
          } catch (err: any) {
            console.error("[AI Execution Error]", err);
            eventBus.emit(
              "taskUpdate",
              create(TaskUpdateSchema, {
                taskId,
                status: 3,
                message: `Error: ${err.message}`,
              }),
            );
          }
        })();

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
            if (presentation.position) {
              node.position = {
                x: presentation.position.x,
                y: presentation.position.y,
              };
            }
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
        // Just an ACK for now, viewport-based filtering can be added later
        return {};
      },

      async cancelTask(req) {
        // Implement task cancellation logic if needed
        console.log(`[Server] Cancel task requested: ${req.taskId}`);
        return {};
      },
    });
  },
});

app.listen({ port: 3000, host: "0.0.0.0" }, (err) => {
  if (err) {
    console.error(err);
    process.exit(1);
  }
  console.log("[Server] Ready with persistence and plugin-registry.");
});
