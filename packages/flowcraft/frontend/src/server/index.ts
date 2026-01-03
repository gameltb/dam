import { fastify } from "fastify";
import { fastifyConnectPlugin } from "@connectrpc/connect-fastify";
import { FlowService } from "../generated/flowcraft/v1/service_pb";
import { create } from "@bufbuild/protobuf";
import {
  FlowMessageSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  TemplateDiscoveryResponseSchema,
  type FlowMessage,
} from "../generated/flowcraft/v1/service_pb";
import {
  ActionDiscoveryResponseSchema,
  ActionTemplateSchema,
} from "../generated/flowcraft/v1/action_pb";
import { MutationSource } from "../generated/flowcraft/v1/base_pb";
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
} from "../generated/flowcraft/v1/node_pb";
import {
  fromProtoNode,
  fromProtoNodeData,
  toProtoNode,
  toProtoEdge,
} from "../utils/protoAdapter";

import { isDynamicNode } from "../types";

const app = fastify();

// 加载持久化数据
loadFromDisk();

// Mock Schema Registry (In a real system, this would be auto-generated from .proto files)
const ACTION_SCHEMAS: Record<string, any> = {
  "prompt-gen": {
    title: "Generate from Prompt",
    type: "object",
    required: ["prompt"],
    properties: {
      prompt: {
        type: "string",
        title: "Positive Prompt",
        description: "What should be in the image",
        default: "A futuristic cityscape",
      },
      negative_prompt: {
        type: "string",
        title: "Negative Prompt",
        description: "What to exclude",
      },
      steps: {
        type: "integer",
        title: "Steps",
        minimum: 1,
        maximum: 100,
        default: 20,
      },
      cfg_scale: {
        type: "number",
        title: "CFG Scale",
        default: 7.5,
      },
    },
  },
  "ai-enhance": {
    title: "AI Enhance",
    type: "object",
    properties: {
      strength: {
        type: "number",
        title: "Enhance Strength",
        minimum: 0,
        maximum: 1,
        default: 0.5,
      },
      model_name: {
        type: "string",
        title: "Model",
        enum: ["Standard", "Realistic", "Anime"],
        default: "Realistic",
      },
    },
  },
};

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

        // Add hierarchical mock actions with schemas
        actions.push(
          create(ActionTemplateSchema, {
            id: "ai-enhance",
            label: "Enhance",
            path: ["AI Tools"],
            strategy: 1,
            paramsSchemaJson: JSON.stringify(ACTION_SCHEMAS["ai-enhance"]),
          }),
          create(ActionTemplateSchema, {
            id: "prompt-gen",
            label: "Generate from Prompt",
            path: ["AI Tools"],
            strategy: 1,
            paramsSchemaJson: JSON.stringify(ACTION_SCHEMAS["prompt-gen"]),
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
        // const params = paramsJson ? JSON.parse(paramsJson) : {};

        // Async execution
        void (async () => {
          // 1. Initial Update
          eventBus.emit(
            "taskUpdate",
            create(TaskUpdateSchema, {
              taskId,
              status: 1, // PROCESSING
              progress: 10,
              message: `Initializing action ${actionId}...`,
            }),
          );

          await new Promise((r) => setTimeout(r, 1000));

          // 2. Specific Logic for AI-Enhance
          if (actionId === "ai-enhance" || actionId === "prompt-gen") {
            const sourceNode = serverGraph.nodes.find(
              (n) => n.id === sourceNodeId,
            );
            if (sourceNode) {
              // Simulate some processing
              eventBus.emit(
                "taskUpdate",
                create(TaskUpdateSchema, {
                  taskId,
                  status: 1,
                  progress: 50,
                  message: "Generating result node...",
                }),
              );

              await new Promise((r) => setTimeout(r, 1500));

              // CREATE A NEW NODE
              const newNodeId = `result-${uuidv4().slice(0, 8)}`;
              const newNode = fromProtoNode(
                create(NodeSchema, {
                  nodeId: newNodeId,
                  nodeKind: 1, // DYNAMIC
                  templateId: "media-img",
                  presentation: {
                    position: {
                      x: sourceNode.position.x + 400,
                      y: sourceNode.position.y,
                    },
                    width: 300,
                    height: 200,
                    isInitialized: true,
                  },
                  state: {
                    displayName: "AI Result",
                    media: {
                      type: 1, // IMAGE
                      url: "https://picsum.photos/id/10/400/300",
                    },
                  },
                } as any),
              );

              serverGraph.nodes.push(newNode);
              incrementVersion();

              // Send mutation to all clients
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
                  source: 2, // SOURCE_REMOTE_TASK
                }),
              );
            }
          }

          // 3. Completion
          eventBus.emit(
            "taskUpdate",
            create(TaskUpdateSchema, {
              taskId,
              status: 2, // COMPLETED
              progress: 100,
              message: "Task finished successfully.",
            }),
          );
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
