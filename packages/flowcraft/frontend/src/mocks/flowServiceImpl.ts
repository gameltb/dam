/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment */
import type { ConnectRouter } from "@connectrpc/connect";
import { FlowService } from "../generated/core/service_pb";
import { mockEventBus } from "./mockEventBus";
import { serverGraph, incrementVersion, serverVersion } from "./db";
import { actionTemplates } from "./templates";
import { TaskUpdateSchema } from "../generated/core/node_pb";
import {
  GraphSnapshotSchema,
  MutationListSchema,
  type MutationList,
  StreamChunkSchema,
  FlowMessageSchema,
  type FlowMessage,
} from "../generated/core/service_pb";
import {
  ActionDiscoveryResponseSchema,
  ActionTemplateSchema,
  ActionExecutionStrategy,
} from "../generated/action_pb";
import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";
import { TaskStatus } from "../types";
import {
  fromProtoEdge,
  fromProtoNode,
  fromProtoNodeData,
  toProtoEdge,
  toProtoNode,
} from "../utils/protoAdapter";

// Mock Session State (Single User for now)
let currentViewport = { x: -5000, y: -5000, width: 10000, height: 10000 };

const isNodeInViewport = (
  node: {
    position: { x: number; y: number };
    measured?: { width?: number; height?: number };
    width?: number;
    height?: number;
  },
  viewport: { x: number; y: number; width: number; height: number },
) => {
  const nodeW = node.measured?.width ?? node.width ?? 200;
  const nodeH = node.measured?.height ?? node.height ?? 200;

  return (
    node.position.x < viewport.x + viewport.width &&
    node.position.x + nodeW > viewport.x &&
    node.position.y < viewport.y + viewport.height &&
    node.position.y + nodeH > viewport.y
  );
};

export const flowServiceImpl = (router: ConnectRouter) => {
  router.service(FlowService, {
    async *watchGraph(_req, ctx) {
      // 1. Send Initial Snapshot (Only visible nodes to start, or just a safe default area)
      // For a better "lazy" experience, we start with what's in the default/current viewport
      const visibleNodes = serverGraph.nodes.filter((n) =>
        isNodeInViewport(n, currentViewport),
      );
      const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));

      // Include edges only if both source/target are visible (or maybe just one?)
      // To prevent "dangling edges", we ideally only show edges if both are there,
      // OR we allow edges to "unknown" nodes (Ghost Nodes).
      // For now: Only if both are visible.
      const visibleEdges = serverGraph.edges.filter(
        (e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target),
      );

      const snapshotNodes = visibleNodes.map(toProtoNode);
      const snapshotEdges = visibleEdges.map(toProtoEdge);

      yield create(FlowMessageSchema, {
        messageId: uuidv4(),
        timestamp: BigInt(Date.now()),
        payload: {
          case: "snapshot",
          value: create(GraphSnapshotSchema, {
            nodes: snapshotNodes,
            edges: snapshotEdges,
            version: BigInt(serverVersion),
          }),
        },
      });

      // 2. Setup Event Listening
      const queue: FlowMessage[] = [];
      let resolveQueue: (() => void) | null = null;

      const handler = (msg: FlowMessage) => {
        queue.push(msg);
        resolveQueue?.();
      };

      const wrap = (payload: any, caseName: string) => {
        return create(FlowMessageSchema, {
          messageId: uuidv4(),
          timestamp: BigInt(Date.now()),
          payload: { case: caseName as any, value: payload },
        });
      };

      const unsubs = [
        mockEventBus.on("mutations", (m) => {
          // Filter mutations based on visibility
          const mutationList = m as MutationList;
          const filteredMutations = mutationList.mutations.filter((mut) => {
            const op = mut.operation;
            if (op.case === "updateNode") {
              // Check if node is in viewport
              const node = serverGraph.nodes.find((n) => n.id === op.value.id);
              if (node) return isNodeInViewport(node, currentViewport);
            }
            // Always allow adds/removes or handle specifically?
            // For AddNode, check its position
            if (op.case === "addNode" && op.value.node) {
              const n = op.value.node;
              // Mock check using proto position
              const posX = n.position ? n.position.x : 0;
              const posY = n.position ? n.position.y : 0;
              const nodeW = n.width || 200;
              const nodeH = n.height || 200;

              return (
                posX < currentViewport.x + currentViewport.width &&
                posX + nodeW > currentViewport.x &&
                posY < currentViewport.y + currentViewport.height &&
                posY + nodeH > currentViewport.y
              );
            }
            return true; // Allow other mutations (edges, etc) for now
          });

          if (filteredMutations.length > 0) {
            // Re-wrap in mutation list
            handler(
              wrap(
                create(MutationListSchema, {
                  mutations: filteredMutations,
                  sequenceNumber: mutationList.sequenceNumber,
                }),
                "mutations",
              ),
            );
          }
        }),
        mockEventBus.on("taskUpdate", (t) => {
          handler(wrap(t, "taskUpdate"));
        }),
        mockEventBus.on("streamChunk", (c) => {
          handler(wrap(c, "streamChunk"));
        }),
        mockEventBus.on("widgetSignal", (s) => {
          handler(wrap(s, "widgetSignal"));
        }),
      ];

      try {
        while (!ctx.signal.aborted) {
          const next = queue.shift();
          if (next) {
            yield next;
          } else {
            await new Promise<void>((r) => {
              resolveQueue = r;
            });
            resolveQueue = null;
          }
        }
      } finally {
        unsubs.forEach((u) => {
          u();
        });
      }
    },

    updateViewport(req, _ctx) {
      if (req.visibleBounds) {
        currentViewport = req.visibleBounds;

        // Find nodes in the new viewport that might be missing on client
        // In a real impl, we'd diff with client state. Here we just blindly send "AddNode" for everything in view.
        // The client (Yjs/Zustand) handles the "Upsert" logic naturally.

        const visibleNodes = serverGraph.nodes.filter((n) =>
          isNodeInViewport(n, currentViewport),
        );
        const visibleNodeIds = new Set(visibleNodes.map((n) => n.id));

        const visibleEdges = serverGraph.edges.filter(
          (e) => visibleNodeIds.has(e.source) && visibleNodeIds.has(e.target),
        );

        const mutations = [
          ...visibleNodes.map((n) => ({
            operation: {
              case: "addNode" as const,
              value: { node: toProtoNode(n) },
            },
          })),
          ...visibleEdges.map((e) => ({
            operation: {
              case: "addEdge" as const,
              value: { edge: toProtoEdge(e) },
            },
          })),
        ];

        // We broadcast this ONLY to the requesting client usually.
        // But since we are using a shared EventBus for this mock, this will broadcast to ALL.
        // To fix this properly, we should push directly to the stream queue of this client.
        // However, 'ctx' from watchGraph is not available here.
        // For this Prototype: We emit to global bus, but because of the 'isNodeInViewport' filter in watchGraph,
        // it acts correctly! (The filter uses global currentViewport, which we just updated).
        // Wait... if multiple clients exist, global currentViewport is race-condition prone.
        // Assumption: Single User for this demo.

        if (mutations.length > 0) {
          mockEventBus.emit(
            "mutations",
            create(MutationListSchema, {
              mutations: mutations,
              sequenceNumber: BigInt(serverVersion),
            }),
          );
        }
      }
      return Promise.resolve({});
    },

    updateNode(req, _ctx) {
      const { nodeId, data } = req;
      const node = serverGraph.nodes.find((n) => n.id === nodeId);
      if (node && data) {
        // Convert Protobuf partial data to AppNode data
        const partialAppData = fromProtoNodeData(data);
        node.data = { ...node.data, ...partialAppData };
        incrementVersion();

        // Convert back to proto for broadcast
        const protoNode = toProtoNode(node);

        const mutationList = create(MutationListSchema, {
          mutations: [
            {
              operation: {
                case: "updateNode",
                value: {
                  id: nodeId,
                  data: protoNode.data,
                  position: protoNode.position,
                  width: protoNode.width,
                  height: protoNode.height,
                  parentId: protoNode.parentId,
                },
              },
            },
          ],
          sequenceNumber: 0n,
        });
        mockEventBus.emit("mutations", mutationList);
      }
      return Promise.resolve({});
    },

    updateWidget(req, _ctx) {
      const { nodeId, widgetId, valueJson } = req;
      const node = serverGraph.nodes.find((n) => n.id === nodeId);
      if (node && node.type === "dynamic" && node.data.widgets) {
        const widget = node.data.widgets.find((w) => w.id === widgetId);
        if (widget) {
          widget.value = JSON.parse(valueJson);
          incrementVersion();

          const protoNode = toProtoNode(node);

          const mutationList = create(MutationListSchema, {
            mutations: [
              {
                operation: {
                  case: "updateNode",
                  value: {
                    id: nodeId,
                    data: protoNode.data,
                    parentId: protoNode.parentId,
                    width: protoNode.width,
                    height: protoNode.height,
                  },
                },
              },
            ],
            sequenceNumber: 0n,
          });
          mockEventBus.emit("mutations", mutationList);
        }
      }
      return Promise.resolve({});
    },

    sendWidgetSignal(req, _ctx) {
      mockEventBus.emit("widgetSignal", req);
      return Promise.resolve({});
    },

    discoverActions(req, _ctx) {
      const { selectedNodeIds } = req;
      const filteredActions = [...actionTemplates];
      if (selectedNodeIds.length > 1) {
        filteredActions.push(
          create(ActionTemplateSchema, {
            id: "batch-process",
            label: "Process Group",
            path: ["Batch"],
            strategy: ActionExecutionStrategy.EXECUTION_BACKGROUND,
          }),
        );
      }

      const mappedActions = filteredActions.map((a) => {
        const path = a.path;
        const label = a.label;
        return create(ActionTemplateSchema, {
          ...a,
          label: [...path, label].join("/"),
        });
      });

      return Promise.resolve(
        create(ActionDiscoveryResponseSchema, {
          actions: mappedActions,
        }),
      );
    },

    executeAction(req, _ctx) {
      const { actionId, sourceNodeId, paramsJson } = req;
      const params = (paramsJson ? JSON.parse(paramsJson) : {}) as Record<
        string,
        unknown
      >;
      const taskId = (params.taskId as string | undefined) ?? uuidv4();

      // Trigger background logic
      void (async () => {
        if (actionId === "stream" && sourceNodeId) {
          const text = "Connecting... Established. Protocol Active.";
          for (const word of text.split(" ")) {
            mockEventBus.emit(
              "streamChunk",
              create(StreamChunkSchema, {
                nodeId: sourceNodeId,
                widgetId: "t1",
                chunkData: `${word} `,
                isDone: false,
              }),
            );
            await new Promise((r) => setTimeout(r, 100));
          }
        } else {
          mockEventBus.emit(
            "taskUpdate",
            create(TaskUpdateSchema, {
              taskId,
              status: TaskStatus.TASK_PROCESSING,
              progress: 0,
              message: `Action ${actionId} started...`,
              resultJson: "{}",
            }),
          );

          for (let i = 25; i <= 75; i += 25) {
            await new Promise((r) => setTimeout(r, 500));
            mockEventBus.emit(
              "taskUpdate",
              create(TaskUpdateSchema, {
                taskId,
                status: TaskStatus.TASK_PROCESSING,
                progress: i,
                message: `Processing ${actionId}...`,
                resultJson: "{}",
              }),
            );
          }

          if (sourceNodeId) {
            const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);
            if (node) {
              node.data.label = `Processed by ${actionId}`;

              const protoNode = toProtoNode(node);

              mockEventBus.emit(
                "mutations",
                create(MutationListSchema, {
                  mutations: [
                    {
                      operation: {
                        case: "updateNode",
                        value: {
                          id: sourceNodeId,
                          data: protoNode.data,
                          parentId: protoNode.parentId,
                          width: protoNode.width,
                          height: protoNode.height,
                        },
                      },
                    },
                  ],
                  sequenceNumber: 0n,
                }),
              );
            }
          }

          mockEventBus.emit(
            "taskUpdate",
            create(TaskUpdateSchema, {
              taskId,
              status: TaskStatus.TASK_COMPLETED,
              progress: 100,
              message: "Action completed successfully.",
              resultJson: "{}",
            }),
          );
        }
      })();

      return Promise.resolve({});
    },

    applyMutations(req, _ctx) {
      const { mutations } = req;
      if (mutations.length === 0) return Promise.resolve({});

      mutations.forEach((mut) => {
        const op = mut.operation;
        if (!op.case) return;

        switch (op.case) {
          case "addNode": {
            const protoNode = op.value.node;
            if (protoNode) {
              serverGraph.nodes.push(fromProtoNode(protoNode));
            }
            break;
          }
          case "updateNode": {
            const val = op.value;
            const node = serverGraph.nodes.find((n) => n.id === val.id);
            if (node) {
              if (val.position) {
                node.position = { x: val.position.x, y: val.position.y };
              }
              if (val.data) {
                const appData = fromProtoNodeData(val.data);
                // Merge strategies could vary, here we do shallow merge
                node.data = { ...node.data, ...appData };
              }
              if (val.width) {
                node.measured = {
                  width: val.width,
                  height: val.height || (node.measured?.height ?? 0),
                };
                node.style = { ...node.style, width: val.width };
              }
              if (val.height) {
                node.measured = {
                  width: val.width || (node.measured?.width ?? 0),
                  height: val.height,
                };
                node.style = { ...node.style, height: val.height };
              }
              if (val.parentId !== "")
                node.parentId = val.parentId || undefined;
            }
            break;
          }
          case "removeNode": {
            const id = op.value.id;
            serverGraph.nodes = serverGraph.nodes.filter((n) => n.id !== id);
            break;
          }
          case "addEdge": {
            const edge = op.value.edge;
            if (edge) serverGraph.edges.push(fromProtoEdge(edge));
            break;
          }
          case "removeEdge": {
            const id = op.value.id;
            serverGraph.edges = serverGraph.edges.filter((e) => e.id !== id);
            break;
          }
          case "clearGraph": {
            serverGraph.nodes = [];
            serverGraph.edges = [];
            break;
          }
        }
      });

      incrementVersion();
      // Broadcast the mutations to all connected clients
      mockEventBus.emit(
        "mutations",
        create(MutationListSchema, {
          mutations,
          sequenceNumber: BigInt(serverVersion),
        }),
      );

      return Promise.resolve({});
    },

    cancelTask(req, _ctx) {
      const { taskId } = req;
      mockEventBus.emit(
        "taskUpdate",
        create(TaskUpdateSchema, {
          taskId,
          status: TaskStatus.TASK_FAILED,
          message: "Task cancelled by user.",
        }),
      );
      return Promise.resolve({});
    },
  });
};
