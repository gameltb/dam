/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
import type { ConnectRouter } from "@connectrpc/connect";
import { FlowService } from "../generated/core/service_pb";
import { mockEventBus } from "./mockEventBus";
import { serverGraph, incrementVersion, serverVersion } from "./db";
import { actionTemplates } from "./templates";
import { NodeSchema, TaskUpdateSchema } from "../generated/core/node_pb";
import {
  GraphSnapshotSchema,
  MutationListSchema,
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

export const flowServiceImpl = (router: ConnectRouter) => {
  router.service(FlowService, {
    async *watchGraph(_req, ctx) {
      // 1. Send Initial Snapshot
      const snapshotNodes = serverGraph.nodes.map((n) => {
        const width = n.measured?.width ?? (n.style?.width as number);
        const height = n.measured?.height ?? (n.style?.height as number);
        return create(NodeSchema, {
          id: n.id,
          type: n.type,
          position: n.position as any,
          width,
          height,
          selected: !!n.selected,
          parentId: n.parentId,
          data: n.data as any,
        });
      });

      yield create(FlowMessageSchema, {
        messageId: uuidv4(),
        timestamp: BigInt(Date.now()),
        payload: {
          case: "snapshot",
          value: create(GraphSnapshotSchema, {
            nodes: snapshotNodes,
            edges: serverGraph.edges as any,
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
          handler(wrap(m, "mutations"));
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

    updateNode(req, _ctx) {
      const { nodeId, data } = req;
      const node = serverGraph.nodes.find((n) => n.id === nodeId);
      if (node && data) {
        node.data = { ...node.data, ...data };
        incrementVersion();

        const mutationList = create(MutationListSchema, {
          mutations: [
            {
              operation: {
                case: "updateNode",
                value: {
                  id: nodeId,
                  data: node.data as any,
                  position: node.position as any,
                  width: node.measured?.width ?? 0,
                  height: node.measured?.height ?? 0,
                  parentId: node.parentId ?? "",
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
      if (node && node.type === "dynamic" && node.data.widgets && valueJson) {
        const widget = (node.data.widgets as any[]).find(
          (w) => w.id === widgetId,
        );
        if (widget) {
          widget.valueJson = valueJson;
          incrementVersion();
          const mutationList = create(MutationListSchema, {
            mutations: [
              {
                operation: {
                  case: "updateNode",
                  value: {
                    id: nodeId,
                    data: node.data as any,
                    parentId: node.parentId ?? "",
                    width: node.measured?.width ?? 0,
                    height: node.measured?.height ?? 0,
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
      const params = paramsJson ? JSON.parse(paramsJson) : {};
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
              mockEventBus.emit(
                "mutations",
                create(MutationListSchema, {
                  mutations: [
                    {
                      operation: {
                        case: "updateNode",
                        value: {
                          id: sourceNodeId,
                          data: node.data as any,
                          parentId: node.parentId ?? "",
                          width: node.measured?.width ?? 0,
                          height: node.measured?.height ?? 0,
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

    applyMutations(_req, _ctx) {
      return Promise.resolve({});
    },

    cancelTask(_req, _ctx) {
      // Mock cancellation
      return Promise.resolve({});
    },
  });
};
