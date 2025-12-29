/* eslint-disable @typescript-eslint/no-explicit-any */
/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-member-access */
import { v4 as uuidv4 } from "uuid";
import {
  TaskStatus,
  NodeSchema,
  TaskUpdateSchema,
} from "../generated/core/node_pb";
import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
  ActionDiscoveryResponseSchema,
} from "../generated/action_pb";
import {
  type FlowMessage,
  FlowMessageSchema,
  GraphSnapshotSchema,
  MutationListSchema,
  StreamChunkSchema,
} from "../generated/core/service_pb";
import { serverGraph, incrementVersion } from "./db";
import { actionTemplates } from "./templates";
import { create, toJson } from "@bufbuild/protobuf";

export const handleWSMessage = async (
  clientMsg: FlowMessage,

  controller: ReadableStreamDefaultController,
) => {
  const encoder = new TextEncoder();

  const send = (payload: any) => {
    const msg = create(FlowMessageSchema, {
      messageId: uuidv4(),
      timestamp: BigInt(Date.now()),
      payload,
    });

    controller.enqueue(
      encoder.encode(JSON.stringify(toJson(FlowMessageSchema, msg)) + "\n"),
    );
  };

  const payload = clientMsg.payload;

  if (!payload.case) return;

  // 1. Sync

  if (payload.case === "syncRequest") {
    // Ensure nodes have top-level width/height for Protobuf compatibility

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

    send({
      case: "snapshot",

      value: create(GraphSnapshotSchema, {
        nodes: snapshotNodes,

        edges: serverGraph.edges as any,

        version: 0n,
      }),
    });
  }

  // 2. Action Discovery
  if (payload.case === "actionDiscovery") {
    const { selectedNodeIds } = payload.value;
    // Simulate dynamic discovery: if multiple nodes selected, show "Group" actions
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

    // Add path hierarchy to names for ContextMenu parsing
    const mappedActions = filteredActions.map((a) => {
      const path = a.path;
      const label = a.label;
      return create(ActionTemplateSchema, {
        ...a,
        label: [...path, label].join("/"),
      });
    });

    send({
      case: "actions",
      value: create(ActionDiscoveryResponseSchema, {
        actions: mappedActions,
      }),
    });
  }

  // 3. Node Update
  if (payload.case === "nodeUpdate") {
    const { nodeId, data } = payload.value;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && data) {
      node.data = { ...node.data, ...data };
      incrementVersion();
      send({
        case: "mutations",
        value: create(MutationListSchema, {
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
        }),
      });
    }
  }

  // 4. Widget Update
  if (payload.case === "widgetUpdate") {
    const { nodeId, widgetId, valueJson } = payload.value;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && node.type === "dynamic" && node.data.widgets && valueJson) {
      const widget = (node.data.widgets as any[]).find(
        (w) => w.id === widgetId,
      );
      if (widget) {
        widget.valueJson = valueJson;
        incrementVersion();
        send({
          case: "mutations",
          value: create(MutationListSchema, {
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
          }),
        });
      }
    }
  }

  // 5. Actions & Tasks
  if (payload.case === "actionExecute") {
    const { actionId, sourceNodeId, paramsJson } = payload.value;
    const params = paramsJson ? JSON.parse(paramsJson) : {};
    const taskId = (params.taskId as string | undefined) ?? uuidv4();

    if (actionId === "stream" && sourceNodeId) {
      const text = "Connecting... Established. Protocol Active.";
      for (const word of text.split(" ")) {
        send({
          case: "streamChunk",
          value: create(StreamChunkSchema, {
            nodeId: sourceNodeId,
            widgetId: "t1",
            chunkData: `${word} `,
            isDone: false,
          }),
        });
        await new Promise((r) => {
          setTimeout(r, 100);
        });
      }
    } else {
      // Simulate a background task that modifies the graph
      send({
        case: "taskUpdate",
        value: create(TaskUpdateSchema, {
          taskId,
          status: TaskStatus.TASK_PROCESSING,
          progress: 0,
          message: `Action ${actionId} started...`,
          resultJson: "{}",
        }),
      });

      for (let i = 25; i <= 75; i += 25) {
        await new Promise((r) => {
          setTimeout(r, 500);
        });
        send({
          case: "taskUpdate",
          value: create(TaskUpdateSchema, {
            taskId,
            status: TaskStatus.TASK_PROCESSING,
            progress: i,
            message: `Processing ${actionId}...`,
            resultJson: "{}",
          }),
        });
      }

      // Final modification: change node label or color
      if (sourceNodeId) {
        const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);
        if (node) {
          node.data.label = `Processed by ${actionId}`;
          send({
            case: "mutations",
            value: create(MutationListSchema, {
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
          });
        }
      }

      send({
        case: "taskUpdate",
        value: create(TaskUpdateSchema, {
          taskId,
          status: TaskStatus.TASK_COMPLETED,
          progress: 100,
          message: "Action completed successfully.",
          resultJson: "{}",
        }),
      });
    }
  }
};
