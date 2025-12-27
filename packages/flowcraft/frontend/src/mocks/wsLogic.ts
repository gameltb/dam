import { v4 as uuidv4 } from "uuid";
import { TaskStatus } from "../types";
import { serverGraph, incrementVersion } from "./db";
import { flowcraft_proto } from "../generated/flowcraft_proto";
import { actionTemplates } from "./templates";

export const handleWSMessage = async (
  clientMsg: flowcraft_proto.v1.IFlowMessage,
  controller: ReadableStreamDefaultController,
) => {
  const encoder = new TextEncoder();
  const send = (payload: flowcraft_proto.v1.IFlowMessage) => {
    const msg = {
      messageId: uuidv4(),
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      timestamp: Date.now() as any,
      ...payload,
    };
    controller.enqueue(encoder.encode(JSON.stringify(msg) + "\n"));
  };

  // 1. Sync
  if (clientMsg.syncRequest) {
    // Ensure nodes have top-level width/height for Protobuf compatibility
    const snapshotNodes = serverGraph.nodes.map((n) => ({
      ...n,
      width: n.measured?.width ?? (n.style?.width as number | undefined) ?? 300,
      height:
        n.measured?.height ?? (n.style?.height as number | undefined) ?? 200,
    }));

    send({
      snapshot: {
        nodes: snapshotNodes as flowcraft_proto.v1.INode[],
        edges: serverGraph.edges as flowcraft_proto.v1.IEdge[],
        version: 0 as unknown as number,
      },
    });
  }

  // 2. Action Discovery
  if (clientMsg.actionDiscovery) {
    const { selectedNodeIds } = clientMsg.actionDiscovery;
    // Simulate dynamic discovery: if multiple nodes selected, show "Group" actions
    const filteredActions = [...actionTemplates];
    if (selectedNodeIds && selectedNodeIds.length > 1) {
      filteredActions.push({
        id: "batch-process",
        label: "Process Group",
        path: ["Batch"],
        strategy: flowcraft_proto.v1.ActionExecutionStrategy.EXECUTION_TASK,
      });
    }

    // Add path hierarchy to names for ContextMenu parsing
    const mappedActions = filteredActions.map((a) => ({
      ...a,
      label: [...(a.path ?? []), a.label ?? ""].join("/"),
    }));

    send({
      actions: {
        actions: mappedActions as flowcraft_proto.v1.IActionTemplate[],
      },
    });
  }

  // 3. Node Update
  if (clientMsg.nodeUpdate) {
    const { nodeId, data } = clientMsg.nodeUpdate;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && data) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment
      node.data = { ...node.data, ...data } as any;
      incrementVersion();
      send({
        mutations: {
          mutations: [
            {
              updateNode: {
                id: nodeId,
                data: node.data as unknown as flowcraft_proto.v1.INodeData,
                position: node.position,
              },
            },
          ],
        },
      });
    }
  }

  // 4. Widget Update
  if (clientMsg.widgetUpdate) {
    const { nodeId, widgetId, valueJson } = clientMsg.widgetUpdate;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && node.type === "dynamic" && node.data.widgets && valueJson) {
      const widget = (node.data.widgets as flowcraft_proto.v1.IWidget[]).find(
        (w) => w.id === widgetId,
      );
      if (widget) {
        widget.valueJson = valueJson;
        incrementVersion();
        send({
          mutations: {
            mutations: [
              {
                updateNode: {
                  id: nodeId,
                  data: node.data as unknown as flowcraft_proto.v1.INodeData,
                },
              },
            ],
          },
        });
      }
    }
  }

  // 5. Actions & Tasks
  if (clientMsg.actionExecute) {
    const { actionId, sourceNodeId, paramsJson } = clientMsg.actionExecute;
    // eslint-disable-next-line @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment
    const params = (paramsJson ? JSON.parse(paramsJson) : {}) as any;
    // eslint-disable-next-line @typescript-eslint/no-unsafe-member-access
    const taskId = (params.taskId as string | undefined) ?? uuidv4();

    if (actionId === "stream" && sourceNodeId) {
      const text = "Connecting... Established. Protocol Active.";
      for (const word of text.split(" ")) {
        send({
          streamChunk: {
            nodeId: sourceNodeId,
            widgetId: "t1",
            chunkData: `${word} `,
            isDone: false,
          },
        });
        await new Promise((r) => {
          setTimeout(r, 100);
        });
      }
    } else {
      // Simulate a background task that modifies the graph
      send({
        taskUpdate: {
          taskId,
          status: TaskStatus.TASK_PROCESSING,
          progress: 0,
          message: `Action ${actionId ?? ""} started...`,
        },
      });

      for (let i = 25; i <= 75; i += 25) {
        await new Promise((r) => {
          setTimeout(r, 500);
        });
        send({
          taskUpdate: {
            taskId,
            status: TaskStatus.TASK_PROCESSING,
            progress: i,
            message: `Processing ${actionId ?? ""}...`,
          },
        });
      }

      // Final modification: change node label or color
      if (sourceNodeId) {
        const node = serverGraph.nodes.find((n) => n.id === sourceNodeId);
        if (node) {
          node.data.label = `Processed by ${actionId ?? ""}`;
          send({
            mutations: {
              mutations: [
                {
                  updateNode: {
                    id: sourceNodeId,
                    data: node.data as flowcraft_proto.v1.INodeData,
                  },
                },
              ],
            },
          });
        }
      }

      send({
        taskUpdate: {
          taskId,
          status: TaskStatus.TASK_COMPLETED,
          progress: 100,
          message: "Action completed successfully.",
        },
      });
    }
  }
};