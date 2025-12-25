import { v4 as uuidv4 } from "uuid";
import { TaskStatus } from "../types";
import { serverGraph, incrementVersion } from "./db";
import { flowcraft } from "../generated/flowcraft";

export const handleWSMessage = async (
  clientMsg: flowcraft.v1.IFlowMessage,
  controller: ReadableStreamDefaultController,
) => {
  const encoder = new TextEncoder();
  const send = (payload: flowcraft.v1.IFlowMessage) => {
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
    send({ snapshot: serverGraph as unknown as flowcraft.v1.IGraphSnapshot });
  }

  // 2. Node Update
  if (clientMsg.nodeUpdate) {
    const { nodeId, data } = clientMsg.nodeUpdate;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && data) {
      node.data = { ...node.data, ...data } as any; // eslint-disable-line @typescript-eslint/no-explicit-any
      incrementVersion();
      send({
        mutations: {
          mutations: [
            {
              updateNode: {
                id: nodeId,
                data: node.data as unknown as flowcraft.v1.INodeData,
                position: node.position,
              },
            },
          ],
        },
      });
    }
  }

  // 3. Widget Update
  if (clientMsg.widgetUpdate) {
    const { nodeId, widgetId, valueJson } = clientMsg.widgetUpdate;
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (node && node.type === "dynamic" && node.data.widgets && valueJson) {
      const widget = (node.data.widgets as flowcraft.v1.IWidget[]).find(
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
                  data: node.data as unknown as flowcraft.v1.INodeData,
                },
              },
            ],
          },
        });
      }
    }
  }

  // 4. Actions & Tasks
  if (clientMsg.actionExecute) {
    const { actionId, sourceNodeId } = clientMsg.actionExecute;
    if (actionId === "stream" && sourceNodeId) {
      const text = "Connecting... Established. Protocol Active.";
      for (const word of text.split(" ")) {
        send({
          streamChunk: {
            nodeId: sourceNodeId,
            widgetId: "t1",
            chunkData: word + " ",
            isDone: false,
          },
        });
        await new Promise((r) => setTimeout(r, 100));
      }
    } else {
      const taskId = uuidv4();
      send({
        taskUpdate: {
          taskId,
          status: TaskStatus.TASK_PROCESSING,
          progress: 0,
          message: "Task initialized.",
        },
      });
      for (let i = 25; i <= 100; i += 25) {
        await new Promise((r) => setTimeout(r, 400));
        send({
          taskUpdate: {
            taskId,
            status: TaskStatus.TASK_PROCESSING,
            progress: i,
            message: `Step ${i / 25}/4`,
          },
        });
      }
      send({
        taskUpdate: {
          taskId,
          status: TaskStatus.TASK_COMPLETED,
          progress: 100,
          message: "Finished",
        },
      });
    }
  }
};
