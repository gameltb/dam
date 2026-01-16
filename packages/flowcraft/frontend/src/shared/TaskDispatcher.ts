import { type PbConnection } from "../utils/pb-client";

import { type TaskPayloads, type TaskQueue } from "./task-protocol";

export class TaskDispatcher {
  constructor(private client: PbConnection) {}

  public cancel(taskId: string) {
    this.client.pbreducers.updateTaskStatus({
      update: {
        displayLabel: "",
        message: "Cancelled by user",
        nodeId: "",
        progress: 0,
        result: undefined,
        status: 4, // TASK_CANCELLED
        taskId,
        type: "",
      } as any,
    });
  }

  public submit<Q extends TaskQueue>(queue: Q, payload: TaskPayloads[Q], nodeId = "") {
    const taskId = crypto.randomUUID();

    this.client.pbreducers.executeAction({
      id: taskId,
      request: {
        actionId: queue,
        contextNodeIds: [],
        params: {
          case: "paramsStruct",
          value: payload as any,
        },
        sourceNodeId: nodeId,
      } as any,
    });

    return taskId;
  }
}
