import { create } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import { TaskStatus, TaskUpdateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type PbConnection } from "../utils/pb-client";
import { type DbConnection } from "@/generated/spacetime";

import { type TaskPayloads, type TaskQueue } from "./task-protocol";

export abstract class BaseWorker<Q extends TaskQueue> {
  protected pbClient: PbConnection;

  constructor(
    protected conn: DbConnection,
    protected queueType: Q,
    pbClient?: PbConnection,
  ) {
    this.pbClient = pbClient!;
    this.startListening();
  }

  protected abstract perform(payload: TaskPayloads[Q], onProgress: (p: number, msg?: string) => void): Promise<unknown>;

  private startListening() {
    this.conn.db.tasks.onInsert((_ctx, task) => {
      const status = task.status as { tag: string };
      if (task.request?.actionId === this.queueType && status.tag === "TASK_PENDING") {
        void this.tryClaimAndExecute(task);
      }
    });
  }

  private async tryClaimAndExecute(task: any) {
    this.pbClient.pbreducers.updateTaskStatus({
      update: create(TaskUpdateSchema, {
        displayLabel: "",
        message: "Starting...",
        nodeId: task.request?.sourceNodeId ?? "",
        progress: 0,
        result: undefined,
        status: TaskStatus.TASK_PROCESSING,
        taskId: task.id,
        type: "",
      }),
    });

    try {
      const payload = task.request?.params?.value as TaskPayloads[Q];

      const result = await this.perform(payload, (progress, message) => {
        this.pbClient.pbreducers.updateTaskStatus({
          update: create(TaskUpdateSchema, {
            displayLabel: "",
            message: message ?? "",
            nodeId: task.request?.sourceNodeId ?? "",
            progress,
            result: undefined,
            status: TaskStatus.TASK_PROCESSING,
            taskId: task.id,
            type: "",
          }),
        });
      });

      this.pbClient.pbreducers.updateTaskStatus({
        update: create(TaskUpdateSchema, {
          displayLabel: "",
          message: "Done",
          nodeId: task.request?.sourceNodeId ?? "",
          progress: 100,
          result: create(ValueSchema, {
            kind: {
              case: "structValue",
              value: result as Record<string, unknown>,
            },
          }),
          status: TaskStatus.TASK_COMPLETED,
          taskId: task.id,
          type: "",
        }),
      });
    } catch (error) {
      console.error(`Task ${task.id} failed:`, error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.pbClient.pbreducers.updateTaskStatus({
        update: create(TaskUpdateSchema, {
          displayLabel: "",
          message: "Failed",
          nodeId: task.request?.sourceNodeId ?? "",
          progress: 0,
          result: create(ValueSchema, {
            kind: {
              case: "stringValue",
              value: errorMessage,
            },
          }),
          status: TaskStatus.TASK_FAILED,
          taskId: task.id,
          type: "",
        }),
      });
    }
  }
}
