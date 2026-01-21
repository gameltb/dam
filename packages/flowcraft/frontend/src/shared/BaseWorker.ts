import { fromJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type DbConnection } from "@/generated/spacetime";

import { createNodeDraft, type Draftable, type Result } from "../utils/draft";
import { type PbConnection } from "../utils/pb-client";
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

  /**
   * Returns a Result containing an ORM-style proxy for a node.
   */
  protected nodeDraft(nodeId: string): Result<Draftable<any>> {
    const nodeRow = this.conn.db.nodes.nodeId.find(nodeId);
    if (!nodeRow) return { error: `[Worker] Node ${nodeId} not found`, ok: false };

    return createNodeDraft(nodeId, nodeRow.state, NodeSchema, (path: string, value: unknown) => {
      this.pbClient.pbreducers.pathUpdatePb({
        req: {
          path: path,
          targetId: nodeId,
          type: 0, // REPLACE
          value: fromJson(ValueSchema, value as any),
        } as any,
      });
    });
  }

  protected abstract perform(
    payload: TaskPayloads[Q],
    onProgress: (p: number, msg?: string) => void,
    nodeId: string,
  ): Promise<unknown>;

  private startListening() {
    this.conn.db.tasks.onInsert((_ctx, task) => {
      const status = task.status as { tag: string };
      if (status.tag === "TASK_STATUS_PENDING") {
        void this.tryClaimAndExecute(task);
      }
    });
  }

  private async tryClaimAndExecute(task: any) {
    const nodeId = task.nodeId ?? "";
    this.pbClient.pbreducers.updateTaskStatus({
      update: {
        displayLabel: "",
        message: "Starting...",
        nodeId: nodeId,
        progress: 0,
        result: undefined,
        status: TaskStatus.RUNNING,
        taskId: task.id,
        type: "",
      },
    });

    try {
      const payload = {} as any; // Legacy payload extraction was from task.request.params

      const result = await this.perform(
        payload,
        (progress, message) => {
          this.pbClient.pbreducers.updateTaskStatus({
            update: {
              displayLabel: "",
              message: message ?? "",
              nodeId: nodeId,
              progress,
              result: undefined,
              status: TaskStatus.RUNNING,
              taskId: task.id,
              type: "",
            },
          });
        },
        nodeId,
      );

      this.pbClient.pbreducers.updateTaskStatus({
        update: {
          displayLabel: "",
          message: "Done",
          nodeId: nodeId,
          progress: 100,
          result: {
            kind: {
              case: "structValue",
              value: result as Record<string, unknown>,
            },
          } as any,
          status: TaskStatus.COMPLETED,
          taskId: task.id,
          type: "",
        },
      });
    } catch (error) {
      console.error(`Task ${task.id} failed:`, error);
      const errorMessage = error instanceof Error ? error.message : String(error);
      this.pbClient.pbreducers.updateTaskStatus({
        update: {
          displayLabel: "",
          message: "Failed",
          nodeId: nodeId,
          progress: 0,
          result: {
            kind: {
              case: "stringValue",
              value: errorMessage,
            },
          } as any,
          status: TaskStatus.FAILED,
          taskId: task.id,
          type: "",
        },
      });
    }
  }
}
