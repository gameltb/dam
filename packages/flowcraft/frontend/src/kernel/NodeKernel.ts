import { fromJson, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { createNodeDraft, type Draftable, type Result } from "@/utils/draft";
import { type PbConnection } from "@/utils/pb-client";

import { type TaskContext } from "./TaskContext";

export class NodeKernel {
  constructor(private conn: PbConnection) {}

  /**
   * Pre-execution guard to prevent concurrent tasks on the same node.
   */
  async checkNodeBusy(nodeId: string): Promise<boolean> {
    const tasks = Array.from(this.conn.db.tasks.iter());
    const busy = tasks.some(
      (t) =>
        t.nodeId === nodeId &&
        (t.status.tag === "TASK_STATUS_RUNNING" ||
          t.status.tag === "TASK_STATUS_CLAIMED" ||
          t.status.tag === "TASK_STATUS_PENDING"),
    );
    return busy;
  }

  /**
   * Creates a standardized context for a task.
   */
  createContext(taskId: string, nodeId: string, params: any): TaskContext {
    return {
      complete: async (result) => {
        const resultValue = isProtobufMessage(result)
          ? JSON.stringify(toJson((result as any).getType(), result as any))
          : JSON.stringify(result);

        await this.conn.pbreducers.completeTask({
          result: resultValue,
          taskId,
        });
      },
      config: {}, // To be populated from node state if needed
      fail: async (error) => {
        await this.conn.pbreducers.failTask({
          error,
          taskId,
        });
      },
      isCancelled: () => {
        const task = this.conn.db.tasks.id.find(taskId);
        return task?.status.tag === "TASK_STATUS_CANCELLED";
      },
      log: async (message, level = "info") => {
        await this.conn.pbreducers.logTaskEvent({
          log: {
            eventType: level,
            message,
            nodeId,
            taskId,
            timestamp: BigInt(Date.now()),
          },
        });
      },
      nodeId,
      params,
      taskId,
      updateProgress: async (percentage, message) => {
        await this.conn.pbreducers.updateTaskProgress({
          update: {
            message: message || "",
            progress: percentage,
            status: TaskStatus.RUNNING,
            taskId,
          },
        });
      },
    };
  }

  /**
   * Returns a Result containing an ORM-style proxy for a node.
   */
  nodeDraft(nodeId: string): Result<Draftable<any>> {
    const nodeRow = this.conn.db.nodes.nodeId.find(nodeId);
    if (!nodeRow) return { error: `[Kernel] Node ${nodeId} not found`, ok: false };

    return createNodeDraft(nodeId, nodeRow.state, NodeSchema, (path: string, value: unknown) => {
      this.conn.pbreducers.pathUpdatePb({
        req: {
          path: path,
          targetId: nodeId,
          type: 0, // REPLACE
          value: fromJson(ValueSchema, value as any),
        } as any,
      });
    });
  }
}

function isProtobufMessage(obj: any): obj is { getType: () => any } {
  return obj && typeof obj.getType === "function";
}
