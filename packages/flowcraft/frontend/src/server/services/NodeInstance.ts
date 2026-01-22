import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { wrapReducers } from "@/utils/pb-client";

import { getSpacetimeConn } from "../spacetimeClient";
import { BaseInstance } from "./BaseInstance";
import { syncToDB } from "./PersistenceService";

export interface TaskRuntimeContext {
  complete: (result?: any) => Promise<void>;
  log: (message: string, level?: "error" | "info" | "warn") => Promise<void>;
  taskId: string;
  updateProgress: (progress: number, message?: string) => Promise<void>;
}

export abstract class NodeInstance extends BaseInstance {
  abstract handleSignal(payload: unknown): Promise<void>;

  async start(params: unknown): Promise<void> {
    this.updateStatus(TaskStatus.RUNNING, "Node instance active");
    await this.onReady(params);
  }

  protected flushPersistence(): void {
    console.log(`[NodeInstance] Buffered persistence flushing for node: ${this.nodeId ?? "unknown"}`);
    syncToDB();
  }

  protected getInstanceType(): string {
    return "NODE_INSTANCE";
  }

  protected abstract onReady(params: unknown): Promise<void>;

  /**
   * Scaffolding for running a logic block as a tracked SpacetimeDB task.
   * Handles: ID generation, cleanup, registration, and automatic completion/failure.
   */
  protected async runTask(
    actionId: string,
    params: any,
    logic: (ctx: TaskRuntimeContext) => Promise<void>,
  ): Promise<null | string> {
    const conn = getSpacetimeConn();
    if (!conn || !this.nodeId) return null;
    const pbConn = wrapReducers(conn as any);

    const genTaskId = `gen-${this.nodeId}-${crypto.randomUUID().slice(0, 8)}`;

    // Generate idempotency key based on params to prevent double-generation
    const idempotencyKey = `${actionId}-${this.nodeId}-${JSON.stringify(params)}`;

    // 1. Cleanup & Register
    pbConn.pbreducers.clearNodeTasks({ nodeId: this.nodeId });
    pbConn.pbreducers.submitTask({
      idempotencyKey,
      task: {
        createdAt: BigInt(Date.now()),
        nodeId: this.nodeId,
        paramsPayload: new TextEncoder().encode(JSON.stringify(params)),
        taskId: genTaskId,
        taskType: actionId,
      } as any,
    });

    const ctx: TaskRuntimeContext = {
      complete: async (result = "Success") => {
        await pbConn.pbreducers.completeTask({ result: String(result), taskId: genTaskId });
      },
      log: async (message, level = "info") => {
        await pbConn.pbreducers.logTaskEvent({
          log: {
            eventType: level,
            message,
            nodeId: this.nodeId ?? "",
            taskId: genTaskId,
            timestamp: BigInt(Date.now()),
          } as any,
        });
      },
      taskId: genTaskId,
      updateProgress: async (progress, message = "") => {
        await pbConn.pbreducers.updateTaskProgress({
          update: {
            message,
            nodeId: this.nodeId ?? "",
            progress,
            status: TaskStatus.RUNNING,
            taskId: genTaskId,
          } as any,
        });
      },
    };

    try {
      await logic(ctx);
      return genTaskId;
    } catch (err: any) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      await pbConn.pbreducers.failTask({
        error: `Execution Error: ${errorMsg}`,
        taskId: genTaskId,
      });
      throw err;
    }
  }
}
