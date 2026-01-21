import { WorkerLanguage } from "@/generated/flowcraft/v1/core/kernel_pb";
import { type PbConnection } from "@/utils/pb-client";

import { NodeKernel } from "./NodeKernel";
import { type TaskContext } from "./TaskContext";

export abstract class BaseWorker {
  protected heartbeatInterval: NodeJS.Timeout | null = null;
  protected kernel: NodeKernel;
  protected workerId: string;

  constructor(
    protected conn: PbConnection,
    protected capabilities: string[],
    protected tags: Record<string, string> = {},
  ) {
    this.kernel = new NodeKernel(conn);
    this.workerId = `worker-${process.env.WORKER_TYPE || "generic"}-${crypto.randomUUID()}`;
  }

  abstract handleTask(type: string, ctx: TaskContext): Promise<void>;

  async start() {
    console.log(`[Worker] ${this.workerId} starting...`);

    // 1. Register
    await this.register();

    // 2. Heartbeat
    this.heartbeatInterval = setInterval(() => this.register(), 5000);

    // 3. Subscribe to tasks
    this.conn.db.tasks.onInsert((_ctx, task) => {
      if (task.status.tag === "TASK_STATUS_PENDING" && this.capabilities.includes(task.taskType)) {
        void this.tryClaimAndExecute(task);
      }
    });

    // Check existing pending tasks on startup
    for (const task of this.conn.db.tasks.iter()) {
      if (task.status.tag === "TASK_STATUS_PENDING" && this.capabilities.includes(task.taskType)) {
        void this.tryClaimAndExecute(task);
      }
    }
  }

  stop() {
    if (this.heartbeatInterval) clearInterval(this.heartbeatInterval);
  }

  private async register() {
    await this.conn.pbreducers.registerWorker({
      info: {
        capabilities: this.capabilities,
        lang: WorkerLanguage.WORKER_LANG_TS,
        lastHeartbeat: BigInt(Date.now()),
        tags: this.tags,
        workerId: this.workerId,
      },
    });
  }

  private async tryClaimAndExecute(task: any) {
    try {
      // Atomic claim
      await this.conn.pbreducers.claimTask({
        taskId: task.id,
        workerId: this.workerId,
      });

      console.log(`[Worker] Claimed task: ${task.id} (${task.taskType})`);

      const ctx = this.kernel.createContext(task.id, task.nodeId, task.paramsPayload);
      await this.handleTask(task.taskType, ctx);
    } catch (err: any) {
      if (err.message?.includes("TASK_ALREADY_CLAIMED")) return;
      console.error(`[Worker] Task execution failed: ${task.id}`, err);
    }
  }
}
