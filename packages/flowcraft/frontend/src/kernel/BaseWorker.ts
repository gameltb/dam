import { type Infer } from "spacetimedb";

import { TaskStatus, WorkerLanguage } from "@/generated/flowcraft/v1/core/kernel_pb";
import { type TasksRow } from "@/generated/spacetime";
import { type PbConnection } from "@/utils/pb-client";

import { NodeKernel } from "./NodeKernel";
import { type TaskContext } from "./TaskContext";

type Task = Infer<typeof TasksRow>;

export abstract class BaseWorker {
  protected activeTasks = new Set<string>();
  protected heartbeatInterval: NodeJS.Timeout | null = null;
  protected kernel: NodeKernel;
  protected taskHeartbeatInterval: NodeJS.Timeout | null = null;
  protected workerId: string;

  constructor(
    protected conn: PbConnection,
    protected capabilities: string[],
    protected tags: Record<string, string> = {},
  ) {
    this.kernel = new NodeKernel(conn);
    const workerType = process.env["WORKER_TYPE"] || "generic";
    this.workerId = `worker-${workerType}-${crypto.randomUUID()}`;
  }

  abstract handleTask(type: string, ctx: TaskContext): Promise<void>;

  async start() {
    console.log(`[Worker] ${this.workerId} starting with capabilities: ${this.capabilities.join(", ")}`);

    // 1. Register
    try {
      await this.register();
      console.log(`[Worker] ${this.workerId} registered successfully.`);
    } catch (err) {
      console.error(`[Worker] Registration failed:`, err);
    }

    // 2. Worker Heartbeat
    this.heartbeatInterval = setInterval(() => void this.register(), 5000);

    // 3. Task Heartbeat (Update liveness for active tasks)
    this.taskHeartbeatInterval = setInterval(() => void this.updateActiveTaskHeartbeats(), 10000);

    // 4. Explicit Subscription
    // IMPORTANT: Node.js client needs explicit subscription to receive row events
    console.log(`[Worker] Subscribing to tasks table...`);
    try {
      await this.conn
        .subscriptionBuilder()
        .onApplied(() => {
          console.log(
            `[Worker] Subscription applied. Initial task count: ${Array.from(this.conn.db.tasks.iter()).length}`,
          );

          // Check existing pending tasks after subscription is applied
          for (const task of this.conn.db.tasks.iter()) {
            this.evaluateTask(task);
          }
        })
        .subscribe(["SELECT * FROM tasks"]);
    } catch (err) {
      console.error(`[Worker] Subscription failed:`, err);
    }

    // 5. Subscribe to new tasks
    this.conn.db.tasks.onInsert((_ctx, task) => {
      this.evaluateTask(task);
    });
  }

  stop() {
    if (this.heartbeatInterval) clearInterval(this.heartbeatInterval);
    if (this.taskHeartbeatInterval) clearInterval(this.taskHeartbeatInterval);
  }

  private evaluateTask(task: Task) {
    // SpacetimeDB enums might be wrapped in an object or be a raw number depending on version/gen
    const status = typeof task.status === "number" ? task.status : (task.status as unknown as { value: number }).value;

    console.log(`[Worker] Evaluating task: ${task.id} (${task.taskType}), status: ${status}`);

    if (status === (TaskStatus.PENDING as number) && this.capabilities.includes(task.taskType)) {
      console.log(`[Worker] Matching task found: ${task.id}`);
      void this.tryClaimAndExecute(task);
    }
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

  private async tryClaimAndExecute(task: Task) {
    console.log(`[Worker] Attempting to CLAIM task: ${task.id} for worker ${this.workerId}`);
    try {
      // Atomic claim
      await this.conn.pbreducers.claimTask({
        taskId: task.id,
        workerId: this.workerId,
      });

      console.log(`[Worker] CLAIM SUCCESS: ${task.id} (${task.taskType}). Starting handleTask...`);
      this.activeTasks.add(task.id);

      const ctx = this.kernel.createContext(task.id, task.nodeId, task.paramsPayload);
      await this.handleTask(task.taskType, ctx);

      console.log(`[Worker] COMPLETED handleTask for: ${task.id}`);
      this.activeTasks.delete(task.id);
    } catch (err) {
      this.activeTasks.delete(task.id);
      const errorMessage = err instanceof Error ? err.message : String(err);
      if (errorMessage.includes("TASK_ALREADY_CLAIMED")) {
        console.log(`[Worker] CLAIM CONFLICT: Task ${task.id} already claimed by another worker.`);
        return;
      }
      console.error(`[Worker] EXECUTION ERROR for task ${task.id}:`, err);

      // Fallback: try to mark as failed if we already claimed it but it crashed early
      try {
        await this.conn.pbreducers.failTask({
          error: errorMessage,
          taskId: task.id,
        });
      } catch (failErr) {
        console.error(`[Worker] Double failure when marking task as failed:`, failErr);
      }
    }
  }

  private async updateActiveTaskHeartbeats() {
    for (const taskId of this.activeTasks) {
      try {
        await this.conn.pbreducers.updateTaskProgress({
          update: {
            message: "Active execution (Heartbeat updated)",
            taskId,
          },
        });
      } catch (err) {
        console.error(`[Worker] Failed to update heartbeat for task ${taskId}:`, err);
      }
    }
  }
}
