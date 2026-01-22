import { wrapReducers } from "@/utils/pb-client";

import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";

const HEARTBEAT_TIMEOUT_MS = 30000; // 30 seconds

/**
 * Service responsible for self-healing and maintenance of the distributed kernel.
 *
 * Functions:
 * 1. Ghost Detection: Finds tasks that are CLAIMED or RUNNING but haven't updated heartbeats.
 * 2. Task Recovery: Resets or fails stale tasks to unblock nodes.
 */
export class DurableWorkflowService {
  private static timer: NodeJS.Timeout | null = null;

  public static start() {
    if (this.timer) return;

    logger.info("[DurableWorkflow] Self-healing monitor started.");
    this.timer = setInterval(() => this.checkStaleTasks(), 10000); // Check every 10s
  }

  public static stop() {
    if (this.timer) clearInterval(this.timer);
    this.timer = null;
  }

  private static async checkStaleTasks() {
    const conn = getSpacetimeConn();
    if (!conn) return;

    const now = Date.now();
    const pbConn = wrapReducers(conn as any);
    const tasks = Array.from(conn.db.tasks.iter());

    for (const task of tasks) {
      const status = (task.status as any).tag || task.status;

      if (status === "TASK_STATUS_CLAIMED" || status === "TASK_STATUS_RUNNING") {
        const lastHeartbeat = Number(task.lastHeartbeat);
        const age = now - lastHeartbeat;

        if (age > HEARTBEAT_TIMEOUT_MS) {
          logger.warn(`[DurableWorkflow] Task ${task.id} is STALE (age: ${age}ms). Mark as failed.`);

          try {
            await pbConn.pbreducers.failTask({
              error: "Workflow Timeout: Worker stopped responding (Heartbeat lost)",
              taskId: task.id,
            });
          } catch (err) {
            logger.error(`[DurableWorkflow] Failed to cleanup stale task ${task.id}:`, err);
          }
        }
      }
    }
  }
}
