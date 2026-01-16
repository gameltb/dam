import { type PbConnection } from "../../utils/pb-client";

import logger from "../utils/logger";

export const initTaskWatcher = () => {
  // Task watching logic
  logger.info("Task watcher initialized");
};

export class TaskService {
  constructor(private conn: PbConnection) {}

  public async getTask(taskId: string) {
    return this.conn.db.tasks.id.find(taskId);
  }

  public async updateStatus(taskId: string, status: string, result: string) {
    this.conn.pbreducers.updateTaskStatus({
      update: {
        displayLabel: "",
        message: status,
        nodeId: "",
        progress: 100,
        result: { kind: { case: "stringValue", value: result } } as any,
        status: status as any,
        taskId,
        type: "",
      } as any,
    });
  }
}
