import { TaskStatus } from "@/generated/flowcraft/v1/core/node_pb";

import { BaseInstance } from "./BaseInstance";

export type ActionHandler = (ctx: { [key: string]: unknown; taskId: string }) => Promise<void>;

export class ActionInstance extends BaseInstance {
  private handler: ActionHandler;
  private label: string;

  constructor(taskId: string, label: string, handler: ActionHandler, nodeId?: string) {
    super(taskId, nodeId);
    this.label = label;
    this.handler = handler;
  }

  async start(params: unknown): Promise<void> {
    this.updateStatus(TaskStatus.TASK_PROCESSING, "Executing action...");
    try {
      await this.handler({
        ...(params as Record<string, unknown>),
        taskId: this.taskId,
      });
      this.updateStatus(TaskStatus.TASK_COMPLETED, "Action completed", 100);
    } catch (err: unknown) {
      this.updateStatus(TaskStatus.TASK_FAILED, err instanceof Error ? err.message : String(err));
    }
  }

  protected flushPersistence(): void {
    // Actions are usually ephemeral, no special persistence needed unless specified
  }

  protected getDisplayLabel(): string {
    return this.label;
  }

  protected getInstanceType(): string {
    return "ACTION";
  }
}
