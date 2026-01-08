import { TaskStatus } from "@/generated/flowcraft/v1/core/node_pb";
import { BaseInstance } from "./BaseInstance";
import { syncToDB } from "./PersistenceService";

export abstract class NodeInstance extends BaseInstance {
  abstract handleSignal(payload: unknown): Promise<void>;

  async start(params: unknown): Promise<void> {
    this.updateStatus(TaskStatus.TASK_PROCESSING, "Node instance active");
    await this.onReady(params);
  }
  protected flushPersistence(): void {
    console.log(
      `[NodeInstance] Buffered persistence flushing for node: ${this.nodeId ?? "unknown"}`,
    );
    syncToDB();
  }

  protected getInstanceType(): string {
    return "NODE_INSTANCE";
  }

  protected abstract onReady(params: unknown): Promise<void>;
}
