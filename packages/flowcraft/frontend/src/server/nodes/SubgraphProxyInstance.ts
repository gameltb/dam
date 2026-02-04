import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";

import { NodeInstance } from "../services/NodeInstance";

export class SubgraphProxyInstance extends NodeInstance {
  async handleSignal(_payload: unknown): Promise<void> {
    // Handle subgraph related signals
  }

  protected getDisplayLabel(): string {
    return `Subgraph Proxy (${this.nodeId ?? "unknown"})`;
  }

  protected onReady(_params: unknown): Promise<void> {
    this.updateStatus(TaskStatus.RUNNING, "Subgraph Proxy Instance Ready");
    return Promise.resolve();
  }
}
