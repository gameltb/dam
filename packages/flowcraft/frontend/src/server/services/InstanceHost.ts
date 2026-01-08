import { BaseInstance } from "./BaseInstance";

export class InstanceHost {
  private static instance?: InstanceHost;
  private instances = new Map<string, BaseInstance>();
  private nodeToInstances = new Map<string, string[]>();

  private constructor() {
    // Singleton instance
  }

  public static getInstance(): InstanceHost {
    return (InstanceHost.instance ??= new InstanceHost());
  }

  public getAllInstances(): BaseInstance[] {
    return Array.from(this.instances.values());
  }

  public getInstancesForNode(nodeId: string): BaseInstance[] {
    const ids = this.nodeToInstances.get(nodeId) ?? [];
    return ids
      .map((id) => this.instances.get(id))
      .filter((i): i is BaseInstance => i !== undefined);
  }

  public getTask(taskId: string): BaseInstance | undefined {
    return this.instances.get(taskId);
  }

  public registerInstance(instance: BaseInstance) {
    this.instances.set(instance.taskId, instance);
    if (instance.nodeId) {
      const existing = this.nodeToInstances.get(instance.nodeId) ?? [];
      this.nodeToInstances.set(instance.nodeId, [...existing, instance.taskId]);
    }
    console.log(`[InstanceHost] Registered ${instance.taskId}`);
  }

  public async stopAllForNode(nodeId: string) {
    const taskIds = this.nodeToInstances.get(nodeId) ?? [];
    for (const tid of taskIds) {
      await this.stopInstance(tid);
    }
  }

  public async stopInstance(taskId: string) {
    const instance = this.instances.get(taskId);
    if (instance) {
      await instance.stop();
      this.instances.delete(taskId);
      if (instance.nodeId) {
        const tasks = this.nodeToInstances.get(instance.nodeId);
        if (tasks) {
          this.nodeToInstances.set(
            instance.nodeId,
            tasks.filter((id) => id !== taskId),
          );
        }
      }
    }
  }
}

export const instanceHost = InstanceHost.getInstance();
