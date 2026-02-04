import { create, fromBinary, type Message, toJson, toJsonString } from "@bufbuild/protobuf";
import { produce } from "immer";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type PbConnection } from "@/utils/pb-client";

import { type TaskPayloads, type TaskQueue } from "./protocol";
import { type TaskContext } from "./TaskContext";

export class NodeKernel {
  constructor(private conn: PbConnection) {}

  /**
   * Cancels a pending or running task.
   */
  public cancel(taskId: string) {
    this.conn.pbreducers.updateTaskStatus({
      update: {
        displayLabel: "",
        message: "Cancelled by user",
        nodeId: "",
        progress: 0,
        result: undefined,
        status: TaskStatus.CANCELLED,
        taskId,
        type: "",
      },
    });
  }

  /**
   * Pre-execution guard to prevent concurrent tasks on the same node.
   */
  async checkNodeBusy(nodeId: string): Promise<boolean> {
    const tasks = Array.from(this.conn.db.tasks.iter());
    const busy = tasks.some(
      (t) =>
        t.nodeId === nodeId &&
        (t.status === TaskStatus.RUNNING || t.status === TaskStatus.CLAIMED || t.status === TaskStatus.PENDING),
    );
    return busy;
  }

  /**
   * Creates a standardized context for a task.
   */
  createContext(taskId: string, nodeId: string, params: any): TaskContext {
    return {
      complete: async (result) => {
        const resultValue = isProtobufMessage(result) ? toJsonString(result.getType(), result) : JSON.stringify(result);

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
        return task?.status === TaskStatus.CANCELLED;
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
   * Directly edits a node using an Immer recipe, syncing changes to SpacetimeDB.
   */
  editNode(nodeId: string, recipe: (draft: any) => void) {
    const nodeRow = this.conn.db.nodes.nodeId.find(nodeId);
    if (!nodeRow) {
      console.warn(`[Kernel] Node ${nodeId} not found`);
      return;
    }

    const transform = this.conn.db.nodeTransforms.nodeId.find(nodeId);
    const metadata = this.conn.db.nodeMetadata.nodeId.find(nodeId);
    const dataRow = this.conn.db.nodeData.nodeId.find(nodeId);

    const fullState = {
      nodeId,
      nodeKind: nodeRow.nodeKind,
      presentation: {
        height: transform?.height || 0,
        parentId: metadata?.parentId || "",
        position: { x: transform?.x || 0, y: transform?.y || 0 },
        width: transform?.width || 0,
      },
      state: dataRow?.state ? fromBinary(NodeDataSchema, dataRow.state) : create(NodeDataSchema),
      templateId: nodeRow.templateId,
    };

    const nextState = produce(fullState, recipe);

    // Diff and Sync
    const p = fullState.presentation;
    const np = nextState.presentation;

    if (p.position.x !== np.position.x || p.position.y !== np.position.y) {
      this.conn.reducers.setNodePosition({ nodeId, x: np.position.x, y: np.position.y });
    }

    if (p.width !== np.width || p.height !== np.height) {
      this.conn.reducers.setNodeSize({ height: np.height, nodeId, width: np.width });
    }

    if (p.parentId !== np.parentId) {
      this.conn.reducers.setNodeParent({ nodeId, parentId: np.parentId });
    }

    const oldDataJson = JSON.stringify(toJson(NodeDataSchema, fullState.state));
    const newDataJson = JSON.stringify(toJson(NodeDataSchema, nextState.state));

    if (oldDataJson !== newDataJson) {
      this.conn.pbreducers.setNodeDataPb({ nodeId, state: nextState.state });
    }
  }

  /**
   * Submits a new task to the SpacetimeDB task queue.
   */
  public submit<Q extends TaskQueue>(queue: Q, payload: TaskPayloads[Q], nodeId = "") {
    const taskId = crypto.randomUUID();

    this.conn.pbreducers.executeAction({
      id: taskId,
      request: {
        actionId: queue,
        contextNodeIds: [],
        params: {
          case: "paramsStruct",
          value: payload as any, // Payload is strictly typed in protocol but needs to be Struct-compatible
        },
        sourceNodeId: nodeId,
      },
    });

    return taskId;
  }
}

function isProtobufMessage(obj: unknown): obj is Message & { getType: () => any } {
  return !!obj && typeof (obj as any).getType === "function";
}
