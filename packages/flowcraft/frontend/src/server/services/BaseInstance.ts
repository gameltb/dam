import { create, toBinary } from "@bufbuild/protobuf";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import {
  TaskStatus,
  TaskUpdateSchema,
} from "@/generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
  MutationListSchema,
} from "@/generated/flowcraft/v1/core/service_pb";

import { executeMutation } from "./MutationExecutor";
import {
  eventBus,
  incrementVersion,
  logMutation,
  serverGraph,
  serverVersion,
} from "./PersistenceService";

export abstract class BaseInstance {
  public readonly nodeId?: string;
  public status: TaskStatus = TaskStatus.TASK_PENDING;
  public readonly taskId: string;

  protected lastUpdateAt = 0;
  private persistenceTimer: NodeJS.Timeout | null = null;

  constructor(taskId: string, nodeId?: string) {
    this.taskId = taskId;
    this.nodeId = nodeId;
  }

  abstract start(params: unknown): Promise<void>;

  async stop(): Promise<void> {
    this.updateStatus(TaskStatus.TASK_CANCELLED, "Instance stopped");
    this.flushPersistence();
    await Promise.resolve();
  }

  protected emitMutation(
    operation: GraphMutation["operation"],
    source: MutationSource = MutationSource.SOURCE_USER,
  ) {
    const mutation = create(GraphMutationSchema, {
      operation: operation,
      originTaskId: this.taskId,
    });

    logMutation(
      mutation.operation.case ?? "unknown",
      toBinary(GraphMutationSchema, mutation),
      source,
    );

    executeMutation(mutation, serverGraph);

    incrementVersion();
    eventBus.emit(
      "mutations",
      create(MutationListSchema, {
        mutations: [mutation],
        sequenceNumber: BigInt(serverVersion),
        source: source,
      }),
    );
  }

  protected abstract flushPersistence(): void;

  protected abstract getDisplayLabel(): string;

  protected abstract getInstanceType(): string;
  protected schedulePersistence(delayMs = 2000) {
    if (this.persistenceTimer) {
      clearTimeout(this.persistenceTimer);
    }
    this.persistenceTimer = setTimeout(() => {
      this.flushPersistence();
    }, delayMs);
  }
  protected updateStatus(status: TaskStatus, message = "", progress = 0) {
    this.status = status;
    const update = create(TaskUpdateSchema, {
      displayLabel: this.getDisplayLabel(),
      message,
      nodeId: this.nodeId ?? "",
      progress,
      status,
      taskId: this.taskId,
      type: this.getInstanceType(),
    });
    eventBus.emit("taskUpdate", update);
  }
}
