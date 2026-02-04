import { create, fromJson, toBinary } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { produceWithPatches } from "immer";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import { TaskStatus, TaskUpdateSchema } from "@/generated/flowcraft/v1/core/kernel_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
  MutationListSchema,
  PathUpdateRequest_UpdateType,
  PathUpdateRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";

import { executeMutation } from "./MutationExecutor";
import { eventBus, incrementVersion, logMutation, serverGraph, serverVersion } from "./PersistenceService";

export abstract class BaseInstance {
  public readonly nodeId?: string;
  public status: TaskStatus = TaskStatus.PENDING;
  public readonly taskId: string;

  protected lastUpdateAt = 0;
  private persistenceTimer: NodeJS.Timeout | null = null;

  constructor(taskId: string, nodeId?: string) {
    this.taskId = taskId;
    this.nodeId = nodeId;
  }

  abstract start(params: unknown): Promise<void>;

  async stop(): Promise<void> {
    this.updateStatus(TaskStatus.CANCELLED, "Instance stopped");
    this.flushPersistence();
    await Promise.resolve();
  }

  protected editNode(nodeId: string, recipe: (draft: any) => void) {
    const node = serverGraph.nodes.find((n) => n.id === nodeId);
    if (!node) {
      console.warn(`[BaseInstance] Node ${nodeId} not found`);
      return;
    }

    const [_, patches] = produceWithPatches(node, recipe);

    patches.forEach((patch) => {
      this.emitMutation(
        {
          case: "pathUpdate",
          value: create(PathUpdateRequestSchema, {
            path: patch.path.join("."),
            targetId: nodeId,
            type: PathUpdateRequest_UpdateType.REPLACE,
            value: fromJson(ValueSchema, patch.value),
          }),
        },
        MutationSource.SOURCE_REMOTE_TASK,
      );
    });
  }

  protected emitMutation(operation: GraphMutation["operation"], source: MutationSource = MutationSource.SOURCE_USER) {
    const mutation = create(GraphMutationSchema, {
      operation: operation,
    });

    logMutation(mutation.operation.case ?? "unknown", toBinary(GraphMutationSchema, mutation), source);

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
