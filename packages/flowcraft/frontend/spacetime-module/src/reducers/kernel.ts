import { type ReducerCtx, t } from "spacetimedb/server";

import {
  TaskAuditLogSchema,
  TaskDefinitionSchema,
  TaskUpdateSchema,
  WorkerInfoSchema,
} from "../generated/flowcraft/v1/core/kernel_pb";
import {
  type core_TaskAuditLog,
  type core_TaskDefinition,
  type core_TaskUpdate,
  type core_WorkerInfo,
} from "../generated/generated_schema";
import { type AppSchema } from "../schema";

export const kernelReducers = {
  claimTask: {
    args: { taskId: t.string(), workerId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId, workerId }: { taskId: string; workerId: string }) => {
      const { tasks } = ctx.db;
      const task = tasks.id.find(taskId);
      if (!task) throw new Error("TASK_NOT_FOUND");

      // 状态转换校验
      if (task.status.tag !== "TASK_STATUS_PENDING") {
        throw new Error(`INVALID_TRANSITION: Cannot claim task in ${task.status.tag} state`);
      }

      tasks.id.update({
        ...task,
        lastHeartbeat: ctx.timestamp.toMillis(),
        ownerId: workerId,
        status: { tag: "TASK_STATUS_CLAIMED" },
      });
    },
  },

  completeTask: {
    args: { result: t.string(), taskId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { result, taskId }: { result: string; taskId: string }) => {
      const { nodeRuntimeStates, tasks } = ctx.db;
      const task = tasks.id.find(taskId);
      if (task && ["TASK_STATUS_CLAIMED", "TASK_STATUS_RUNNING"].includes(task.status.tag)) {
        tasks.id.update({
          ...task,
          lastHeartbeat: ctx.timestamp.toMillis(),
          result,
          status: { tag: "TASK_STATUS_COMPLETED" },
          version: task.version + 1,
        });

        const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          nodeRuntimeStates.nodeId.update({
            ...runtime,
            lastUpdated: ctx.timestamp.toMillis(),
            status: "idle",
          });
        }
      }
    },
  },

  failTask: {
    args: { error: t.string(), taskId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { error, taskId }: { error: string; taskId: string }) => {
      const { nodeRuntimeStates, taskAuditLog, tasks } = ctx.db;
      const task = tasks.id.find(taskId);
      if (task && task.status.tag !== "TASK_STATUS_COMPLETED") {
        tasks.id.update({
          ...task,
          lastHeartbeat: ctx.timestamp.toMillis(),
          result: error,
          status: { tag: "TASK_STATUS_FAILED" },
          version: task.version + 1,
        });

        taskAuditLog.insert({
          eventType: "error",
          id: `${taskId}-fail-${ctx.timestamp.toMillis()}-${ctx.db.taskAuditLog.count()}`,
          message: error,
          nodeId: task.nodeId,
          taskId,
          timestamp: ctx.timestamp.toMillis(),
        });

        const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          nodeRuntimeStates.nodeId.update({
            ...runtime,
            error,
            lastUpdated: ctx.timestamp.toMillis(),
            status: "error",
          });
        }
      }
    },
  },

  logTaskEvent: {
    args: { log: TaskAuditLogSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { log }: { log: core_TaskAuditLog }) => {
      ctx.db.taskAuditLog.insert({
        ...log,
        id: log.id || `${log.taskId}-${ctx.timestamp.toMillis()}-${ctx.db.taskAuditLog.count()}`,
      });
    },
  },

  registerWorker: {
    args: { info: WorkerInfoSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { info }: { info: core_WorkerInfo }) => {
      const { workers } = ctx.db;
      const existing = workers.workerId.find(info.workerId);

      const workerRow = {
        capabilities: info.capabilities.join(","),
        lang: { tag: info.lang?.tag || "WORKER_LANG_TS" } as any,
        lastHeartbeat: ctx.timestamp.toMillis(),
        tagsJson: JSON.stringify(info.tags || {}),
        workerId: info.workerId,
      };

      if (existing) {
        workers.workerId.update(workerRow);
      } else {
        workers.insert(workerRow);
      }
    },
  },

  submitTask: {
    args: { idempotencyKey: t.string(), task: TaskDefinitionSchema },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { idempotencyKey, task }: { idempotencyKey: string; task: core_TaskDefinition },
    ) => {
      const { nodeRuntimeStates, tasks } = ctx.db;

      // 1. 幂等性检查
      const existing = Array.from(tasks.iter()).find((t) => t.idempotencyKey === idempotencyKey);
      if (existing) {
        if (existing.status.tag === "TASK_STATUS_COMPLETED") {
          return; // 已完成，直接跳过
        }
        // 如果任务卡住了，则允许覆盖/重置
        tasks.id.delete(existing.id);
      }

      // 2. Busy Guard (仅针对非幂等重复的全新任务)
      const busy = Array.from(tasks.iter()).some(
        (t) =>
          t.nodeId === task.nodeId &&
          ["TASK_STATUS_CLAIMED", "TASK_STATUS_PENDING", "TASK_STATUS_RUNNING"].includes(t.status.tag),
      );

      if (busy) {
        throw new Error(`NODE_BUSY: Node ${task.nodeId} is already executing a task.`);
      }

      tasks.insert({
        id: task.taskId,
        idempotencyKey: idempotencyKey,
        lastHeartbeat: ctx.timestamp.toMillis(),
        nodeId: task.nodeId,
        ownerId: "",
        paramsPayload: task.paramsPayload,
        result: "",
        status: { tag: "TASK_STATUS_PENDING" },
        taskType: task.taskType,
        timestamp: ctx.timestamp.toMillis(),
        version: 0,
      });

      const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
      if (runtime) {
        nodeRuntimeStates.nodeId.update({
          ...runtime,
          lastUpdated: ctx.timestamp.toMillis(),
          status: "busy",
        });
      }
    },
  },

  updateTaskProgress: {
    args: { update: TaskUpdateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { update }: { update: core_TaskUpdate }) => {
      const { tasks } = ctx.db;
      const task = tasks.id.find(update.taskId);
      if (!task) return;

      // 状态转换校验：只能更新 RUNNING 或 CLAIMED 任务
      if (!["TASK_STATUS_CLAIMED", "TASK_STATUS_RUNNING"].includes(task.status.tag)) {
        return;
      }

      if (update.status !== undefined) {
        let tag = (update.status as any).tag || update.status;
        const STATUS_MAP: Record<number | string, string> = {
          0: "TASK_STATUS_PENDING",
          1: "TASK_STATUS_CLAIMED",
          2: "TASK_STATUS_RUNNING",
          3: "TASK_STATUS_COMPLETED",
          4: "TASK_STATUS_FAILED",
          5: "TASK_STATUS_CANCELLED",
        };

        if (typeof tag === "number") {
          tag = STATUS_MAP[tag] || tag;
        }

        let taskMessage = update.message;
        if (!taskMessage && tag === "TASK_STATUS_FAILED") {
          if ((update as any).result?.kind?.case === "stringValue") {
            taskMessage = (update as any).result.kind.value;
          }
        }

        tasks.id.update({
          ...task,
          lastHeartbeat: ctx.timestamp.toMillis(),
          result: taskMessage || task.result,
          status: { tag: tag },
          version: task.version + 1,
        });
      }
    },
  },
};
