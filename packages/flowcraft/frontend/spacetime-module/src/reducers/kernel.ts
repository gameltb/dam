import { type ReducerCtx, t } from "spacetimedb/server";

import {
  type TaskAuditLog as ProtoTaskAuditLog,
  type TaskDefinition as ProtoTaskDefinition,
  type TaskUpdate as ProtoTaskUpdate,
  type WorkerInfo as ProtoWorkerInfo,
  TaskAuditLogSchema,
  TaskDefinitionSchema,
  TaskStatus,
  TaskUpdateSchema,
  WorkerInfoSchema,
  WorkerLanguage,
} from "../generated/flowcraft/v1/core/kernel_pb";
import { type AppSchema } from "../schema";

export const kernelReducers = {
  claimTask: {
    args: { taskId: t.string(), workerId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId, workerId }: { taskId: string; workerId: string }) => {
      const task = ctx.db.tasks.id.find(taskId);
      if (!task) throw new Error("TASK_NOT_FOUND");

      const status = task.status;

      // State transition validation
      if (status !== (TaskStatus.PENDING as number)) {
        throw new Error(`INVALID_TRANSITION: Cannot claim task in state ${status.toString()}`);
      }

      ctx.db.tasks.id.update({
        ...task,
        lastHeartbeat: ctx.timestamp.toMillis(),
        ownerId: workerId,
        status: TaskStatus.CLAIMED as number,
      });
    },
  },

  completeTask: {
    args: { result: t.string(), taskId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { result, taskId }: { result: string; taskId: string }) => {
      const task = ctx.db.tasks.id.find(taskId);
      if (task) {
        const status = task.status as unknown as TaskStatus;
        if (status === TaskStatus.CLAIMED || status === TaskStatus.RUNNING) {
          ctx.db.tasks.id.update({
            ...task,
            lastHeartbeat: ctx.timestamp.toMillis(),
            result,
            status: TaskStatus.COMPLETED as unknown as number,
            version: task.version + 1,
          });
        }

        const runtime = ctx.db.nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          ctx.db.nodeRuntimeStates.nodeId.update({
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
      const task = ctx.db.tasks.id.find(taskId);
      if (task) {
        const status = task.status as unknown as TaskStatus;
        if (status !== TaskStatus.COMPLETED) {
          ctx.db.tasks.id.update({
            ...task,
            lastHeartbeat: ctx.timestamp.toMillis(),
            result: error,
            status: TaskStatus.FAILED as unknown as number,
            version: task.version + 1,
          });
        }

        ctx.db.taskAuditLog.insert({
          eventType: "error",
          id: `${taskId}-fail-${ctx.timestamp.toMillis().toString()}-${ctx.db.taskAuditLog.count().toString()}`,
          message: error,
          nodeId: task.nodeId,
          taskId,
          timestamp: ctx.timestamp.toMillis(),
        });

        const runtime = ctx.db.nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          ctx.db.nodeRuntimeStates.nodeId.update({
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
    handler: (ctx: ReducerCtx<AppSchema>, { log }: { log: ProtoTaskAuditLog }) => {
      ctx.db.taskAuditLog.insert({
        eventType: log.eventType,
        id: log.id || `${log.taskId}-${ctx.timestamp.toMillis().toString()}-log`,
        message: log.message,
        nodeId: log.nodeId,
        taskId: log.taskId,
        timestamp: log.timestamp,
      });
    },
  },

  registerWorker: {
    args: { info: WorkerInfoSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { info }: { info: ProtoWorkerInfo }) => {
      const existing = ctx.db.workers.workerId.find(info.workerId);

      const workerRow = {
        capabilities: info.capabilities.join(","),
        lang: info.lang ?? WorkerLanguage.WORKER_LANG_TS,
        lastHeartbeat: ctx.timestamp.toMillis(),
        tagsJson: JSON.stringify(info.tags ?? {}),
        workerId: info.workerId,
      };

      if (existing) {
        ctx.db.workers.workerId.update(workerRow);
      } else {
        ctx.db.workers.insert(workerRow);
      }
    },
  },

  submitTask: {
    args: { idempotencyKey: t.string(), task: TaskDefinitionSchema },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { idempotencyKey, task }: { idempotencyKey: string; task: ProtoTaskDefinition },
    ) => {
      // 1. Idempotency check
      const existing = Array.from(ctx.db.tasks.iter()).find((t) => t.idempotencyKey === idempotencyKey);
      if (existing) {
        if (existing.status === (TaskStatus.COMPLETED as number)) {
          return; // Already completed, skip directly
        }
        // If the task is stuck, allow override/reset
        ctx.db.tasks.id.delete(existing.id);
      }

      // 2. Busy Guard (only for new tasks that are not idempotent duplicates)
      const busy = Array.from(ctx.db.tasks.iter()).some((t) => {
        const s = t.status;
        return (
          t.nodeId === task.nodeId &&
          (s === (TaskStatus.CLAIMED as number) ||
            s === (TaskStatus.PENDING as number) ||
            s === (TaskStatus.RUNNING as number))
        );
      });

      if (busy) {
        throw new Error(`NODE_BUSY: Node ${task.nodeId} is already executing a task.`);
      }

      ctx.db.tasks.insert({
        id: task.taskId,
        idempotencyKey: idempotencyKey,
        lastHeartbeat: ctx.timestamp.toMillis(),
        nodeId: task.nodeId,
        ownerId: "",
        paramsPayload: task.paramsPayload,
        result: "",
        status: TaskStatus.PENDING as number,
        taskType: task.taskType,
        timestamp: ctx.timestamp.toMillis(),
        version: 0,
      });

      const runtime = ctx.db.nodeRuntimeStates.nodeId.find(task.nodeId);
      if (runtime) {
        ctx.db.nodeRuntimeStates.nodeId.update({
          ...runtime,
          lastUpdated: ctx.timestamp.toMillis(),
          status: "busy",
        });
      }
    },
  },

  updateTaskProgress: {
    args: { update: TaskUpdateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { update }: { update: ProtoTaskUpdate }) => {
      const task = ctx.db.tasks.id.find(update.taskId);
      if (!task) return;

      // State transition validation: can only update RUNNING or CLAIMED tasks
      const currentStatus = task.status;
      if (currentStatus !== (TaskStatus.CLAIMED as number) && currentStatus !== (TaskStatus.RUNNING as number)) {
        return;
      }

      if (update.status !== undefined) {
        let taskMessage = update.message;
        if (!taskMessage && update.status === (TaskStatus.FAILED as number)) {
          if (update.result?.kind?.case === "stringValue") {
            taskMessage = update.result.kind.value;
          }
        }

        ctx.db.tasks.id.update({
          ...task,
          lastHeartbeat: ctx.timestamp.toMillis(),
          result: taskMessage || task.result,
          status: update.status,
          version: task.version + 1,
        });
      }
    },
  },
};
