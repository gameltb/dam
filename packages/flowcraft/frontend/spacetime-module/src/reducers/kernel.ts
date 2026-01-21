import { type ReducerCtx, t } from "spacetimedb/server";
import { 
  TaskDefinitionSchema, 
  TaskUpdateSchema, 
  WorkerInfoSchema,
  TaskAuditLogSchema
} from "../generated/flowcraft/v1/core/kernel_pb";
import {
  type core_TaskDefinition,
  type core_WorkerInfo,
  type core_TaskUpdate,
  type core_TaskAuditLog,
} from "../generated/generated_schema";
import { type AppSchema } from "../schema";

export const kernelReducers = {
  submitTask: {
    args: { task: TaskDefinitionSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { task }: { task: core_TaskDefinition }) => {
      const { tasks, nodeRuntimeStates } = ctx.db;
      
      // Guard: Busy Check
      const busy = Array.from(tasks.iter()).some(t => 
        t.nodeId === task.nodeId && 
        ["TASK_STATUS_PENDING", "TASK_STATUS_CLAIMED", "TASK_STATUS_RUNNING"].includes(t.status.tag)
      );

      if (busy) {
        throw new Error(`NODE_BUSY: Node ${task.nodeId} is already executing a task.`);
      }

      tasks.insert({
        id: task.taskId,
        nodeId: task.nodeId,
        taskType: task.taskType,
        paramsPayload: task.paramsPayload,
        selectorJson: "", // Placeholder or from task
        status: { tag: "TASK_STATUS_PENDING" },
        ownerId: "",
        result: "",
        timestamp: BigInt(Date.now())
      });

      // Update runtime state
      const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
      if (runtime) {
        nodeRuntimeStates.nodeId.update({
          ...runtime,
          status: "busy",
          lastUpdated: BigInt(Date.now())
        });
      }
    }
  },

  registerWorker: {
    args: { info: WorkerInfoSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { info }: { info: core_WorkerInfo }) => {
      const { workers } = ctx.db;
      const existing = workers.workerId.find(info.workerId);
      
      const workerRow = {
        workerId: info.workerId,
        lang: { tag: info.lang?.tag || "WORKER_LANG_TS" } as any,
        capabilities: info.capabilities.join(","),
        tagsJson: JSON.stringify(info.tags || {}),
        lastHeartbeat: BigInt(Date.now())
      };

      if (existing) {
        workers.workerId.update(workerRow);
      } else {
        workers.insert(workerRow);
      }
    }
  },

  claimTask: {
    args: { taskId: t.string(), workerId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId, workerId }: { taskId: string, workerId: string }) => {
      const { tasks } = ctx.db;
      const task = tasks.id.find(taskId);
      if (!task) throw new Error("TASK_NOT_FOUND");
      if (task.status.tag !== "TASK_STATUS_PENDING") throw new Error("TASK_ALREADY_CLAIMED");

      tasks.id.update({
        ...task,
        status: { tag: "TASK_STATUS_CLAIMED" },
        ownerId: workerId
      });
    }
  },

  updateTaskProgress: {
    args: { update: TaskUpdateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { update }: { update: core_TaskUpdate }) => {
      const { tasks } = ctx.db;
      const task = tasks.id.find(update.taskId);
      if (task && update.status) {
        tasks.id.update({
          ...task,
          status: { tag: (update.status as any).tag || update.status }
        });
      }
    }
  },

  completeTask: {
    args: { taskId: t.string(), result: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId, result }: { taskId: string, result: string }) => {
      const { tasks, nodeRuntimeStates } = ctx.db;
      const task = tasks.id.find(taskId);
      if (task) {
        tasks.id.update({
          ...task,
          status: { tag: "TASK_STATUS_COMPLETED" },
          result
        });

        const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          nodeRuntimeStates.nodeId.update({
            ...runtime,
            status: "idle",
            lastUpdated: BigInt(Date.now())
          });
        }
      }
    }
  },

  failTask: {
    args: { taskId: t.string(), error: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId, error }: { taskId: string, error: string }) => {
      const { tasks, nodeRuntimeStates, taskAuditLog } = ctx.db;
      const task = tasks.id.find(taskId);
      if (task) {
        tasks.id.update({
          ...task,
          status: { tag: "TASK_STATUS_FAILED" }
        });

        taskAuditLog.insert({
          id: crypto.randomUUID(),
          taskId,
          nodeId: task.nodeId,
          eventType: "error",
          message: error,
          timestamp: BigInt(Date.now())
        });

        const runtime = nodeRuntimeStates.nodeId.find(task.nodeId);
        if (runtime) {
          nodeRuntimeStates.nodeId.update({
            ...runtime,
            status: "error",
            error,
            lastUpdated: BigInt(Date.now())
          });
        }
      }
    }
  },

  logTaskEvent: {
    args: { log: TaskAuditLogSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { log }: { log: core_TaskAuditLog }) => {
      ctx.db.taskAuditLog.insert({
        ...log,
        id: log.id || crypto.randomUUID()
      });
    }
  }
};
