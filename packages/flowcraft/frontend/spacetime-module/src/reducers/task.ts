import { type ReducerCtx, t } from "spacetimedb/server";

import { type ActionExecutionRequest as ProtoActionExecutionRequest } from "../generated/flowcraft/v1/core/action_pb";
import { ActionExecutionRequestSchema } from "../generated/flowcraft/v1/core/action_pb";
import { type TaskUpdate as ProtoTaskUpdate } from "../generated/flowcraft/v1/core/kernel_pb";
import { TaskUpdateSchema } from "../generated/flowcraft/v1/core/kernel_pb";
import { type NodeSignal as ProtoNodeSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { NodeSignalSchema } from "../generated/flowcraft/v1/core/signals_pb";
import { core_NodeSignal as StdbNodeSignal } from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";

export const taskReducers = {
  assignCurrentTask: {
    args: { taskId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { taskId }: { taskId: string }) => {
      const identity = ctx.sender.toHexString();
      const assignments = ctx.db.clientTaskAssignments;
      const existing = assignments.clientIdentity.find(identity);
      if (existing) {
        assignments.clientIdentity.update({
          clientIdentity: identity,
          taskId: taskId,
        });
      } else {
        assignments.insert({ clientIdentity: identity, taskId: taskId });
      }
    },
  },

  clear_node_tasks: {
    args: { nodeId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId }: { nodeId: string }) => {
      const tasks = [...ctx.db.tasks.iter()];
      for (const task of tasks) {
        if (task.nodeId === nodeId) {
          ctx.db.tasks.id.delete(task.id);
        }
      }
    },
  },

  executeAction: {
    args: {
      id: t.string(),
      request: ActionExecutionRequestSchema,
    },
    handler: (ctx: ReducerCtx<AppSchema>, { id, request }: { id: string; request: ProtoActionExecutionRequest }) => {
      ctx.db.tasks.insert({
        id: id,
        idempotencyKey: id, // Fallback to task id if not provided via submitTask
        lastHeartbeat: ctx.timestamp.toMillis(),
        nodeId: request.sourceNodeId,
        ownerId: "",
        paramsPayload: new TextEncoder().encode(JSON.stringify(request.params || {})),
        result: "",
        status: { tag: "TASK_STATUS_PENDING" },
        taskType: request.actionId,
        timestamp: ctx.timestamp.toMillis(),
        version: 0,
      });
    },
  },

  sendNodeSignal: {
    args: {
      signal: NodeSignalSchema,
    },
    handler: (ctx: ReducerCtx<AppSchema>, { signal }: { signal: ProtoNodeSignal }) => {
      const stSignal = pbToStdb(NodeSignalSchema, StdbNodeSignal, signal) as StdbNodeSignal;
      if (!stSignal.payload) throw new Error("[Task] Missing payload in signal");

      ctx.db.nodeSignals.insert({
        id: `sig-${signal.nodeId}-${ctx.timestamp.toMillis()}-${ctx.db.nodeSignals.count()}`,
        nodeId: signal.nodeId,
        payload: stSignal.payload,
        timestamp: ctx.timestamp.toMillis(),
      });
    },
  },

  updateTaskStatus: {
    args: { update: TaskUpdateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { update }: { update: ProtoTaskUpdate }) => {
      const task = ctx.db.tasks.id.find(update.taskId);
      if (task && update.status !== undefined) {
        // Handle both string tags and numeric enum values from Protobuf
        let tag = (update.status as any).tag || update.status;

        // Map numeric indices back to variant names if necessary
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
          // If status is failed but message is empty, try to extract from result
          if (update.result?.kind?.case === "stringValue") {
            taskMessage = update.result.kind.value;
          }
        }

        ctx.db.tasks.id.update({
          ...task,
          result: taskMessage || task.result,
          status: { tag: tag },
        });
      }
    },
  },
};
