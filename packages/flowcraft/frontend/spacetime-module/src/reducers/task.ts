import { type ReducerCtx, t } from "spacetimedb/server";

import { type ActionExecutionRequest as ProtoActionExecutionRequest } from "../generated/flowcraft/v1/core/action_pb";
import { ActionExecutionRequestSchema } from "../generated/flowcraft/v1/core/action_pb";
import { type TaskUpdate as ProtoTaskUpdate } from "../generated/flowcraft/v1/core/kernel_pb";
import { TaskUpdateSchema } from "../generated/flowcraft/v1/core/kernel_pb";
import { type NodeSignal as ProtoNodeSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { NodeSignalSchema } from "../generated/flowcraft/v1/core/signals_pb";
import {
  core_NodeSignal as StdbNodeSignal,
} from "../generated/generated_schema";
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

  executeAction: {
    args: {
      id: t.string(),
      request: ActionExecutionRequestSchema,
    },
    handler: (ctx: ReducerCtx<AppSchema>, { id, request }: { id: string; request: ProtoActionExecutionRequest }) => {
      ctx.db.tasks.insert({
        id: id,
        nodeId: request.sourceNodeId,
        taskType: request.actionId,
        paramsPayload: new TextEncoder().encode(JSON.stringify(request.params || {})),
        selectorJson: "",
        ownerId: "",
        result: "",
        status: { tag: "TASK_STATUS_PENDING" },
        timestamp: ctx.timestamp.toMillis(),
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
        id: crypto.randomUUID(),
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
      if (task && update.status) {
        ctx.db.tasks.id.update({
            ...task,
            status: { tag: (update.status as any).tag || update.status }
        });
      }
    },
  },
};