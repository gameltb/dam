import { type ReducerCtx, t } from "spacetimedb/server";

import { ActionExecutionRequest as ProtoActionExecutionRequest } from "../generated/flowcraft/v1/core/action_pb";
import { ActionExecutionRequestSchema } from "../generated/flowcraft/v1/core/action_pb";
import { TaskUpdate as ProtoTaskUpdate } from "../generated/flowcraft/v1/core/node_pb";
import { TaskUpdateSchema } from "../generated/flowcraft/v1/core/node_pb";
import { NodeSignal as ProtoNodeSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { NodeSignalSchema } from "../generated/flowcraft/v1/core/signals_pb";
import {
  ActionExecutionRequest as StdbActionExecutionRequest,
  NodeSignal as StdbNodeSignal,
  TaskUpdate as StdbTaskUpdate,
} from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";

export const taskReducers = {
  assign_current_task: {
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

  execute_action: {
    args: {
      id: t.string(),
      request: ActionExecutionRequestSchema,
    },
    handler: (ctx: ReducerCtx<AppSchema>, { id, request }: { id: string; request: ProtoActionExecutionRequest }) => {
      ctx.db.tasks.insert({
        id: id,
        request: pbToStdb(
          ActionExecutionRequestSchema,
          StdbActionExecutionRequest,
          request,
        ) as StdbActionExecutionRequest,
        result: "",
        status: { tag: "TASK_PENDING" },
        timestamp: ctx.timestamp.toMillis(),
      });
    },
  },

  send_node_signal: {
    args: {
      signal: NodeSignalSchema,
    },
    handler: (ctx: ReducerCtx<AppSchema>, { signal }: { signal: ProtoNodeSignal }) => {
      const stSignal = pbToStdb(NodeSignalSchema, StdbNodeSignal, signal) as StdbNodeSignal;
      ctx.db.nodeSignals.insert({
        id: crypto.randomUUID(),
        nodeId: signal.nodeId,
        payload: stSignal.payload,
        timestamp: ctx.timestamp.toMillis(),
      });
    },
  },

  update_task_status: {
    args: { update: TaskUpdateSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { update }: { update: ProtoTaskUpdate }) => {
      const task = ctx.db.tasks.id.find(update.taskId);
      if (task) {
        const stUpdate = pbToStdb(TaskUpdateSchema, StdbTaskUpdate, update) as StdbTaskUpdate;
        const updated = { ...task };
        if (stUpdate.status) {
          updated.status = stUpdate.status;
        }
        if (stUpdate.result) {
          updated.result = stUpdate.result;
        }
        ctx.db.tasks.id.update(updated);
      }
    },
  },
};
