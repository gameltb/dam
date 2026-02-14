import { type ReducerCtx } from "spacetimedb/server";

import {
  type ResetNodeRequest as ProtoResetNodeRequest,
  ResetNodeRequestSchema,
} from "../generated/flowcraft/v1/core/service_pb";
import { type AppSchema } from "../schema";

export const runtimeReducers = {
  reset_node: {
    args: { req: ResetNodeRequestSchema },
    handler: (ctx: ReducerCtx<AppSchema>, { req }: { req: ProtoResetNodeRequest }) => {
      // 1. Clear runtime state
      const existing = ctx.db.nodeRuntimeStates.nodeId.find(req.nodeId);
      if (existing) {
        ctx.db.nodeRuntimeStates.nodeId.delete(req.nodeId);
      }

      // 2. Clear related tasks (if any)
      for (const task of ctx.db.tasks.iter()) {
        if (task.nodeId === req.nodeId) {
          ctx.db.tasks.id.delete(task.id);
        }
      }

      // 3. If data is cleared, handle it here
      if (req.clearData) {
        // Implementation for clearing node data
      }
    },
  },
};
