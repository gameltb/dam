import { type ReducerCtx } from "spacetimedb/server";
import { type AppSchema } from "../schema";
import { services_ResetNodeRequest } from "../generated/generated_schema";

export const runtimeReducers = {
  reset_node: {
    args: { req: services_ResetNodeRequest },
    handler: (ctx: ReducerCtx<AppSchema>, params: { req: services_ResetNodeRequest }) => {
      const { req } = params;
      
      // 1. 清除运行时状态
      const existing = ctx.db.nodeRuntimeStates.nodeId.find(req.nodeId);
      if (existing) {
        ctx.db.nodeRuntimeStates.nodeId.delete(req.nodeId);
      }

      // 2. 清除相关的任务 (如果有)
      for (const task of ctx.db.tasks.iter()) {
        if (task.nodeId === req.nodeId) {
          ctx.db.tasks.id.delete(task.id);
        }
      }

      // 3. 如果清除了数据，可以在这里处理 (目前暂时留空)
      if (req.clearData) {
        // Implementation for clearing node data if requested
      }
    }
  }
};
