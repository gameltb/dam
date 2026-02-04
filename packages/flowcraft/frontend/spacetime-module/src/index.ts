import { type ReducerCtx, t } from "spacetimedb/server";

import { chatReducers } from "./reducers/chat";
import { configReducers } from "./reducers/config";
import { kernelReducers } from "./reducers/kernel";
import { nodeReducers } from "./reducers/node";
import { runtimeReducers } from "./reducers/runtime";
import { taskReducers } from "./reducers/task";
import { uiReducers } from "./reducers/ui";
import { type AppSchema, spacetimedb } from "./schema";
import { wrapPbHandler } from "./utils/reducer-wrapper";

const ALL: Record<string, ReducerDefinition<any>> = {
  ...nodeReducers,
  ...chatReducers,
  ...configReducers,
  ...taskReducers,
  ...runtimeReducers,
  ...kernelReducers,
  ...uiReducers,
};

interface ReducerDefinition<P extends Record<string, any>> {
  args: Record<string, unknown>;
  handler: (ctx: ReducerCtx<AppSchema>, params: P) => void;
}

// 1. Auto-register and wrap
for (const [name, def] of Object.entries(ALL)) {
  const stArgs: Record<string, unknown> = {};
  for (const [argName, argType] of Object.entries(def.args)) {
    if (argType && typeof argType === "object" && "typeName" in argType) {
      stArgs[argName] = t.byteArray();
    } else {
      stArgs[argName] = argType;
    }
  }

  spacetimedb.reducer(name, stArgs as any, wrapPbHandler(def.args, def.handler));
}

spacetimedb.clientDisconnected((ctx: ReducerCtx<AppSchema>) => {
  const identity = ctx.sender.toHexString();
  const assignments = ctx.db.clientTaskAssignments;
  if (!assignments) return;
  const existing = Array.from(assignments.iter()).find((r: any) => r.clientIdentity === identity);
  if (existing) {
    assignments.delete(existing);
  }
});

export default spacetimedb;
export * from "./reducers/runtime";
export * from "./reducers/task";
export * from "./reducers/ui";
export * from "./schema";
