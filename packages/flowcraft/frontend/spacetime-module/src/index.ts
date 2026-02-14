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

interface ReducerDefinition {
  args: Record<string, unknown>;
  handler: (ctx: ReducerCtx<AppSchema>, params: any) => void;
}

const ALL: Record<string, ReducerDefinition> = {
  ...nodeReducers,
  ...chatReducers,
  ...configReducers,
  ...taskReducers,
  ...runtimeReducers,
  ...kernelReducers,
  ...uiReducers,
};

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

  // Use 'any' for stArgs to avoid complex type mapping from Record<string, unknown> to ParamsObj
  spacetimedb.reducer(name, stArgs as any, wrapPbHandler(def.args, def.handler));
}

spacetimedb.clientDisconnected((ctx: ReducerCtx<AppSchema>) => {
  const identity = ctx.sender.toHexString();
  const assignments = ctx.db.clientTaskAssignments;
  const existing = Array.from(assignments.iter()).find((r) => r.clientIdentity === identity);
  if (existing) {
    assignments.delete(existing);
  }
});

export default spacetimedb;
export * from "./reducers/runtime";
export * from "./reducers/task";
export * from "./reducers/ui";
export * from "./schema";
