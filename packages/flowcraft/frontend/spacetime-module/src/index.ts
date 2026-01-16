import { type ReducerCtx, t } from "spacetimedb/server";

import { chatReducers } from "./reducers/chat";
import { configReducers } from "./reducers/config";
import { nodeReducers } from "./reducers/node";
import { taskReducers } from "./reducers/task";
import { type AppSchema, spacetimedb } from "./schema";
import { wrapPbHandler } from "./utils/reducer-wrapper";

const ALL = {
  ...nodeReducers,
  ...chatReducers,
  ...configReducers,
  ...taskReducers,
};

interface ReducerDefinition {
  args: Record<string, unknown>;
  handler: (ctx: ReducerCtx<AppSchema>, params: any) => void;
}

// 1. 自动注册并包装
for (const [name, def] of Object.entries(ALL as Record<string, ReducerDefinition>)) {
  const stArgs: Record<string, unknown> = {};
  for (const [argName, argType] of Object.entries(def.args)) {
    if (argType && typeof argType === "object" && "typeName" in argType) {
      stArgs[argName] = t.byteArray();
    } else {
      stArgs[argName] = argType;
    }
  }

  spacetimedb.reducer(name, stArgs as any, wrapPbHandler<AppSchema>(def.args, def.handler));
}

spacetimedb.clientDisconnected((ctx: ReducerCtx<AppSchema>) => {
  const identity = ctx.sender.toHexString();
  const assignments = ctx.db.clientTaskAssignments;
  const existing = assignments.clientIdentity.find(identity);
  if (existing) {
    assignments.clientIdentity.delete(identity);
  }
});

export default spacetimedb;
