import { type ReducerCtx } from "spacetimedb/server";

import { type AppSchema } from "../schema";

/**
 * Logs operations.
 */
export function logOperation(ctx: ReducerCtx<AppSchema>, type: string, payload: Record<string, unknown> | string) {
  const identity = ctx.sender.toHexString();
  let taskId = "";

  // Try to find the taskId bound to the current identity
  const assignments = ctx.db?.clientTaskAssignments;
  if (assignments) {
    for (const assignment of assignments.iter()) {
      if ((assignment as any).clientIdentity === identity) {
        taskId = (assignment as any).taskId;
        break;
      }
    }
  }

  ctx.db?.operationLogs?.insert({
    clientIdentity: identity,
    id: `${identity}-${ctx.timestamp.toMillis()}-${ctx.db?.operationLogs?.count() || 0}`,
    operationType: type,
    payloadJson: typeof payload === "string" ? payload : JSON.stringify(payload),
    taskId,
    timestamp: ctx.timestamp.toMillis(),
  });
}
