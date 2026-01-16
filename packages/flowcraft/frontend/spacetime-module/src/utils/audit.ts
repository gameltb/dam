import { type ReducerCtx } from "spacetimedb/server";

import { type AppSchema } from "../schema";

/**
 * Shared logic to log operations using the implicit task context of the sender.
 */
export function logOperation(ctx: ReducerCtx<AppSchema>, type: string, payload: Record<string, unknown> | string) {
  const identity = ctx.sender.toHexString();
  let taskId = "unassigned";

  // Use simple iteration to find the assignment if direct index find fails in TS
  for (const assignment of ctx.db.clientTaskAssignments) {
    if (assignment.clientIdentity === identity) {
      taskId = assignment.taskId;
      break;
    }
  }

  ctx.db.operationLogs.insert({
    clientIdentity: identity,
    id: ctx.timestamp.toMillis().toString() + "_" + Math.random().toString(36).substring(2, 9),
    operationType: type,
    payloadJson: typeof payload === "string" ? payload : JSON.stringify(payload),
    taskId: taskId,
    timestamp: ctx.timestamp.toMillis(),
  });
}
