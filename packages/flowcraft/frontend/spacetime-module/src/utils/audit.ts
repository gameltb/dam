import { type ReducerCtx } from "spacetimedb/server";

/**
 * Shared logic to log operations using the implicit task context of the sender.
 */
export function logOperation(
  ctx: ReducerCtx<any>,
  type: string,
  payload: any,
) {
  const identity = ctx.sender.toHexString();
  let taskId = "unassigned";

  // Use simple iteration to find the assignment if direct index find fails in TS
  for (const assignment of ctx.db.clientTaskAssignments) {
    if ((assignment as any).clientIdentity === identity) {
      taskId = (assignment as any).taskId as string;
      break;
    }
  }

  ctx.db.operationLogs.insert({
    id: ctx.timestamp.toMillis().toString() + "_" + Math.random().toString(36).substring(2, 9),
    clientIdentity: identity,
    operationType: type,
    payloadJson: typeof payload === "string" ? payload : JSON.stringify(payload),
    taskId: taskId,
    timestamp: ctx.timestamp.toMillis(),
  });
}
