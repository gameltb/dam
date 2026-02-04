import { t, table } from "spacetimedb/server";

export const taskAuditLog = table(
  {
    name: "task_audit_log",
    public: true,
  },
  {
    eventType: t.string(),
    id: t.string().primaryKey(),
    message: t.string(),
    nodeId: t.string(),
    taskId: t.string(),
    timestamp: t.u64(),
  },
);

export const operationLogs = table(
  {
    name: "operation_logs",
    public: true,
  },
  {
    clientIdentity: t.string(),
    id: t.string().primaryKey(),
    operationType: t.string(),
    payloadJson: t.string(),
    taskId: t.string(),
    timestamp: t.u64(),
  },
);

export const nodeSignals = table(
  {
    name: "node_signals",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    nodeId: t.string(),
    payload: Object.assign(t.byteArray(), { __pb_schema: "NodeSignal" }),
    timestamp: t.u64(),
  },
);
