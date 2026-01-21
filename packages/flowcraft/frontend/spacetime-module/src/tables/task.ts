import { t, table } from "spacetimedb/server";

import { core_TaskStatus, core_NodeSignal_payload, core_WorkerLanguage } from "../generated/generated_schema";

export const tasks = table(
  {
    name: "tasks",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    nodeId: t.string(),
    taskType: t.string(),
    paramsPayload: t.byteArray(),
    selectorJson: t.string(), // Opaque JSON for selector
    status: core_TaskStatus,
    ownerId: t.string(),
    result: t.string(),
    timestamp: t.u64(),
  },
);

export const workers = table(
  {
    name: "workers",
    public: true,
  },
  {
    workerId: t.string().primaryKey(),
    lang: core_WorkerLanguage,
    capabilities: t.string(), // Comma separated for simplicity in indexing
    tagsJson: t.string(),
    lastHeartbeat: t.u64(),
  }
);

export const taskAuditLog = table(
  {
    name: "task_audit_log",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    taskId: t.string(),
    nodeId: t.string(),
    eventType: t.string(),
    message: t.string(),
    timestamp: t.u64(),
  }
);

export const nodeSignals = table(
  {
    name: "node_signals",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    nodeId: t.string(),
    payload: core_NodeSignal_payload, // Protobuf Enum/Union
    timestamp: t.u64(),
  },
);

export const widgetValues = table(
  {
    name: "widget_values",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    nodeId: t.string(),
    value: t.string(), // Keep string for simple values
    widgetId: t.string(),
  },
);

export const clientTaskAssignments = table(
  {
    name: "client_task_assignments",
    public: true,
  },
  {
    clientIdentity: t.string().primaryKey(),
    taskId: t.string(),
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