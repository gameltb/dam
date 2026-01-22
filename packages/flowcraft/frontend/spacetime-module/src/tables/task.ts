import { t, table } from "spacetimedb/server";

import { core_NodeSignal_payload, core_TaskStatus, core_WorkerLanguage } from "../generated/generated_schema";

export const tasks = table(
  {
    name: "tasks",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    idempotencyKey: t.string().index(), // 幂等键
    lastHeartbeat: t.u64(), // 任务级心跳
    nodeId: t.string(),
    ownerId: t.string(),
    paramsPayload: t.byteArray(),
    result: t.string(),
    status: core_TaskStatus,
    taskType: t.string(),
    timestamp: t.u64(),
    version: t.u32(), // 状态版本
  },
);

export const workers = table(
  {
    name: "workers",
    public: true,
  },
  {
    capabilities: t.string(), // Comma separated for simplicity in indexing
    lang: core_WorkerLanguage,
    lastHeartbeat: t.u64(),
    tagsJson: t.string(),
    workerId: t.string().primaryKey(),
  },
);

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
