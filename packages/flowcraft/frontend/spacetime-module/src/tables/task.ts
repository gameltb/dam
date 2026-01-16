import { t, table } from "spacetimedb/server";

import { ActionExecutionRequest, NodeSignal, TaskStatus, Value } from "../generated/generated_schema";

export const tasks = table(
  {
    name: "tasks",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    request: ActionExecutionRequest, // Store full PB request
    result: Value,
    status: TaskStatus,
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
    payload: NodeSignal.elements.payload, // Protobuf Enum/Union
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
