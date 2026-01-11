import { t, table } from "spacetimedb/server";

export const tasks = table(
  {
    name: "tasks",
    public: true,
  },
  {
    actionId: t.string(),
    id: t.string().primaryKey(),
    nodeId: t.string(),
    paramsJson: t.string(),
    resultJson: t.string(),
    status: t.string(),
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
    payloadJson: t.string(),
    signalCase: t.string(),
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
    valueJson: t.string(),
    widgetId: t.string(),
  },
);

export const operationLogs = table(
  {
    name: "operation_logs",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    clientIdentity: t.string(), // Hex string of ctx.sender
    operationType: t.string(),  // e.g., "move_node", "add_chat_message"
    payloadJson: t.string(),
    taskId: t.string(),         // The originTaskId fetched from mapping
    timestamp: t.u64(),
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


