import { t, table } from "spacetimedb/server";

export const tasks = table(
  {
    name: "tasks",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    idempotencyKey: t.string().index(),
    lastHeartbeat: t.u64(),
    nodeId: t.string(),
    ownerId: t.string(),
    paramsPayload: t.byteArray(),
    result: t.string(),
    status: Object.assign(t.u32(), { __pb_schema: "TaskStatus" }),
    taskType: t.string(),
    timestamp: t.u64(),
    version: t.u32(),
  },
);

export const workers = table(
  {
    name: "workers",
    public: true,
  },
  {
    capabilities: t.string(),
    lang: t.u32(),
    lastHeartbeat: t.u64(),
    tagsJson: t.string(),
    workerId: t.string().primaryKey(),
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

export const nodeRuntimeStates = table(
  {
    name: "node_runtime_states",
    public: true,
  },
  {
    activeUserId: t.option(t.string()),
    error: t.option(t.string()),
    lastUpdated: t.u64(),
    message: t.string(),
    nodeId: t.string().primaryKey(),
    progress: t.u32(),
    status: t.string(),
  },
);
