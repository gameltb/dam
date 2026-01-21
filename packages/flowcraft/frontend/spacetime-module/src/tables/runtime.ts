import { t, table } from "spacetimedb/server";

export const nodeRuntimeStates = table(
  {
    name: "node_runtime_states",
    public: true,
  },
  {
    nodeId: t.string().primaryKey(),
    status: t.string(), // "idle" | "busy" | "error"
    progress: t.u32(),
    message: t.string(),
    error: t.option(t.string()),
    lastUpdated: t.u64(),
    activeUserId: t.option(t.string()),
  },
);
