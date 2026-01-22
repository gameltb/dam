import { t, table } from "spacetimedb/server";

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
    status: t.string(), // "idle" | "busy" | "error"
  },
);
