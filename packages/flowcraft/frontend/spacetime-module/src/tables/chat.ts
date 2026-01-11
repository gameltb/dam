import { t, table } from "spacetimedb/server";

/**
 * Structural Sharing: Content table stores the heavy payload.
 */
export const chatContents = table(
  {
    name: "chat_contents",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    partsJson: t.string(),
    role: t.string(),
  },
);

/**
 * Topology table stores the tree structure and references content.
 */
export const chatMessages = table(
  {
    name: "chat_messages",
    public: true,
  },
  {
    contentId: t.string(), // Reference to chat_contents
    id: t.string().primaryKey(),
    modelId: t.string(),
    nodeId: t.string(),
    parentId: t.string(),
    timestamp: t.u64(),
    treeId: t.string(),
  },
);

export const chatStreams = table(
  {
    name: "chat_streams",
    public: true,
  },
  {
    content: t.string(),
    nodeId: t.string().primaryKey(),
    parentId: t.string(),
    status: t.string(),
  },
);
