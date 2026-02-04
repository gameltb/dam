import { t, table } from "spacetimedb/server";

export const chatMessages = table(
  {
    name: "chat_messages",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    state: Object.assign(t.byteArray(), { __pb_schema: "ChatMessage" }),
    treeId: t.string().index(),
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
    status: t.string(),
  },
);
