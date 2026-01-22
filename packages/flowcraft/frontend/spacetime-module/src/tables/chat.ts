import { t, table } from "spacetimedb/server";

import { services_ChatMessage } from "../generated/generated_schema";

/**
 * Chat messages table using the Protobuf-defined structure.
 */
export const chatMessages = table(
  {
    name: "chat_messages",
    public: true,
  },
  {
    id: t.string().primaryKey(),
    state: services_ChatMessage,
  },
);

// Keep chatStreams for internal worker state.
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
