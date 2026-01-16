import { create } from "@bufbuild/protobuf";
import { type ReducerCtx, t } from "spacetimedb/server";

import { ChatSyncMessage as ProtoChatSyncMessage } from "../generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatSyncMessageSchema } from "../generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatMessageSchema } from "../generated/flowcraft/v1/core/service_pb";
import { ChatMessage as StdbChatMessage } from "../generated/generated_schema";
import { pbToStdb } from "../generated/proto-stdb-bridge";
import { type AppSchema } from "../schema";

export const chatReducers = {
  add_chat_message: {
    args: {
      message: ChatSyncMessageSchema,
      nodeId: t.string(),
    },
    handler: (ctx: ReducerCtx<AppSchema>, { message, nodeId }: { message: ProtoChatSyncMessage; nodeId: string }) => {
      // Convert sync message to full message structure
      const fullMsg = create(ChatMessageSchema, {
        id: message.id,
        metadata: {
          case: "chatMetadata",
          value: { attachmentUrls: [], modelId: message.modelId },
        },
        parentId: "",
        parts: message.parts,
        role: message.role,
        siblingIds: [],
        timestamp: message.timestamp,
        treeId: nodeId,
      });

      ctx.db.chatMessages.insert({
        id: message.id,
        state: pbToStdb(ChatMessageSchema, StdbChatMessage, fullMsg) as StdbChatMessage,
      });
    },
  },

  clear_chat_history: {
    args: { nodeId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId }: { nodeId: string }) => {
      // In this version, treeId is tied to nodeId
      const messages = [...ctx.db.chatMessages.iter()];
      for (const msg of messages) {
        // msg.state is inferred from StdbChatMessage
        if (msg.state.treeId === nodeId) {
          ctx.db.chatMessages.id.delete(msg.id);
        }
      }
    },
  },

  update_chat_stream: {
    args: {
      content: t.string(),
      nodeId: t.string(),
      status: t.string(),
    },
    handler: (
      ctx: ReducerCtx<AppSchema>,
      { content, nodeId, status }: { content: string; nodeId: string; status: string },
    ) => {
      const existing = ctx.db.chatStreams.nodeId.find(nodeId);
      if (existing) {
        ctx.db.chatStreams.nodeId.update({ content, nodeId, status });
      } else {
        ctx.db.chatStreams.insert({ content, nodeId, status });
      }
    },
  },
};
