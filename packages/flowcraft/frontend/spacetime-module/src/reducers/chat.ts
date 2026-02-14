import { create, toBinary } from "@bufbuild/protobuf";
import { type ReducerCtx, t } from "spacetimedb/server";

import { type ChatSyncMessage as ProtoChatSyncMessage } from "../generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatSyncMessageSchema } from "../generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatMessageSchema } from "../generated/flowcraft/v1/core/service_pb";
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
        parentId: message.parentId,
        parts: message.parts,
        role: message.role,
        siblingIds: [],
        timestamp: message.timestamp,
        treeId: nodeId,
      });

      ctx.db.chatMessages.insert({
        id: message.id,
        state: toBinary(ChatMessageSchema, fullMsg),
        treeId: nodeId,
      });
    },
  },

  clear_chat_history: {
    args: { nodeId: t.string() },
    handler: (ctx: ReducerCtx<AppSchema>, { nodeId }: { nodeId: string }) => {
      // Optimization: Delete related messages directly using index
      for (const msg of ctx.db.chatMessages.treeId.filter(nodeId)) {
        ctx.db.chatMessages.id.delete(msg.id);
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
