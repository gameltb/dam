import { t } from "spacetimedb/server";


export const chatReducers = {
  add_chat_message: {
    args: {
      contentId: t.string(),
      id: t.string(),
      modelId: t.string(),
      nodeId: t.string(),
      parentId: t.string(),
      partsJson: t.string(),
      role: t.string(),
      timestamp: t.u64(),
      treeId: t.string(),
    },
    handler: (ctx: any, args: any) => {
      // 1. Structural Sharing: Check if content already exists
      const existingContent = ctx.db.chatContents.id.find(args.contentId);
      if (!existingContent) {
        ctx.db.chatContents.insert({
          id: args.contentId,
          partsJson: args.partsJson,
          role: args.role,
        });
      }

      // 2. Topology: Always insert the link
      ctx.db.chatMessages.insert({
        contentId: args.contentId,
        id: args.id,
        modelId: args.modelId,
        nodeId: args.nodeId,
        parentId: args.parentId,
        timestamp: args.timestamp,
        treeId: args.treeId,
      });
    },
  },

  clear_chat_history: {
    args: { nodeId: t.string() },
    handler: (ctx: any, { nodeId }: any) => {
      // Note: In CoW approach, we might want to keep contents.
      // We only delete topology links for the specific node.
      for (const msg of ctx.db.chatMessages) {
        if (msg.nodeId === nodeId) {
          ctx.db.chatMessages.id.delete(msg.id);
        }
      }
    },
  },

  update_chat_stream: {
    args: {
      content: t.string(),
      nodeId: t.string(),
      parentId: t.string(),
      status: t.string(),
    },
    handler: (ctx: any, args: any) => {
      const existing = ctx.db.chatStreams.nodeId.find(args.nodeId);
      if (existing) {
        ctx.db.chatStreams.nodeId.update({ ...existing, ...args });
      } else {
        ctx.db.chatStreams.insert(args);
      }
    },
  },
};
