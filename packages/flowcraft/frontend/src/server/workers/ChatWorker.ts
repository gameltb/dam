import { BaseWorker } from "@/kernel/BaseWorker";
import { type TaskContext } from "@/kernel/TaskContext";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";
import { type PbConnection } from "@/utils/pb-client";

import { addChatMessage, getChatHistory } from "../services/ChatService";
import { inferenceService } from "../services/InferenceService";
import logger from "../utils/logger";

export class ChatWorker extends BaseWorker {
  constructor(conn: PbConnection) {
    super(conn, ["chat.openai"]);
  }

  async handleTask(type: string, ctx: TaskContext): Promise<void> {
    if (type !== "chat.openai") return;

    const { nodeId, params } = ctx;
    const { endpointId, modelId } = params as any;

    logger.info(`[ChatWorker] Starting generation for node ${nodeId}`);

    await ctx.updateProgress(10, "Initializing...");

    const stNode = this.conn.db.nodes.nodeId.find(nodeId);
    if (!stNode) {
      await ctx.fail(`Node ${nodeId} not found`);
      return;
    }

    const extension = stNode.state.state?.extension as any;
    const chatData = extension?.value && extension?.tag === "chat" ? extension.value : null;
    const treeId = chatData?.treeId || nodeId;
    const parentId = chatData?.conversationHeadId || "";

    await ctx.updateProgress(20, "Fetching history...");
    const history = await getChatHistory(treeId);
    const messages = mapHistoryToOpenAI(history);

    try {
      await ctx.updateProgress(30, "AI is thinking...");
      this.conn.pbreducers.updateChatStream({
        content: "",
        nodeId: nodeId,
        status: "thinking",
      });

      const stream = await inferenceService.chatCompletion({
        endpointId,
        messages,
        model: modelId,
        stream: true,
      });

      if (!("controller" in stream)) throw new Error("Stream failed");

      this.conn.pbreducers.updateChatStream({
        content: "",
        nodeId: nodeId,
        status: "streaming",
      });

      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;
          this.conn.pbreducers.updateChatStream({
            content: fullContent,
            nodeId: nodeId,
            status: "streaming",
          });
        }
      }

      await ctx.updateProgress(90, "Saving response...");
      const assistantMsgId = await addChatMessage(nodeId, "assistant", fullContent, parentId);

      // Update node head
      if (assistantMsgId) {
        const res = this.kernel.nodeDraft(nodeId);
        if (res.ok) {
          const draft = res.value;
          if (draft.data.extension?.case === "chat") {
            draft.data.extension.value.conversationHeadId = assistantMsgId;
          }
        }
      }

      this.conn.pbreducers.updateChatStream({
        content: "",
        nodeId: nodeId,
        status: "idle",
      });

      await ctx.complete({ assistantMsgId, content: fullContent });
    } catch (err: any) {
      logger.error(`[ChatWorker] Generation failed:`, err);
      this.conn.pbreducers.updateChatStream({
        content: "",
        nodeId: nodeId,
        status: "idle",
      });
      await ctx.fail(err.message || String(err));
    }
  }
}
