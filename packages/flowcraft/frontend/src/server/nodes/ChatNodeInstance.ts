import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { NodeEventSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppNode, isChatNode, NodeSignalCase } from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";
import { wrapReducers } from "@/utils/pb-client";

import { addChatMessage, getChatHistory } from "../services/ChatService";
import { inferenceService } from "../services/InferenceService";
import { NodeInstance } from "../services/NodeInstance";
import { eventBus } from "../services/PersistenceService";
import { getSpacetimeConn } from "../spacetimeClient";
import logger from "../utils/logger";

export class ChatNodeInstance extends NodeInstance {
  private treeId = "";

  async handleSignal(payload: unknown): Promise<void> {
    const signal = payload as NodeSignal["payload"];
    logger.info(`Dispatching signal case: ${String(signal.case)}`);

    const conn = getSpacetimeConn();
    if (!conn || !this.nodeId) return;

    const stNode = conn.db.nodes.nodeId.find(this.nodeId);
    if (!stNode) return;

    const nodeData = stNode.state.state;
    if (!nodeData) return;
    const node = {
      data: nodeData,
      id: this.nodeId,
      type: "dynamic",
    } as unknown as AppNode;

    if (!isChatNode(node)) return;

    const extension = nodeData.extension as any;
    if (!extension) return;
    const chatData = extension.chat || (extension.value && extension.tag === "chat" ? extension.value : undefined);
    if (!chatData) return;

    const handlers: Partial<Record<NodeSignalCase, () => Promise<void> | void>> = {
      [NodeSignalCase.CHAT_EDIT]: () => {
        if (signal.case !== "chatEdit") return;
      },
      [NodeSignalCase.CHAT_GENERATE]: async () => {
        if (signal.case !== "chatGenerate") return;
        const { endpointId, modelId, userContent } = signal.value;

        await this.runTask("chat.generate", { endpointId, modelId, userContent }, async (ctx) => {
          let currentHeadId = chatData.conversationHeadId;

          if (userContent.trim()) {
            const userMsgId = await addChatMessage(node.id, "user", userContent.trim(), currentHeadId);
            if (userMsgId) {
              currentHeadId = userMsgId;
              this.updateNodeHead(userMsgId);
            }
          } else if (!currentHeadId) {
            return;
          }

          await this.generateResponse(currentHeadId, modelId, endpointId, ctx);
        });
      },
      [NodeSignalCase.CHAT_SWITCH]: () => {
        if (signal.case !== "chatSwitch") return;
        this.updateNodeHead(signal.value.targetMessageId);
      },
      [NodeSignalCase.CHAT_SYNC]: async () => {
        if (signal.case !== "chatSync") return;
      },
      [NodeSignalCase.RESTART_INSTANCE]: () => {},
    };

    const caseKey = signal.case as NodeSignalCase;
    const handler = handlers[caseKey];
    if (handler) await handler();
  }

  protected getDisplayLabel(): string {
    return `Chat Instance (${this.nodeId ?? "unknown"})`;
  }

  protected override getInstanceType(): string {
    return "NODE_INSTANCE";
  }

  protected onReady(_params: unknown): Promise<void> {
    const conn = getSpacetimeConn();
    if (conn && this.nodeId) {
      const stNode = conn.db.nodes.nodeId.find(this.nodeId);
      if (stNode) {
        try {
          const nodeData = stNode.state.state;
          if (nodeData) {
            const extension = nodeData.extension as any;
            if (extension) {
              const chatData =
                extension.chat || (extension.value && extension.tag === "chat" ? extension.value : undefined);
              if (chatData) {
                this.treeId = chatData.treeId || uuidv4();
              }
            }
          }
        } catch (e) {
          logger.error("Failed to parse node data on ready", e);
        }
      }
    }
    this.updateStatus(TaskStatus.RUNNING, "Chat Instance Ready");
    return Promise.resolve();
  }

  private async generateResponse(headId: string, modelId: string, endpointId: string, ctx: any) {
    const genTaskId = ctx.taskId;
    logger.info(`generateResponse started for task ${genTaskId}. Head: ${headId}`);

    const conn = getSpacetimeConn();
    if (!conn) return;
    const pbConn = wrapReducers(conn as any);

    let fullContent = "";

    await ctx.updateProgress(10, "AI is thinking...");

    pbConn.pbreducers.updateChatStream({
      content: "",
      nodeId: this.nodeId ?? "",
      status: "thinking",
    });

    const history = await getChatHistory(this.treeId);

    try {
      const messages = mapHistoryToOpenAI(history);

      const stream = await inferenceService.chatCompletion({
        endpointId,
        messages,
        model: modelId,
        stream: true,
      });

      if (!("controller" in stream)) throw new Error("Stream failed");

      pbConn.pbreducers.updateChatStream({
        content: "",
        nodeId: this.nodeId ?? "",
        status: "streaming",
      });

      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;
          pbConn.pbreducers.updateChatStream({
            content: fullContent,
            nodeId: this.nodeId ?? "",
            status: "streaming",
          });

          eventBus.emit(
            "nodeEvent",
            create(NodeEventSchema, {
              nodeId: this.nodeId ?? "unknown",
              payload: {
                case: "chatStream",
                value: { chunkData: content, isDone: false, messageId: "" },
              },
            }),
          );
        }
      }

      await addChatMessage(this.nodeId ?? "unknown", "assistant", fullContent, headId);

      pbConn.pbreducers.updateChatStream({
        content: "",
        nodeId: this.nodeId ?? "",
        status: "idle",
      });

      eventBus.emit(
        "nodeEvent",
        create(NodeEventSchema, {
          nodeId: this.nodeId ?? "unknown",
          payload: {
            case: "chatStream",
            value: { chunkData: "", isDone: true, messageId: "" },
          },
        }),
      );

      await ctx.complete("Success");
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : String(err);
      logger.error(`Generation failed for task ${genTaskId}:`, err);

      pbConn.pbreducers.updateChatStream({
        content: fullContent,
        nodeId: this.nodeId ?? "",
        status: "error",
      });

      // Save partial content if available to ensure it persists after refresh
      if (fullContent.trim()) {
        const assistantMsgId = await addChatMessage(
          this.nodeId ?? "unknown",
          "assistant",
          `${fullContent}\n\n[Generation Interrupted]`,
          headId,
        );
        if (assistantMsgId) {
          this.updateNodeHead(assistantMsgId);
        }
      }

      await pbConn.pbreducers.failTask({
        error: `Generation Error: ${errorMsg}`,
        taskId: genTaskId,
      });

      throw err;
    }
  }

  private updateNodeHead(headId: string) {
    if (this.nodeId) {
      const res = this.nodeDraft(this.nodeId);
      if (res.ok) {
        const draft = res.value;
        if (draft.data?.extension?.case === "chat") {
          draft.data.extension.value.conversationHeadId = headId;
        }
      }
    }
  }
}
