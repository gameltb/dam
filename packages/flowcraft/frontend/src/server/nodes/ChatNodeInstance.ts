import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { NodeEventSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppNode, isChatNode, NodeSignalCase } from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";

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

        if (!userContent.trim() && chatData.conversationHeadId) {
          const history = await getChatHistory(this.treeId);
          const headMsg = history.pop();
          if (headMsg?.role === "user") {
            await this.generateResponse(chatData.conversationHeadId, modelId, endpointId);
            return;
          }
        }

        await addChatMessage(node.id, "user", userContent);
        await this.generateResponse(chatData.conversationHeadId, modelId, endpointId);
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

  private async generateResponse(headId: string, modelId: string, endpointId: string) {
    logger.info(`generateResponse started. Head: ${headId}, Model: ${modelId}`);
    this.updateStatus(TaskStatus.RUNNING, "AI is thinking...");
    const conn = getSpacetimeConn();

    if (conn) {
      conn.reducers.updateChatStream({
        content: "",
        nodeId: this.nodeId ?? "",
        status: "thinking",
      });
    }

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

      if (conn) {
        conn.reducers.updateChatStream({
          content: "",
          nodeId: this.nodeId ?? "",
          status: "streaming",
        });
      }

      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;
          if (conn) {
            conn.reducers.updateChatStream({
              content: fullContent,
              nodeId: this.nodeId ?? "",
              status: "streaming",
            });
          }

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

      await addChatMessage(this.nodeId ?? "unknown", "assistant", fullContent);

      if (conn) {
        conn.reducers.updateChatStream({
          content: "",
          nodeId: this.nodeId ?? "",
          status: "idle",
        });
      }

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

      this.updateStatus(TaskStatus.RUNNING, "AI Answered");
    } catch (err: unknown) {
      logger.error(`Generation failed:`, err);
      if (conn) {
        conn.reducers.updateChatStream({
          content: "",
          nodeId: this.nodeId ?? "",
          status: "idle",
        });
      }

      this.updateStatus(TaskStatus.FAILED, err instanceof Error ? err.message : String(err));
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
