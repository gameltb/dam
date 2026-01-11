import { create, fromJson } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import {
  ChatMessagePartSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import {
  NodeDataSchema,
  TaskStatus,
} from "@/generated/flowcraft/v1/core/node_pb";
import {
  NodeEventSchema,
  UpdateNodeSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { isChatNode, NodeSignalCase } from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";

import { inferenceService } from "../services/InferenceService";
import { NodeInstance } from "../services/NodeInstance";
import { getSpacetimeConn } from "../spacetimeClient";
import {
  addChatMessage,
  branchAndEditMessage,
  eventBus,
  getChatHistory,
} from "../services/PersistenceService";
import logger from "../utils/logger";

export class ChatNodeInstance extends NodeInstance {
  private treeId = "";

  async handleSignal(payload: unknown): Promise<void> {
    const signal = payload as NodeSignal["payload"];
    logger.info(`Dispatching signal case: ${signal.case}`);

    const conn = getSpacetimeConn();
    if (!conn || !this.nodeId) return;

    const stNode = conn.db.nodes.id.find(this.nodeId);
    if (!stNode) return;

    const nodeData = fromJson(NodeDataSchema, JSON.parse(stNode.dataJson));
    const node = { id: this.nodeId, type: "dynamic", data: nodeData } as any;

    if (!isChatNode(node)) return;

    const handlers: {
      [K in NodeSignalCase]?: () => Promise<void> | void;
    } = {
      [NodeSignalCase.CHAT_SYNC]: () => {
        if (signal.case !== "chatSync") return;
        const { anchorMessageId, newMessages, treeId } = signal.value;
        let lastId = anchorMessageId;
        for (const msg of newMessages) {
          addChatMessage({
            id: msg.id || uuidv4(),
            metadata: { modelId: msg.modelId },
            nodeId: node.id,
            parentId: lastId || null,
            parts: msg.parts,
            role: msg.role,
            treeId: treeId || this.treeId,
          });
          lastId = msg.id;
        }
        this.updateNodeHead(treeId || this.treeId, lastId);
      },
      [NodeSignalCase.CHAT_GENERATE]: async () => {
        if (signal.case !== "chatGenerate") return;
        const { endpointId, modelId, userContent } = signal.value;
        const chatData = (node.data as any).extension.value;

        if (!userContent.trim() && chatData.conversationHeadId) {
          const headMsg = getChatHistory(chatData.conversationHeadId).pop();
          if (headMsg?.role === "user") {
            await this.generateResponse(chatData.conversationHeadId, modelId, endpointId);
            return;
          }
        }

        const userMsgId = uuidv4();
        addChatMessage({
          id: userMsgId,
          nodeId: node.id,
          parentId: chatData.conversationHeadId || null,
          parts: [create(ChatMessagePartSchema, { part: { case: "text", value: userContent } })],
          role: "user",
          treeId: this.treeId,
        });
        await this.generateResponse(userMsgId, modelId, endpointId);
      },
      [NodeSignalCase.CHAT_EDIT]: () => {
        if (signal.case !== "chatEdit") return;
        const { messageId, newParts } = signal.value;
        const newHeadId = branchAndEditMessage({
          messageId,
          newParts,
          nodeId: node.id,
          treeId: this.treeId,
        });
        this.updateNodeHead(this.treeId, newHeadId);
      },
      [NodeSignalCase.CHAT_SWITCH]: () => {
        if (signal.case !== "chatSwitch") return;
        this.updateNodeHead(this.treeId, signal.value.targetMessageId);
      },
      [NodeSignalCase.RESTART_INSTANCE]: () => {
        // Handled by runNodeSignal before reaching here
      },
    };

    const caseKey = signal.case as NodeSignalCase;
    if (caseKey && handlers[caseKey]) {
      const handler = handlers[caseKey];
      if (handler) await handler();
    }
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
      const stNode = conn.db.nodes.id.find(this.nodeId);
      if (stNode) {
        try {
          const nodeData = fromJson(
            NodeDataSchema,
            JSON.parse(stNode.dataJson),
          );
          // Manual check since we don't have isChatNode helper handy without casting
          if (nodeData.extension.case === "chat") {
            this.treeId = nodeData.extension.value.treeId || uuidv4();
          }
        } catch (e) {
          logger.error("Failed to parse node data on ready", e);
        }
      }
    }
    this.updateStatus(TaskStatus.TASK_PROCESSING, "Chat Instance Ready");
    return Promise.resolve();
  }

  private async generateResponse(
    headId: string,
    modelId: string,
    endpointId: string,
  ) {
    logger.info(`generateResponse started. Head: ${headId}, Model: ${modelId}`);
    this.updateStatus(TaskStatus.TASK_PROCESSING, "AI is thinking...");
    const conn = getSpacetimeConn();

    // 1. Set Thinking State
    if (conn) {
      conn.reducers.updateChatStream({
        content: "",
        nodeId: this.nodeId ?? "",
        parentId: headId,
        status: "thinking",
      });
    }

    const history = getChatHistory(headId);

    try {
      const messages = mapHistoryToOpenAI(history);

      logger.info(`Calling InferenceService...`);
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
          parentId: headId,
          status: "streaming",
        });
      }

      logger.info(`Stream started. Reading chunks...`);
      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;

          // Update stream in DB
          if (conn) {
            conn.reducers.updateChatStream({
              content: fullContent,
              nodeId: this.nodeId ?? "",
              parentId: headId,
              status: "streaming",
            });
          }

          // Legacy event
          eventBus.emit(
            "nodeEvent",
            create(NodeEventSchema, {
              nodeId: this.nodeId ?? "unknown",
              payload: {
                case: "chatStream",
                value: { chunkData: content, isDone: false },
              },
            }),
          );
        }
      }
      logger.info(
        `Stream completed. Full content length: ${fullContent.length}`,
      );

      const aiMsgId = uuidv4();
      addChatMessage({
        id: aiMsgId,
        metadata: { modelId },
        nodeId: this.nodeId ?? "unknown",
        parentId: headId,
        parts: [
          create(ChatMessagePartSchema, {
            part: { case: "text", value: fullContent },
          }),
        ],
        role: "assistant",
        treeId: this.treeId,
      });

      // Clear stream state
      if (conn) {
        conn.reducers.updateChatStream({
          content: "",
          nodeId: this.nodeId ?? "",
          parentId: headId,
          status: "idle",
        });
      }

      eventBus.emit(
        "nodeEvent",
        create(NodeEventSchema, {
          nodeId: this.nodeId ?? "unknown",
          payload: {
            case: "chatStream",
            value: { chunkData: "", isDone: true },
          },
        }),
      );

      this.updateNodeHead(this.treeId, aiMsgId);
      this.updateStatus(TaskStatus.TASK_PROCESSING, "AI Answered");
      this.schedulePersistence();
    } catch (err: unknown) {
      logger.error(`Generation failed:`, err);
      // Clear stream state on error
      if (conn) {
        conn.reducers.updateChatStream({
          content: "",
          nodeId: this.nodeId ?? "",
          parentId: headId,
          status: "idle",
        });
      }

      this.updateStatus(
        TaskStatus.TASK_FAILED,
        err instanceof Error ? err.message : String(err),
      );
      throw err;
    }
  }

  private updateNodeHead(treeId: string, headId: string) {
    this.emitMutation(
      {
        case: "updateNode",
        value: create(UpdateNodeSchema, {
          data: create(NodeDataSchema, {
            extension: {
              case: "chat",
              value: {
                conversationHeadId: headId,
                isHistoryCleared: false,
                treeId: treeId,
              },
            },
          }),
          id: this.nodeId ?? "unknown",
        }),
      },
      MutationSource.SOURCE_REMOTE_TASK,
    );
  }
}
