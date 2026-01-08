import type OpenAI from "openai";

import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
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
import { isChatNode } from "@/types";
import { inferenceService } from "../services/InferenceService";
import { NodeInstance } from "../services/NodeInstance";
import { eventBus, serverGraph } from "../services/PersistenceService";
import { addChatMessage, getChatHistory } from "../services/PersistenceService";

export class ChatNodeInstance extends NodeInstance {
  private treeId = "";

  async handleSignal(payload: unknown): Promise<void> {
    const node = serverGraph.nodes.find((n) => n.id === this.nodeId);
    if (!node || !isChatNode(node)) return;

    const signal = payload as NodeSignal["payload"];

    if (signal.case === "chatSync") {
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
    } else if (signal.case === "chatGenerate") {
      const { endpointId, modelId, userContent } = signal.value;
      const chatData = node.data.extension.value;
      const userMsgId = uuidv4();

      addChatMessage({
        id: userMsgId,
        nodeId: node.id,
        parentId: chatData.conversationHeadId || null,
        parts: [
          create(ChatMessagePartSchema, {
            part: { case: "text", value: userContent },
          }),
        ],
        role: "user",
        treeId: this.treeId,
      });

      await this.generateResponse(userMsgId, modelId, endpointId);
    }
  }

  protected getDisplayLabel(): string {
    return `Chat Instance (${this.nodeId ?? "unknown"})`;
  }

  protected override getInstanceType(): string {
    return "NODE_INSTANCE";
  }

  protected onReady(_params: unknown): Promise<void> {
    const node = serverGraph.nodes.find((n) => n.id === this.nodeId);
    if (node && isChatNode(node)) {
      this.treeId = node.data.extension.value.treeId || uuidv4();
    }
    this.updateStatus(TaskStatus.TASK_PROCESSING, "Chat Instance Ready");
    return Promise.resolve();
  }

  private async generateResponse(
    headId: string,
    modelId: string,
    endpointId: string,
  ) {
    this.updateStatus(TaskStatus.TASK_PROCESSING, "AI is thinking...");
    const history = getChatHistory(headId);

    try {
      const messages: OpenAI.Chat.ChatCompletionMessageParam[] = history.map(
        (m) => ({
          content: m.parts
            .map((p) => (p.part.case === "text" ? p.part.value : ""))
            .join("\n"),
          role: (["assistant", "system", "user"].includes(m.role)
            ? m.role
            : "user") as "assistant" | "system" | "user",
        }),
      );

      const stream = await inferenceService.chatCompletion({
        endpointId,
        messages,
        model: modelId,
        stream: true,
      });

      if (!("controller" in stream)) throw new Error("Stream failed");

      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;
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
      this.updateStatus(
        TaskStatus.TASK_FAILED,
        err instanceof Error ? err.message : String(err),
      );
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
