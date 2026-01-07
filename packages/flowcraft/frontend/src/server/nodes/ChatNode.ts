import type OpenAI from "openai";

import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import {
  ActionExecutionStrategy,
  ActionTemplateSchema,
} from "../../generated/flowcraft/v1/core/action_pb";
import { MutationSource } from "../../generated/flowcraft/v1/core/base_pb";
import {
  NodeDataSchema,
  NodeTemplateSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import {
  GraphMutationSchema,
  MutationListSchema,
  NodeEventSchema,
} from "../../generated/flowcraft/v1/core/service_pb";
import { type DynamicNodeData } from "../../types";
import { aiService } from "../aiService";
import { addChatMessage, getChatHistory } from "../db";
import { type NodeExecutionContext, NodeRegistry } from "../registry";

NodeRegistry.register({
  actions: [
    create(ActionTemplateSchema, {
      id: "flowcraft.action.node.chat.generate",
      label: "Generate Chat Response",
      strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
    }),
  ],
  execute: async (ctx: NodeExecutionContext) => {
    const { actionId, emitMutation, emitNodeEvent, node, params } = ctx;

    if (actionId !== "flowcraft.action.node.chat.generate") return;

    // Strong typed params from oneof
    if (params.case !== "chatGenerate") return;
    const { endpointId, modelId, userContent } = params.value;

    // 1. 获取当前节点状态
    const data = node.data as DynamicNodeData;
    const chatState =
      data.extension?.case === "chat" ? data.extension.value : null;
    const currentHeadId = chatState?.conversationHeadId;

    // 2. 重建历史
    const history = currentHeadId ? getChatHistory(currentHeadId) : [];

    // 为用户消息创建持久化记录 (COW)
    const userMsgId = uuidv4();
    addChatMessage({
      content: userContent,
      id: userMsgId,
      nodeId: node.id,
      parentId: currentHeadId ?? null,
      role: "user",
    });

    try {
      const messages: OpenAI.Chat.ChatCompletionMessageParam[] = [
        ...history.map((m) => ({
          content: m.content,
          role: (["assistant", "system", "user"].includes(m.role)
            ? m.role
            : "user") as "assistant" | "system" | "user",
        })),
        { content: userContent, role: "user" as const },
      ];

      const stream = await aiService.chatCompletion({
        endpointId,
        messages,
        model: modelId,
        stream: true,
      });

      if (!("controller" in stream)) {
        // Handle non-stream response if it ever happens
        throw new Error("Expected stream response");
      }

      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta.content ?? "";
        if (content) {
          fullContent += content;
          emitNodeEvent(
            create(NodeEventSchema, {
              nodeId: node.id,
              payload: {
                case: "chatStream",
                value: {
                  chunkData: content,
                  isDone: false,
                },
              },
            }),
          );
        }
      }

      // 4. 为 AI 回复创建持久化记录 (COW)
      const aiMsgId = uuidv4();
      addChatMessage({
        content: fullContent,
        id: aiMsgId,
        nodeId: node.id,
        parentId: userMsgId,
        role: "assistant",
      });

      // 5. 通过强类型 Extension 更新节点的 HEAD 引用
      const mutationList = create(MutationListSchema, {
        mutations: [
          create(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                data: create(NodeDataSchema, {
                  extension: {
                    case: "chat",
                    value: {
                      conversationHeadId: aiMsgId,
                    },
                  },
                }),
                id: node.id,
              },
            },
          }),
        ],
        source: MutationSource.SOURCE_REMOTE_TASK,
      });

      emitMutation(mutationList);

      // Final done message
      emitNodeEvent(
        create(NodeEventSchema, {
          nodeId: node.id,
          payload: {
            case: "chatStream",
            value: {
              chunkData: "",
              isDone: true,
            },
          },
        }),
      );
    } catch (err) {
      console.error("Chat generation failed:", err);
      emitNodeEvent(
        create(NodeEventSchema, {
          nodeId: node.id,
          payload: {
            case: "log",
            value: {
              level: 2, // ERROR
              message: `Error: ${err instanceof Error ? err.message : "Unknown error"}`,
            },
          },
        }),
      );
    }
  },
  template: create(NodeTemplateSchema, {
    defaultState: create(NodeDataSchema, {
      activeMode: RenderMode.MODE_CHAT,
      availableModes: [RenderMode.MODE_CHAT, RenderMode.MODE_WIDGETS],
      displayName: "AI Assistant",
      extension: {
        case: "chat",
        value: {
          conversationHeadId: "",
        },
      },
    }),
    displayName: "AI Conversational Assistant",
    menuPath: ["AI"],
    templateId: "flowcraft.node.ai.conversational",
  }),
});
