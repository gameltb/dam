import { create } from "@bufbuild/protobuf";
import {
  NodeTemplateSchema,
  NodeDataSchema,
  RenderMode,
} from "../../generated/flowcraft/v1/core/node_pb";
import {
  StreamChunkSchema,
  GraphMutationSchema,
  MutationListSchema,
} from "../../generated/flowcraft/v1/core/service_pb";
import { NodeRegistry } from "../registry";
import {
  ActionTemplateSchema,
  ActionExecutionStrategy,
} from "../../generated/flowcraft/v1/core/action_pb";
import { aiService } from "../aiService";
import { addChatMessage, getChatHistory } from "../db";
import { v4 as uuidv4 } from "uuid";
import { PathUpdate_UpdateType } from "../../generated/flowcraft/v1/core/service_pb";
import { MutationSource } from "../../generated/flowcraft/v1/core/base_pb";

NodeRegistry.register({
  template: create(NodeTemplateSchema, {
    templateId: "tpl-ai-chat",
    displayName: "AI Chat",
    menuPath: ["AI"],
    defaultState: create(NodeDataSchema, {
      displayName: "AI Assistant",
      availableModes: [RenderMode.MODE_CHAT, RenderMode.MODE_WIDGETS],
      activeMode: RenderMode.MODE_CHAT,
      metadata: {
        conversation_head_id: "",
      },
    }),
  }),
  actions: [
    create(ActionTemplateSchema, {
      id: "chat:generate",
      label: "Generate Chat",
      strategy: ActionExecutionStrategy.EXECUTION_IMMEDIATE,
    }),
  ],
  execute: async (ctx) => {
    const { actionId, params, emitStreamChunk, emitMutation, node } = ctx;

    if (actionId !== "chat:generate") return;
    if (!node) return;

    const { messages: clientMessages, model, endpointId } = params as any;

    // 1. 获取当前节点状态
    const metadata = (node.data?.metadata || {}) as any;
    const currentHeadId = metadata.conversation_head_id as string | undefined;

    // 2. 重建历史
    const history = currentHeadId ? getChatHistory(currentHeadId) : [];

    // 3. 处理新消息 (假设 clientMessages 的最后一条是用户新发送的)
    const lastUserMsg = Array.isArray(clientMessages)
      ? clientMessages[clientMessages.length - 1]
      : null;
    if (!lastUserMsg) return;

    // 为用户消息创建持久化记录 (COW)
    const userMsgId = uuidv4();
    addChatMessage({
      id: userMsgId,
      parentId: currentHeadId || null,
      role: "user",
      content: lastUserMsg.content,
      nodeId: node.id,
    });

    try {
      const stream = (await aiService.chatCompletion({
        model,
        endpointId,
        messages: [
          ...history.map((m: any) => ({ role: m.role, content: m.content })),
          { role: "user", content: lastUserMsg.content },
        ],
        stream: true,
      })) as any;

      let fullContent = "";
      for await (const chunk of stream) {
        const content = chunk.choices[0]?.delta?.content || "";
        if (content) {
          fullContent += content;
          emitStreamChunk(
            create(StreamChunkSchema, {
              nodeId: node.id,
              chunkData: content,
              isDone: false,
            }),
          );
        }
      }

      // 4. 为 AI 回复创建持久化记录 (COW)
      const aiMsgId = uuidv4();
      addChatMessage({
        id: aiMsgId,
        parentId: userMsgId,
        role: "assistant",
        content: fullContent,
        nodeId: node.id,
      });

      // 5. 通过细粒度突变 (PathUpdate) 更新节点的 HEAD 引用
      const mutationList = create(MutationListSchema, {
        mutations: [
          create(GraphMutationSchema, {
            operation: {
              case: "pathUpdate",
              value: {
                targetId: node.id,
                path: "data.metadata.conversation_head_id",
                valueJson: JSON.stringify(aiMsgId),
                type: PathUpdate_UpdateType.REPLACE,
              },
            },
          }),
        ],
        source: MutationSource.SOURCE_REMOTE_TASK,
      });

      emitMutation(mutationList);

      // Final done message
      emitStreamChunk(
        create(StreamChunkSchema, {
          nodeId: node.id,
          chunkData: "",
          isDone: true,
        }),
      );
    } catch (err: any) {
      console.error("Chat generation failed:", err);
      emitStreamChunk(
        create(StreamChunkSchema, {
          nodeId: node.id,
          chunkData: `Error: ${err.message}`,
          isDone: true,
        }),
      );
    }
  },
});
