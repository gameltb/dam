import { create } from "@bufbuild/protobuf";
import OpenAI from "openai";
import { useCallback } from "react";
import { toast } from "react-hot-toast";
import { v4 as uuidv4 } from "uuid";

import {
  type ChatMessagePart,
  ChatMessagePartSchema,
  type ChatSyncMessage,
  ChatSyncMessageSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { ChatStatus, type LocalLLMClientConfig, TaskStatus, TaskType } from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";

import { type ChatMessage } from "./types";

export function useLocalInference(nodeId: string) {
  const { localClients } = useUiStore((s) => s.settings);
  const { registerTask, updateTask } = useTaskStore();

  const performLocalInference = useCallback(
    async (
      localClient: LocalLLMClientConfig,
      selectedModel: string,
      getHistory: () => ChatMessage[],
      setStatus: (s: ChatStatus) => void,
      handleStreamChunk: (chunk: string) => void,
      userMsgId?: string,
      userParts?: ChatMessagePart[],
    ) => {
      const taskId = `local-${uuidv4()}`;
      try {
        registerTask({
          label: `Local Chat (${localClient.name})`,
          nodeId,
          source: MutationSource.SOURCE_USER,
          status: TaskStatus.TASK_PROCESSING,
          taskId,
          type: TaskType.REMOTE,
        });

        const client = new OpenAI({
          apiKey: localClient.apiKey || "no-key",
          baseURL: localClient.baseUrl,
          dangerouslyAllowBrowser: true,
        });

        const history = getHistory();
        const openaiMessages = mapHistoryToOpenAI(history);

        setStatus(ChatStatus.STREAMING);
        updateTask(taskId, { message: "Connecting to local LLM..." });

        const stream = await client.chat.completions.create({
          messages: openaiMessages,
          model: selectedModel || localClient.model,
          stream: true,
        });

        updateTask(taskId, { message: "Streaming response..." });

        let fullContent = "";
        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta.content ?? "";
          if (delta) {
            fullContent += delta;
            handleStreamChunk(delta);
          }
        }

        const aiMsgId = uuidv4();
        setStatus(ChatStatus.READY);
        updateTask(taskId, {
          message: "Generation complete",
          status: TaskStatus.TASK_COMPLETED,
        });

        const newMessagesToSync: ChatSyncMessage[] = [];
        if (userMsgId && userParts) {
          newMessagesToSync.push(
            create(ChatSyncMessageSchema, {
              id: userMsgId,
              parts: userParts,
              role: "user",
              timestamp: BigInt(Date.now()),
            }),
          );
        }

        newMessagesToSync.push(
          create(ChatSyncMessageSchema, {
            id: aiMsgId,
            modelId: selectedModel || localClient.model,
            parts: [
              create(ChatMessagePartSchema, {
                part: { case: "text", value: fullContent },
              }),
            ],
            role: "assistant",
            timestamp: BigInt(Date.now()),
          }),
        );

        const spacetimeConn = useFlowStore.getState().spacetimeConn;
        if (spacetimeConn) {
          for (const msg of newMessagesToSync) {
            spacetimeConn.pbreducers.addChatMessage({
              message: msg,
              nodeId: nodeId,
            });
          }
        }
      } catch (err) {
        console.error("Local inference failed:", err);
        const errorMessage = err instanceof Error ? err.message : String(err);
        toast.error(`Local inference failed: ${errorMessage}`);
        setStatus(ChatStatus.READY);
        updateTask(taskId, {
          message: errorMessage,
          status: TaskStatus.TASK_FAILED,
        });
      }
    },
    [nodeId, registerTask, updateTask],
  );

  return { localClients, performLocalInference };
}
