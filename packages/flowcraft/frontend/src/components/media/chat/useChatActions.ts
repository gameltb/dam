import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import OpenAI from "openai";
import { useCallback } from "react";
import { toast } from "react-hot-toast";
import { v4 as uuidv4 } from "uuid";

import {
  ChatActionParamsSchema,
  ChatEditParamsSchema,
  type ChatMessagePart,
  ChatMessagePartSchema,
  ChatSwitchBranchParamsSchema,
  ChatSyncBranchParamsSchema,
  type ChatSyncMessage,
  ChatSyncMessageSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { ClearChatHistoryRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import {
  type LocalLLMClientConfig,
  MutationSource,
  TaskStatus,
  TaskType,
} from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";

import { type ChatMessage, type ChatStatus, type ContextNode } from "./types";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const sendNodeSignal = useFlowStore((s) => s.sendNodeSignal);
  const { localClients } = useUiStore((s) => s.settings);
  const { registerTask, updateTask } = useTaskStore();

  const uploadFile = async (file: File): Promise<null | string> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/upload", {
        body: formData,
        method: "POST",
      });
      const asset = (await response.json()) as { url: string };
      return asset.url;
    } catch (err) {
      console.error("Upload failed:", err);
      return null;
    }
  };

  const performLocalInference = useCallback(
    async (
      localClient: LocalLLMClientConfig,
      selectedModel: string,
      userMsgId?: string,
      userParts?: ChatMessagePart[],
    ) => {
      const taskId = `local-${uuidv4()}`;
      try {
        registerTask({
          label: `Local Chat (${localClient.name})`,
          nodeId,
          source: MutationSource.SOURCE_LOCAL_TASK,
          status: TaskStatus.TASK_PROCESSING,
          taskId,
          type: TaskType.NODE_EXECUTION,
        });

        const client = new OpenAI({
          apiKey: localClient.apiKey || "no-key",
          baseURL: localClient.baseUrl,
          dangerouslyAllowBrowser: true,
        });

        const history = getHistory();
        const openaiMessages = mapHistoryToOpenAI(history);

        setStatus("streaming");
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
        setStatus("ready");
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

        sendNodeSignal(
          create(NodeSignalSchema, {
            nodeId,
            payload: {
              case: "chatSync",
              value: create(ChatSyncBranchParamsSchema, {
                anchorMessageId: history[history.length - 1]?.id ?? "",
                newMessages: newMessagesToSync,
                treeId: history[0]?.treeId ?? "",
              }),
            },
          }),
        );
      } catch (err) {
        console.error("Local inference failed:", err);
        const errorMessage = err instanceof Error ? err.message : String(err);
        toast.error(`Local inference failed: ${errorMessage}`);
        setStatus("ready");
        updateTask(taskId, {
          message: errorMessage,
          status: TaskStatus.TASK_FAILED,
        });
      }
    },
    [
      nodeId,
      setStatus,
      handleStreamChunk,
      getHistory,
      sendNodeSignal,
      registerTask,
      updateTask,
    ],
  );

  const sendMessage = useCallback(
    async (
      content: string,
      selectedModel: string,
      selectedEndpoint: string,
      useWebSearch: boolean,
      files: FileUIPart[] = [],
      contextNodes: ContextNode[] = [],
    ) => {
      setStatus("submitted");

      const finalAttachments: FileUIPart[] = [];
      for (const file of files) {
        if (file.url.startsWith("blob:")) {
          const response = await fetch(file.url);
          const blob = await response.blob();
          const url = await uploadFile(
            new File([blob], file.filename ?? "img.png", {
              type: file.mediaType,
            }),
          );
          if (url) finalAttachments.push({ ...file, url });
        } else {
          finalAttachments.push(file);
        }
      }

      const userMsgId = uuidv4();
      const userParts: ChatMessagePart[] = [
        create(ChatMessagePartSchema, {
          part: { case: "text", value: content.trim() },
        }),
      ];

      finalAttachments.forEach((att) => {
        userParts.push(
          create(ChatMessagePartSchema, {
            part: {
              case: "media",
              value: {
                aspectRatio: 0,
                content: "",
                galleryUrls: [],
                type: att.mediaType.startsWith("image")
                  ? MediaType.MEDIA_IMAGE
                  : MediaType.MEDIA_UNSPECIFIED,
                url: att.url,
              },
            },
          }),
        );
      });

      const userMsg: ChatMessage = {
        attachments: finalAttachments,
        contextNodes,
        createdAt: Date.now(),
        id: userMsgId,
        parts: userParts,
        role: "user",
      };

      appendUserMessage(userMsg);

      // Check if selectedEndpoint matches a local client
      const localClient = localClients.find((c) => c.id === selectedEndpoint);

      if (localClient) {
        await performLocalInference(
          localClient,
          selectedModel,
          userMsgId,
          userParts,
        );
        return;
      }

      try {
        sendNodeSignal(
          create(NodeSignalSchema, {
            nodeId,
            payload: {
              case: "chatGenerate",
              value: create(ChatActionParamsSchema, {
                endpointId: selectedEndpoint,
                modelId: selectedModel,
                userContent: content.trim(),
                useWebSearch: useWebSearch,
              }),
            },
          }),
        );
      } catch (err) {
        console.error("Send failed:", err);
        setStatus("ready");
        toast.error("Failed to send message");
      }
    },
    [
      setStatus,
      appendUserMessage,
      localClients,
      performLocalInference,
      nodeId,
      sendNodeSignal,
    ],
  );

  const editMessage = useCallback(
    (messageId: string, newParts: ChatMessagePart[]) => {
      sendNodeSignal(
        create(NodeSignalSchema, {
          nodeId,
          payload: {
            case: "chatEdit",
            value: create(ChatEditParamsSchema, {
              messageId,
              newParts,
            }),
          },
        }),
      );
    },
    [nodeId, sendNodeSignal],
  );

  const switchBranch = useCallback(
    (messageId: string) => {
      sendNodeSignal(
        create(NodeSignalSchema, {
          nodeId,
          payload: {
            case: "chatSwitch",
            value: create(ChatSwitchBranchParamsSchema, {
              targetMessageId: messageId,
            }),
          },
        }),
      );
    },
    [nodeId, sendNodeSignal],
  );

  const clearHistory = useCallback(async () => {
    const { socketClient } = await import("@/utils/SocketClient");
    await socketClient.send({
      payload: {
        case: "chatClear",
        value: create(ClearChatHistoryRequestSchema, {
          nodeId,
        }),
      },
    });
  }, [nodeId]);

  const continueChat = useCallback(
    async (selectedModel: string, selectedEndpoint: string) => {
      setStatus("submitted");

      // Check local client
      const localClient = localClients.find((c) => c.id === selectedEndpoint);
      if (localClient) {
        await performLocalInference(localClient, selectedModel);
        return;
      }

      try {
        sendNodeSignal(
          create(NodeSignalSchema, {
            nodeId,
            payload: {
              case: "chatGenerate",
              value: create(ChatActionParamsSchema, {
                endpointId: selectedEndpoint,
                modelId: selectedModel,
                userContent: "", // Empty tells backend to use current head
              }),
            },
          }),
        );
      } catch (err) {
        console.error("Continue failed:", err);
        setStatus("ready");
      }
    },
    [setStatus, localClients, performLocalInference, nodeId, sendNodeSignal],
  );

  return { clearHistory, continueChat, editMessage, sendMessage, switchBranch };
}
