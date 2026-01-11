import { create, toJson } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
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
import { MediaType, MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import {
  ActionId,
  ChatStatus,
  type DynamicNodeData,
  type LocalLLMClientConfig,
  TaskStatus,
  TaskType,
} from "@/types";
import { mapHistoryToOpenAI } from "@/utils/chatUtils";

import { type ChatMessage, type ContextNode } from "./types";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));
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

        const newMessagesToSync: (ChatSyncMessage & { contentId?: string })[] = [];
        if (userMsgId && userParts) {
          newMessagesToSync.push(
            create(ChatSyncMessageSchema, {
              id: userMsgId,
              parts: userParts,
              role: "user",
              timestamp: BigInt(Date.now()),
            }) as any,
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
          }) as any,
        );

        if (useFlowStore.getState().spacetimeConn) {
           const conn = useFlowStore.getState().spacetimeConn!;
           for (const msg of newMessagesToSync) {
              const partsJson = JSON.stringify(
                (msg.parts || []).map((p) => toJson(ChatMessagePartSchema, p)),
              );
              conn.reducers.addChatMessage({
                contentId: crypto.randomUUID(),
                id: msg.id,
                modelId: (msg as any).modelId || "",
                nodeId: nodeId,
                parentId: history[history.length - 1]?.id ?? "",
                partsJson,
                role: msg.role,
                timestamp: msg.timestamp,
                treeId: history[0]?.treeId ?? "",
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
    [
      nodeId,
      setStatus,
      handleStreamChunk,
      getHistory,
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
      setStatus(ChatStatus.SUBMITTED);

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

      const chatExtension = (node?.data as DynamicNodeData)?.extension;
      const currentTreeId =
        chatExtension?.case === "chat" ? chatExtension.value.treeId : "";

      useFlowStore.getState().updateNodeData(nodeId, {
        extension: {
          case: "chat",
          value: {
            conversationHeadId: userMsgId,
            isHistoryCleared: false,
            treeId: currentTreeId || uuidv4(),
          },
        },
      });

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

      if (useFlowStore.getState().spacetimeConn) {
        useFlowStore.getState().spacetimeConn.reducers.executeAction({
          actionId: ActionId.CHAT_GENERATE,
          id: crypto.randomUUID(),
          nodeId: nodeId,
          paramsJson: JSON.stringify({
            endpointId: selectedEndpoint,
            modelId: selectedModel,
            userContent: content.trim(),
            useWebSearch: useWebSearch,
          }),
        });
        return;
      }

      try {
        sendNodeSignal(
          create(NodeSignalSchema, {
            nodeId,
            payload: {
              case: "chatGenerate",
              value: {
                endpointId: selectedEndpoint,
                modelId: selectedModel,
                userContent: content.trim(),
                useWebSearch: useWebSearch,
              } as any,
            },
          }),
        );
      } catch (err) {
        console.error("Send failed:", err);
        setStatus(ChatStatus.READY);
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
      node,
    ],
  );

  const editMessage = useCallback(
    (messageId: string, newParts: ChatMessagePart[]) => {
      const conn = useFlowStore.getState().spacetimeConn;
      if (conn) {
        conn.reducers.executeAction({
          actionId: ActionId.CHAT_EDIT,
          id: crypto.randomUUID(),
          nodeId: nodeId,
          paramsJson: JSON.stringify({
            messageId,
            newParts: newParts.map((p) => toJson(ChatMessagePartSchema, p)),
          }),
        });
        return;
      }
      sendNodeSignal(
        create(NodeSignalSchema, {
          nodeId,
          payload: {
            case: "chatEdit",
            value: {
              messageId,
              newParts,
            } as any,
          },
        }),
      );
    },
    [nodeId, sendNodeSignal],
  );

  const switchBranch = useCallback(
    (messageId: string) => {
      const chatExtension = (node?.data as DynamicNodeData)?.extension;
      if (chatExtension?.case === "chat") {
        useFlowStore.getState().updateNodeData(nodeId, {
          extension: {
            case: "chat",
            value: {
              ...chatExtension.value,
              conversationHeadId: messageId,
            },
          },
        });
      }
    },
    [nodeId, node],
  );

  const continueChat = useCallback(
    async (selectedModel: string, selectedEndpoint: string) => {
      setStatus(ChatStatus.SUBMITTED);

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
              value: {
                endpointId: selectedEndpoint,
                modelId: selectedModel,
                userContent: "", 
              } as any,
            },
          }),
        );
      } catch (err) {
        console.error("Continue failed:", err);
        setStatus(ChatStatus.READY);
      }
    },
    [setStatus, localClients, performLocalInference, nodeId, sendNodeSignal],
  );

  return { continueChat, editMessage, sendMessage, switchBranch };
}