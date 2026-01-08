import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import OpenAI from "openai";
import { useCallback } from "react";
import { toast } from "react-hot-toast";
import { v4 as uuidv4 } from "uuid";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";

import {
  ChatActionParamsSchema,
  ChatEditParamsSchema,
  type ChatMessagePart,
  ChatMessagePartSchema,
  ChatSwitchBranchParamsSchema,
  ChatSyncBranchParamsSchema,
  ChatSyncMessageSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { ClearChatHistoryRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type ChatMessage, type ChatStatus, type ContextNode } from "./types";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const sendNodeSignal = useFlowStore((s) => s.sendNodeSignal);
  const { activeLocalClientId, localClients } = useUiStore((s) => s.settings);

  const activeLocalClient = localClients.find(
    (c) => c.id === activeLocalClientId,
  );

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

      if (activeLocalClient) {
        try {
          const client = new OpenAI({
            apiKey: activeLocalClient.apiKey || "no-key",
            baseURL: activeLocalClient.baseUrl,
            dangerouslyAllowBrowser: true,
          });

          const history = getHistory();
          const openaiMessages: OpenAI.Chat.ChatCompletionMessageParam[] =
            history.map((m) => {
              const text =
                m.parts
                  ?.map((p) => {
                    if (p.part.case === "text") {
                      return p.part.value;
                    }
                    return "";
                  })
                  .join("\n") ?? "";
              return { content: text, role: m.role as "assistant" | "user" };
            });

          openaiMessages.push({ content: content.trim(), role: "user" });

          setStatus("streaming");
          const stream = await client.chat.completions.create({
            messages: openaiMessages,
            model: activeLocalClient.model,
            stream: true,
          });

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

          sendNodeSignal(
            create(NodeSignalSchema, {
              nodeId,
              payload: {
                case: "chatSync",
                value: create(ChatSyncBranchParamsSchema, {
                  anchorMessageId: history[history.length - 1]?.id ?? "",
                  newMessages: [
                    create(ChatSyncMessageSchema, {
                      id: userMsgId,
                      parts: userParts,
                      role: "user",
                      timestamp: BigInt(Date.now()),
                    }),
                    create(ChatSyncMessageSchema, {
                      id: aiMsgId,
                      modelId: activeLocalClient.model,
                      parts: [
                        create(ChatMessagePartSchema, {
                          part: { case: "text", value: fullContent },
                        }),
                      ],
                      role: "assistant",
                      timestamp: BigInt(Date.now()),
                    }),
                  ],
                  treeId: history[0]?.treeId ?? "",
                }),
              },
            }),
          );

          return;
        } catch (err) {
          console.error("Local inference failed:", err);
          toast.error("Local inference failed. Check your settings.");
          setStatus("ready");
          return;
        }
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
      nodeId,
      setStatus,
      appendUserMessage,
      handleStreamChunk,
      getHistory,
      sendNodeSignal,
      activeLocalClient,
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

  return { clearHistory, editMessage, sendMessage, switchBranch };
}
