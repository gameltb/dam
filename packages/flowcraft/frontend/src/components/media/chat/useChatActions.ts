import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import {
  ChatActionParamsSchema,
  ChatEditParamsSchema,
  type ChatMessagePart,
  ChatMessagePartSchema,
  ChatSwitchBranchParamsSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { ChatNodeStateSchema } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import { TaskQueue } from "@/shared/task-protocol";
import { TaskDispatcher } from "@/shared/TaskDispatcher";
import { useFlowStore } from "@/store/flowStore";
import { ChatStatus, type DynamicNodeData } from "@/types";
import { uploadFile } from "@/utils/assetUtils";

import { type ChatMessage, type ContextNode } from "./types";
import { useLocalInference } from "./useLocalInference";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));
  const sendNodeSignal = useFlowStore((s) => s.sendNodeSignal);
  const { localClients, performLocalInference } = useLocalInference(nodeId);

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
      const userParts = [
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
                type: att.mediaType.startsWith("image") ? MediaType.MEDIA_IMAGE : MediaType.MEDIA_UNSPECIFIED,
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

      if (!node) return;
      const data = node.data as DynamicNodeData;
      const chatExtension = data.extension;
      const currentTreeId = chatExtension?.case === "chat" ? chatExtension.value.treeId : "";

      useFlowStore.getState().updateNodeData(nodeId, {
        extension: {
          case: "chat",
          value: create(ChatNodeStateSchema, {
            conversationHeadId: userMsgId,
            isHistoryCleared: false,
            treeId: currentTreeId || uuidv4(),
          }),
        },
      });

      const localClient = localClients.find((c) => c.id === selectedEndpoint);
      if (localClient) {
        await performLocalInference(
          localClient,
          selectedModel,
          getHistory,
          setStatus,
          handleStreamChunk,
          userMsgId,
          userParts,
        );
        return;
      }

      const conn = useFlowStore.getState().spacetimeConn;
      if (conn) {
        const dispatcher = new TaskDispatcher(conn);
        dispatcher.submit(
          TaskQueue.CHAT_GENERATE,
          {
            endpointId: selectedEndpoint,
            modelId: selectedModel,
            userContent: content.trim(),
            useWebSearch: useWebSearch,
          },
          nodeId,
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
        console.error("Failed to send chat generate signal:", err);
        setStatus(ChatStatus.READY);
      }
    },
    [
      node,
      nodeId,
      setStatus,
      appendUserMessage,
      localClients,
      performLocalInference,
      getHistory,
      handleStreamChunk,
      sendNodeSignal,
    ],
  );

  const continueChat = useCallback(
    (selectedModel: string, selectedEndpoint: string) => {
      setStatus(ChatStatus.SUBMITTED);

      const conn = useFlowStore.getState().spacetimeConn;
      if (conn) {
        const dispatcher = new TaskDispatcher(conn);
        dispatcher.submit(
          TaskQueue.CHAT_GENERATE,
          {
            endpointId: selectedEndpoint,
            modelId: selectedModel,
            userContent: "",
            useWebSearch: false,
          },
          nodeId,
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
                userContent: "",
                useWebSearch: false,
              }),
            },
          }),
        );
      } catch (err) {
        console.error("Failed to send chat continue signal:", err);
        setStatus(ChatStatus.READY);
      }
    },
    [nodeId, setStatus, sendNodeSignal],
  );

  const editMessage = useCallback(
    (messageId: string, parts: ChatMessagePart[] | string) => {
      const newParts =
        typeof parts === "string"
          ? [
              create(ChatMessagePartSchema, {
                part: { case: "text", value: parts.trim() },
              }),
            ]
          : parts;

      try {
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
      } catch (err) {
        console.error("Failed to send chat edit signal:", err);
      }
    },
    [nodeId, sendNodeSignal],
  );

  const switchBranch = useCallback(
    (targetMessageId: string) => {
      try {
        sendNodeSignal(
          create(NodeSignalSchema, {
            nodeId,
            payload: {
              case: "chatSwitch",
              value: create(ChatSwitchBranchParamsSchema, {
                targetMessageId,
              }),
            },
          }),
        );
      } catch (err) {
        console.error("Failed to send chat switch signal:", err);
      }
    },
    [nodeId, sendNodeSignal],
  );

  const clearHistory = useCallback(() => {
    if (!node) return;
    useFlowStore.getState().updateNodeData(nodeId, {
      extension: {
        case: "chat",
        value: create(ChatNodeStateSchema, {
          conversationHeadId: "",
          isHistoryCleared: true,
          treeId: uuidv4(),
        }),
      },
    });
  }, [node, nodeId]);

  return { clearHistory, continueChat, editMessage, sendMessage, switchBranch };
}
