import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import { useShallow } from "zustand/react/shallow";

import {
  ChatActionParamsSchema,
  ChatEditParamsSchema,
  type ChatMessagePart,
  ChatMessagePartSchema,
  ChatSwitchBranchParamsSchema,
} from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { TaskQueue } from "@/kernel/protocol";
import { useFlowStore } from "@/store/flowStore";
import { ChatStatus } from "@/types";
import { uploadFile } from "@/utils/assetUtils";

import { type ChatMessage, ChatRole, type ContextNode } from "./types";
import { useLocalInference } from "./useLocalInference";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const { allNodes, nodeDraft, spacetimeConn } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      nodeDraft: s.nodeDraft,
      spacetimeConn: s.spacetimeConn,
    })),
  );
  const node = allNodes.find((n) => n.id === nodeId);
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
        role: ChatRole.USER,
        timestamp: BigInt(Date.now()),
      };

      appendUserMessage(userMsg);

      if (!node) {
        console.warn("[useChatActions] Node not found for head update:", nodeId);
        return;
      }

      console.log("[useChatActions] Updating head to:", userMsgId);
      // Use ORM Draft for Chat head updates
      const res = nodeDraft(node);
      if (res.ok) {
        const draft = res.value;
        if (draft.data.extension?.case === "chat") {
          console.log("[useChatActions] Found chat extension, setting head...");
          draft.data.extension.value.conversationHeadId = userMsgId;
          draft.data.extension.value.isHistoryCleared = false;
        }
      }

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

      if (spacetimeConn) {
        spacetimeConn.kernel.submit(
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
      nodeDraft,
    ],
  );

  const continueChat = useCallback(
    (selectedModel: string, selectedEndpoint: string) => {
      setStatus(ChatStatus.SUBMITTED);

      if (spacetimeConn) {
        spacetimeConn.kernel.submit(
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
    const res = nodeDraft(node);
    if (res.ok) {
      const draft = res.value;
      if (draft.data.extension?.case === "chat") {
        draft.data.extension.value.conversationHeadId = "";
        draft.data.extension.value.isHistoryCleared = true;
      }
    }
  }, [node, nodeId, nodeDraft]);

  return { clearHistory, continueChat, editMessage, sendMessage, switchBranch };
}
