import { create, toJsonString } from "@bufbuild/protobuf";
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
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { NodeSignalSchema } from "@/generated/flowcraft/v1/core/signals_pb";
import { TaskQueue } from "@/kernel/protocol";
import { useFlowStore } from "@/store/flowStore";
import { commit, editNode } from "@/store/orchestrator";
import { AppNodeType, type ChatNodeData, ChatStatus, isChatNode, RenderMode } from "@/types";
import { uploadFile } from "@/utils/assetUtils";
import { log } from "@/utils/logger";

import { type ChatMessage, ChatRole, type ContextNode } from "./types";
import { useLocalInference } from "./useLocalInference";

export function useChatActions(
  nodeId: string,
  setStatus: (s: ChatStatus) => void,
  appendUserMessage: (msg: ChatMessage) => void,
  handleStreamChunk: (chunk: string) => void,
  getHistory: () => ChatMessage[],
) {
  const { nodesById } = useFlowStore(
    useShallow((s) => ({
      nodesById: s.nodesById,
    })),
  );

  const node = nodesById[nodeId];
  const sendNodeSignal = useFlowStore((s) => s.sendNodeSignal);
  const { localClients, performLocalInference } = useLocalInference(nodeId);

  const getSpacetimeConn = useCallback(() => useFlowStore.getState().spacetimeConn, []);

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
        log.warn("[useChatActions] Node not found for head update:", nodeId);
        return;
      }

      // Atomic commit: add node + establish connection + update Head
      commit(
        (draft) => {
          const chatNode = draft.nodesById[nodeId];
          if (chatNode && isChatNode(chatNode)) {
            const data = chatNode.data as ChatNodeData;
            const currentHead = data.extension.value.conversationHeadId;

            // 1. Add message node
            draft.nodesById[userMsgId] = {
              data: {
                $typeName: NodeDataSchema.typeName,
                activeMode: RenderMode.MODE_MARKDOWN,
                availableModes: [RenderMode.MODE_MARKDOWN],
                displayName: "User Message",
                extension: { case: undefined, value: undefined },
                inputPorts: [],
                metadata: {
                  parts_json: JSON.stringify([
                    JSON.parse(
                      toJsonString(
                        ChatMessagePartSchema,
                        create(ChatMessagePartSchema, {
                          part: { case: "text", value: content.trim() },
                        }),
                      ),
                    ),
                  ]),
                  role: "user",
                  timestamp: Date.now().toString(),
                },
                outputPorts: [],
                schemaVersion: 1,
                taskId: "",
                widgets: [],
              },
              height: 150,
              id: userMsgId,
              parentId: nodeId,
              position: { x: chatNode.position.x, y: chatNode.position.y + 300 },
              scopeId: chatNode.scopeId, // Inherit scope from parent
              type: AppNodeType.CHAT_MESSAGE,
              width: 300,
            };
            // 2. Establish connection (from old Head to new message)
            const edgeId = `e-${currentHead || nodeId}-${userMsgId}`;
            draft.edgesById[edgeId] = {
              id: edgeId,
              source: currentHead || nodeId,
              target: userMsgId,
            };

            // 3. Update Head
            data.extension.value.conversationHeadId = userMsgId;
            data.extension.value.isHistoryCleared = false;
          }
        },
        { description: "User sent message", isHistoryOp: true },
      );
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

      const conn = getSpacetimeConn();
      if (conn) {
        conn.kernel.submit(
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
        log.error("useChatActions/sendMessage", err);
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

      const conn = getSpacetimeConn();
      if (conn) {
        conn.kernel.submit(
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
        log.error("useChatActions/continueChat", err);
        setStatus(ChatStatus.READY);
      }
    },
    [nodeId, setStatus, sendNodeSignal, getSpacetimeConn],
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
        log.error("useChatActions/editMessage", err);
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
        log.error("useChatActions/switchBranch", err);
      }
    },
    [nodeId, sendNodeSignal],
  );

  const clearHistory = useCallback(() => {
    if (!node) return;
    editNode(nodeId, (draft) => {
      if (isChatNode(draft)) {
        const data = draft.data as ChatNodeData;
        data.extension.value.conversationHeadId = "";
        data.extension.value.isHistoryCleared = true;
      }
    });
  }, [node, nodeId]);

  return { clearHistory, continueChat, editMessage, sendMessage, switchBranch };
}
