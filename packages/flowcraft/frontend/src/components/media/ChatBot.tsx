import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import React, { useEffect, useMemo, useState } from "react";
import { toast } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { InferenceConfigDiscoveryResponseSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useNodeController } from "@/hooks/useNodeController";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { type DynamicNodeData, TaskStatus } from "@/types";
import { ChatStatus as ChatStatusEnum } from "@/types";

import { ChatConversationArea } from "./chat/ChatConversationArea";
import { ChatInputArea } from "./chat/ChatInputArea";
import { ChatRole, type ContextNode } from "./chat/types";
import { useChatActions } from "./chat/useChatActions";
import { useChatController } from "./chat/useChatController";
import { partsToText } from "./chat/utils";

interface ChatBotProps {
  nodeId: string;
}

export const ChatBot: React.FC<ChatBotProps> = ({ nodeId }) => {
  const { allNodes, nodeDraft } = useFlowStore(
    useShallow((s) => ({
      allNodes: s.allNodes,
      nodeDraft: s.nodeDraft,
    })),
  );
  const node = allNodes.find((n) => n.id === nodeId);
  const { inferenceConfig } = useFlowSocket();
  const nodeController = useNodeController(nodeId);

  const data = node?.data as DynamicNodeData | undefined;

  // Extract chat extension safely
  const chatExtension = data?.extension?.case === "chat" ? data.extension.value : undefined;

  const conversationHeadId =
    chatExtension?.conversationHeadId || ((data as any)?.chat?.conversation_head_id as string | undefined);
  const treeId = chatExtension?.treeId || ((data as any)?.chat?.tree_id as string | undefined) || nodeId;

  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);

  const {
    appendUserMessage,
    handleStreamChunk,
    messages,
    sliceHistory,
    status: chatStatus,
    streamingMessage,
  } = useChatController(conversationHeadId, nodeId, treeId);

  // Combine chat status with global node runtime status
  const effectiveStatus = useMemo(() => {
    if (nodeController.status === "busy") return ChatStatusEnum.SUBMITTED;
    return chatStatus;
  }, [nodeController.status, chatStatus]);

  const failedTask = useTaskStore((s) =>
    Object.values(s.tasks).find((t) => t.nodeId === nodeId && t.status === TaskStatus.FAILED),
  );

  const [selectedModel, setSelectedModel] = useState("gpt-4o-mini");
  const [selectedEndpoint, setSelectedEndpoint] = useState("openai");
  const [useWebSearch, setUseWebSearch] = useState(false);

  const {
    continueChat,
    editMessage,
    sendMessage: rawSendMessage,
    switchBranch,
  } = useChatActions(
    nodeId,
    () => {},
    appendUserMessage,
    handleStreamChunk,
    () => messages,
  );

  useEffect(() => {
    if (!inferenceConfig) return;
    queueMicrotask(() => {
      if (inferenceConfig.defaultModel && selectedModel === "gpt-4o-mini") {
        setSelectedModel(inferenceConfig.defaultModel);
      }
      if (inferenceConfig.defaultEndpointId && selectedEndpoint === "openai") {
        setSelectedEndpoint(inferenceConfig.defaultEndpointId);
      }
    });
  }, [inferenceConfig, selectedModel, selectedEndpoint]);

  if (!node || !data) return null;

  const sendMessageWrapper = async (
    content: string,
    model: string,
    endpoint: string,
    search: boolean,
    files: FileUIPart[],
    context: ContextNode[],
  ) => {
    try {
      await rawSendMessage(content, model, endpoint, search, files, context);
    } catch (err) {
      console.error("Failed to send message", err);
    }
  };

  const handleRegenerate = (index: number) => {
    const targetMsg = messages[index];
    if (!targetMsg) return;

    if (targetMsg.role === "assistant") {
      const userMsg = index > 0 ? messages[index - 1] : null;
      if (userMsg?.role === "user") {
        sliceHistory(index - 1);
        switchBranch(userMsg.parentId ?? "");
        const text = (userMsg.parts?.map((p) => (p.part.case === "text" ? p.part.value : "")) ?? []).join("\n");
        void sendMessageWrapper(
          text,
          selectedModel,
          selectedEndpoint,
          useWebSearch,
          userMsg.attachments ?? [],
          userMsg.contextNodes ?? [],
        );
        toast.success("Regenerating...");
      }
    } else if (targetMsg.role === "user") {
      if (index === messages.length - 1) {
        continueChat(selectedModel, selectedEndpoint);
      } else {
        sliceHistory(index);
        switchBranch(targetMsg.id);
        continueChat(selectedModel, selectedEndpoint);
      }
    }
  };

  const handleDeleteBranch = (id: string) => {
    const idx = messages.findIndex((m) => m.id === id);
    if (idx === -1) return;

    const prevMsg = idx > 0 ? messages[idx - 1] : null;
    const newHead = prevMsg ? prevMsg.id : "";

    const res = nodeDraft(node);
    if (res.ok) {
      const draft = res.value;
      if (draft.data.extension?.case === "chat") {
        draft.data.extension.value.conversationHeadId = newHead;
      }
    }

    sliceHistory(idx);
  };

  const handleEdit = (id: string, newContent: string, attachments: string[] = []) => {
    const newParts = [create(ChatMessagePartSchema, { part: { case: "text", value: newContent } })];
    attachments.forEach((url) => {
      newParts.push(
        create(ChatMessagePartSchema, {
          part: {
            case: "media",
            value: {
              aspectRatio: 0,
              content: "",
              galleryUrls: [],
              type: url.includes("image") ? MediaType.MEDIA_IMAGE : MediaType.MEDIA_UNSPECIFIED,
              url,
            },
          },
        }),
      );
    });
    editMessage(id, newParts);
  };

  const handleSwitchBranch = (targetId: string) => {
    switchBranch(targetId);
  };

  const handleDrop = (e: React.DragEvent) => {
    const dt = e.dataTransfer.getData("application/flowcraft-node");
    if (dt) {
      try {
        const n = JSON.parse(dt) as ContextNode;
        if (!droppedNodes.find((item) => item.id === n.id)) {
          setDroppedNodes((prev) => [...prev, n]);
        }
        e.preventDefault();
      } catch (err) {
        console.error(err);
      }
    }
  };

  const handleStreamingEditSave = (content: string) => {
    const newMsgId = crypto.randomUUID();
    const res = nodeDraft(node);
    if (res.ok) {
      const draft = res.value;
      if (draft.data.extension?.case === "chat") {
        draft.data.extension.value.conversationHeadId = newMsgId;
        draft.data.extension.value.isHistoryCleared = false;
      }
    }

    const conn = useFlowStore.getState().spacetimeConn;
    if (conn) {
      conn.pbreducers.addChatMessage({
        message: create(ChatSyncMessageSchema, {
          id: newMsgId,
          modelId: "gpt-4o",
          parts: [create(ChatMessagePartSchema, { part: { case: "text", value: content } })],
          role: ChatRole.USER,
          timestamp: BigInt(Date.now()),
        }),
        nodeId: nodeId,
      });
    }
    toast.success("Branch created.");
  };

  return (
    <div
      className="flex flex-col h-full w-full overflow-hidden relative"
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={handleDrop}
    >
      <ChatConversationArea
        errorMessage={failedTask?.message}
        history={messages}
        isUploading={false}
        onDelete={handleDeleteBranch}
        onEdit={handleEdit}
        onRegenerate={handleRegenerate}
        onStreamingEditSave={handleStreamingEditSave}
        onSwitchBranch={handleSwitchBranch}
        status={effectiveStatus}
        streamingContent={partsToText(streamingMessage?.parts)}
      />
      <ChatInputArea
        droppedNodes={droppedNodes}
        inferenceConfig={inferenceConfig ? create(InferenceConfigDiscoveryResponseSchema, inferenceConfig) : null}
        onModelChange={(model, endpoint) => {
          setSelectedModel(model);
          if (endpoint) setSelectedEndpoint(endpoint);
        }}
        onSubmit={(msg, model, endpoint, search) => {
          void sendMessageWrapper(msg.text, model, endpoint, search, msg.files, droppedNodes);
        }}
        onWebSearchChange={setUseWebSearch}
        selectedEndpoint={selectedEndpoint}
        selectedModel={selectedModel}
        setDroppedNodes={setDroppedNodes}
        status={effectiveStatus}
        useWebSearch={useWebSearch}
      />
    </div>
  );
};
