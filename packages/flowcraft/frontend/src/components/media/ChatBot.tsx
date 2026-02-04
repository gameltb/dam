import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import React, { useEffect, useMemo, useState } from "react";
import { toast } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import { ChatSyncMessageSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { InferenceConfigDiscoveryResponseSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowSocket } from "@/hooks/integration/useFlowSocket";
import { useNodeController } from "@/hooks/nodes/useNodeController";
import { useNodeMutation } from "@/hooks/nodes/useNodeMutation";
import { useSyncedBinding } from "@/hooks/core/useSyncedBinding";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { ChatStatus as ChatStatusEnum, TaskStatus } from "@/types";
import { NodeLenses } from "@/utils/lenses";

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
  const { spacetimeConn: _spacetimeConn } = useFlowStore(
    useShallow((s) => ({
      spacetimeConn: s.spacetimeConn,
    })),
  );

  const { inferenceConfig } = useFlowSocket();
  const nodeController = useNodeController(nodeId);
  const { setChatHead } = useNodeMutation(nodeId);

  // Two-way binding for conversation head (current active leaf in the message tree)
  const [conversationHeadId, setConversationHeadId] = useSyncedBinding(
    useMemo(() => NodeLenses.chatHead(nodeId), [nodeId]),
  );

  // Read-only binding for treeId (context boundary for the conversation)
  const [treeId] = useSyncedBinding(useMemo(() => NodeLenses.chatTreeId(nodeId), [nodeId]));

  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);

  const {
    appendUserMessage,
    handleStreamChunk,
    messages,
    sliceHistory,
    status: chatStatus,
    streamingMessage,
  } = useChatController(conversationHeadId, nodeId, treeId);

  // Combine local chat status with global task execution status
  const effectiveStatus = useMemo(() => {
    return nodeController.status === "busy" ? ChatStatusEnum.SUBMITTED : chatStatus;
  }, [nodeController.status, chatStatus]);

  const failedTask = useTaskStore((s) =>
    Object.values(s.tasks).find((t) => t.nodeId === nodeId && t.status === TaskStatus.FAILED),
  );

  const displayErrorMessage = useMemo(() => {
    if (!failedTask) return undefined;
    return failedTask.message || (failedTask as any).error || "Unknown execution error";
  }, [failedTask]);

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
    if (inferenceConfig.defaultModel && selectedModel === "gpt-4o-mini") {
      setSelectedModel(inferenceConfig.defaultModel);
    }
    if (inferenceConfig.defaultEndpointId && selectedEndpoint === "openai") {
      setSelectedEndpoint(inferenceConfig.defaultEndpointId);
    }
  }, [inferenceConfig, selectedModel, selectedEndpoint]);

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

    setConversationHeadId(newHead);
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
    setChatHead(newMsgId);

    const conn = useFlowStore.getState().spacetimeConn;
    if (conn) {
      conn.pbreducers.addChatMessage({
        message: create(ChatSyncMessageSchema, {
          id: newMsgId,
          modelId: selectedModel,
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
        errorMessage={displayErrorMessage}
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
