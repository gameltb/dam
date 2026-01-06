import React, { useState, useCallback } from "react";
import { useNodeStream } from "../../hooks/useNodeStream";
import { useFlowStore } from "../../store/flowStore";
import { type DynamicNodeData } from "../../types";
import { type ContextNode, type ChatStatus } from "./chat/types";
import { useChatHistory } from "./chat/useChatHistory";
import { useChatActions } from "./chat/useChatActions";
import { ChatConversationArea } from "./chat/ChatConversationArea";
import { ChatInputArea } from "./chat/ChatInputArea";
import { toast } from "react-hot-toast";

interface ChatBotProps {
  nodeId: string;
}

export const ChatBot: React.FC<ChatBotProps> = ({ nodeId }) => {
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const data = node?.data as DynamicNodeData;
  const metadata = (data?.metadata || {}) as Record<string, unknown>;
  const conversationHeadId = metadata.conversation_head_id as
    | string
    | undefined;

  const [status, setStatus] = useState<ChatStatus>("ready");
  const [streamingContent, setStreamingContent] = useState("");
  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // State lifted from ChatInputArea
  const [selectedModel, setSelectedModel] = useState("gpt-4o");
  const [useWebSearch, setUseWebSearch] = useState(false);

  // 1. 业务逻辑钩子
  const { history, setHistory } = useChatHistory(conversationHeadId);
  const { sendMessage } = useChatActions(nodeId, setStatus, setHistory);

  // 2. 流式处理
  useNodeStream(
    nodeId,
    useCallback(
      (chunk, isDone) => {
        if (isDone) {
          setStatus("ready");
          setStreamingContent("");
          setIsUploading(false);
        } else {
          setStatus("streaming");
          setStreamingContent((prev) => prev + chunk);
        }
      },
      [nodeId],
    ),
  );

  const handleRegenerate = (index: number) => {
    const targetMsg = history[index];
    if (targetMsg) {
      updateNodeData(nodeId, {
        metadata: { ...metadata, conversation_head_id: targetMsg.id },
      });
      toast.success("Rolled back to this point.");
    }
  };

  const handleDeleteMessage = (id: string) => {
    const idx = history.findIndex((m) => m.id === id);
    if (idx === -1) return;

    const prevMsg = idx > 0 ? history[idx - 1] : null;
    const newHead = prevMsg ? prevMsg.id : "";

    updateNodeData(nodeId, {
      metadata: { ...metadata, conversation_head_id: newHead },
    });
    // Optimistically update history
    setHistory(history.slice(0, idx));
  };

  const handleEdit = (id: string, newContent: string) => {
    const idx = history.findIndex((m) => m.id === id);
    if (idx === -1) return;

    const prevMsg = idx > 0 ? history[idx - 1] : null;
    const newHead = prevMsg ? prevMsg.id : "";

    // Rollback to parent
    updateNodeData(nodeId, {
      metadata: { ...metadata, conversation_head_id: newHead },
    });

    // Optimistically truncate history
    const slicedHistory = history.slice(0, idx);
    setHistory(slicedHistory);

    // Send new message
    void sendMessage(
      newContent,
      selectedModel,
      useWebSearch,
      [], // Attachments are lost on text edit for now, or we could try to preserve them from original message
      droppedNodes, // Context nodes preserved? Or should we use original message's context?
      // Ideally original message's context, but for simplicity using current droppedNodes or empty.
      // Let's use droppedNodes for now as user might have dropped new context.
    );
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
        console.error("Drop parse error", err);
      }
    }
  };

  if (!node) return null;

  return (
    <div
      className="flex flex-col h-full w-full overflow-hidden"
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={handleDrop}
    >
      <ChatConversationArea
        history={history}
        streamingContent={streamingContent}
        status={status}
        isUploading={isUploading}
        onRegenerate={handleRegenerate}
        onDelete={handleDeleteMessage}
        onEdit={handleEdit}
      />
      <ChatInputArea
        status={status}
        droppedNodes={droppedNodes}
        setDroppedNodes={setDroppedNodes}
        selectedModel={selectedModel}
        onModelChange={setSelectedModel}
        useWebSearch={useWebSearch}
        onWebSearchChange={setUseWebSearch}
        onSubmit={(msg, model, search) => {
          void sendMessage(msg.text, model, search, msg.files, droppedNodes);
        }}
      />
    </div>
  );
};
