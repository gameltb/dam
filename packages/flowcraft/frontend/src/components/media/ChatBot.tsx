import { create } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import { AlertCircle, RefreshCw } from "lucide-react";
import React, { useEffect, useState } from "react";
import { toast } from "react-hot-toast";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useNodeStream } from "@/hooks/useNodeStream";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { type DynamicNodeData, TaskStatus } from "@/types";
import { Button } from "../ui/button";
import { ChatConversationArea } from "./chat/ChatConversationArea";
import { ChatInputArea } from "./chat/ChatInputArea";
import { type ChatStatus, type ContextNode } from "./chat/types";
import { useChatActions } from "./chat/useChatActions";
import { useChatController } from "./chat/useChatController";
import { partsToText } from "./chat/utils";

interface ChatBotProps {
  nodeId: string;
}

export const ChatBot: React.FC<ChatBotProps> = ({ nodeId }) => {
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const { inferenceConfig } = useFlowSocket();

  const data = node?.data as DynamicNodeData;
  const conversationHeadId =
    data.extension?.case === "chat"
      ? data.extension.value.conversationHeadId
      : undefined;

  const [status, setStatus] = useState<ChatStatus>("ready");
  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);

  // Controller handles state, history, and streaming buffers
  const {
    appendUserMessage,
    handleStreamChunk,
    lastRequest,
    messages, // Replaces 'history'
    setLastRequest,
    sliceHistory,
    streamingMessage,
  } = useChatController(conversationHeadId);

  // Error handling: find failed task for this node
  const failedTask = useTaskStore((s) =>
    Object.values(s.tasks).find(
      (t) => t.nodeId === nodeId && t.status === TaskStatus.TASK_FAILED,
    ),
  );

  // Handle streaming updates via Controller
  useNodeStream(nodeId, (chunk, isDone) => {
    if (isDone) {
      setStatus("ready");
    } else {
      setStatus("streaming");
      handleStreamChunk(chunk);
    }
  });

  // State lifted from ChatInputArea
  const [selectedModel, setSelectedModel] = useState("gpt-4o-mini");
  const [selectedEndpoint, setSelectedEndpoint] = useState("openai");
  const [useWebSearch, setUseWebSearch] = useState(false);

  // Sync selected model with backend default when it arrives if not already customized
  useEffect(() => {
    if (!inferenceConfig) return;

    // Use a microtask to avoid synchronous setState during render/effect phase
    queueMicrotask(() => {
      if (inferenceConfig.defaultModel && selectedModel === "gpt-4o-mini") {
        setSelectedModel(inferenceConfig.defaultModel);
      }
      if (inferenceConfig.defaultEndpointId && selectedEndpoint === "openai") {
        setSelectedEndpoint(inferenceConfig.defaultEndpointId);
      }
    });
  }, [inferenceConfig, selectedModel, selectedEndpoint]); // Only depend on inferenceConfig

  // Pass a dummy setHistory to useChatActions because we manage it via controller now.
  // We will intercept 'sendMessage' to update controller.
  const {
    editMessage,
    sendMessage: rawSendMessage,
    switchBranch,
  } = useChatActions(
    nodeId,
    setStatus,
    appendUserMessage,
    handleStreamChunk,
    () => messages,
  );

  const sendMessageWrapper = async (
    content: string,
    model: string,
    endpoint: string,
    search: boolean,
    files: FileUIPart[],
    context: ContextNode[],
  ) => {
    // rawSendMessage now handles appendUserMessage internally for local inference
    // but for consistency we let it handle the logic.
    setLastRequest({ content, endpoint, model, search });
    await rawSendMessage(content, model, endpoint, search, files, context);
  };

  const handleRetry = () => {
    if (lastRequest) {
      void sendMessageWrapper(
        lastRequest.content,
        lastRequest.model,
        lastRequest.endpoint,
        lastRequest.search,
        [], // attachments are complex to restore perfectly here, but could be added to lastRequest if needed
        droppedNodes,
      );
    }
  };

  const handleRegenerate = (index: number) => {
    const targetMsg = messages[index];
    if (targetMsg?.role !== "assistant") return;

    const userMsg = index > 0 ? messages[index - 1] : null;

    if (userMsg?.role === "user") {
      sliceHistory(index - 1);
      switchBranch(userMsg.parentId ?? "");

      // Extract text content for resending
      const text = (
        userMsg.parts?.map((p) =>
          p.part.case === "text" ? p.part.value : "",
        ) ?? []
      ).join("\n");

      void sendMessageWrapper(
        text,
        selectedModel,
        selectedEndpoint,
        useWebSearch,
        userMsg.attachments ?? [],
        userMsg.contextNodes ?? [],
      );
      toast.success("Regenerating...");
    } else {
      toast.error("Could not find original user message to regenerate.");
    }
  };

  const handleDeleteBranch = (id: string) => {
    const idx = messages.findIndex((m) => m.id === id);
    if (idx === -1) return;

    const prevMsg = idx > 0 ? messages[idx - 1] : null;
    const newHead = prevMsg ? prevMsg.id : "";

    const chatExtension =
      data.extension?.case === "chat" ? data.extension.value : null;

    updateNodeData(nodeId, {
      extension: {
        case: "chat",
        value: {
          conversationHeadId: newHead,
          isHistoryCleared: false,
          treeId: chatExtension?.treeId ?? "",
        },
      },
    });

    // Optimistic slice
    sliceHistory(idx);
  };

  const handleEdit = (
    id: string,
    newContent: string,
    _attachments: string[] = [],
  ) => {
    const newParts = [
      create(ChatMessagePartSchema, {
        part: { case: "text", value: newContent },
      }),
    ];
    // If we have attachments, we'd add them here too.
    editMessage(id, newParts);
    toast.success("Message branched. You can now generate a new response.");
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
        console.error("Drop parse error", err);
      }
    }
  };

  if (!node) return null;

  return (
    <div
      className="flex flex-col h-full w-full overflow-hidden relative"
      onDragOver={(e) => {
        e.preventDefault();
      }}
      onDrop={handleDrop}
    >
      <ChatConversationArea
        history={messages}
        isUploading={false}
        onDelete={handleDeleteBranch}
        onEdit={handleEdit}
        onRegenerate={handleRegenerate}
        onSwitchBranch={handleSwitchBranch}
        status={status}
        streamingContent={partsToText(streamingMessage?.parts)}
      />

      {failedTask && (
        <div className="absolute bottom-24 left-4 right-4 z-50 animate-in fade-in slide-in-from-bottom-2">
          <div className="bg-destructive/15 border border-destructive/20 backdrop-blur-md p-3 rounded-lg flex items-center justify-between gap-3 shadow-xl">
            <div className="flex items-center gap-2 text-destructive text-sm font-medium">
              <AlertCircle size={16} />
              <span className="truncate">
                {failedTask.message || "Generation failed"}
              </span>
            </div>
            <Button
              className="h-8 gap-1.5 shadow-sm shrink-0"
              onClick={handleRetry}
              size="sm"
              variant="destructive"
            >
              <RefreshCw size={14} />
              Retry
            </Button>
          </div>
        </div>
      )}

      <ChatInputArea
        droppedNodes={droppedNodes}
        inferenceConfig={inferenceConfig}
        onModelChange={(model, endpoint) => {
          setSelectedModel(model);
          if (endpoint) setSelectedEndpoint(endpoint);
        }}
        onSubmit={(msg, model, endpoint, search) => {
          void sendMessageWrapper(
            msg.text,
            model,
            endpoint,
            search,
            msg.files,
            droppedNodes,
          );
        }}
        onWebSearchChange={setUseWebSearch}
        selectedModel={selectedModel}
        setDroppedNodes={setDroppedNodes}
        status={status}
        useWebSearch={useWebSearch}
      />
    </div>
  );
};
