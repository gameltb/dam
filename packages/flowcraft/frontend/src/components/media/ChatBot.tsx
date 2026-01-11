import { create, toJson } from "@bufbuild/protobuf";
import { type FileUIPart } from "ai";
import React, { useEffect, useState } from "react";
import { toast } from "react-hot-toast";

import { ChatMessagePartSchema } from "@/generated/flowcraft/v1/actions/chat_actions_pb";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { type DynamicNodeData, TaskStatus } from "@/types";

import { ChatConversationArea } from "./chat/ChatConversationArea";
import { ChatInputArea } from "./chat/ChatInputArea";
import { type ContextNode } from "./chat/types";
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

  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);

  // Controller handles state, history, and streaming buffers
  const {
    appendUserMessage,
    handleStreamChunk,
    messages, // Replaces 'history'
    sliceHistory,
    status, // Derived status from SpacetimeDB
    streamingMessage,
  } = useChatController(conversationHeadId, nodeId);

  // Error handling: find failed task for this node
  const failedTask = useTaskStore((s) =>
    Object.values(s.tasks).find(
      (t) => t.nodeId === nodeId && t.status === TaskStatus.TASK_FAILED,
    ),
  );

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

  // Pass a dummy setStatus to useChatActions because we manage it via controller now.
  const {
    continueChat,
    editMessage,
    sendMessage: rawSendMessage,
    switchBranch,
  } = useChatActions(
    nodeId,
    () => {}, // setStatus is no-op, controller derives it
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
    await rawSendMessage(content, model, endpoint, search, files, context);
  };

  const handleRegenerate = (index: number) => {
    const targetMsg = messages[index];
    if (!targetMsg) return;

    if (targetMsg.role === "assistant") {
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
    } else if (targetMsg.role === "user") {
      // If it's the last message, just continue
      if (index === messages.length - 1) {
        continueChat(selectedModel, selectedEndpoint);
        toast.success("Generating response...");
      } else {
        // If it's in the middle, slice and branch
        sliceHistory(index);
        switchBranch(targetMsg.id);
        continueChat(selectedModel, selectedEndpoint);
        toast.success("Branching and generating...");
      }
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
    attachments: string[] = [],
  ) => {
    const newParts = [
      create(ChatMessagePartSchema, {
        part: { case: "text", value: newContent },
      }),
    ];

    attachments.forEach((url) => {
      newParts.push(
        create(ChatMessagePartSchema, {
          part: {
            case: "media",
            value: {
              aspectRatio: 0,
              content: "",
              galleryUrls: [],
              type: url.includes("image")
                ? MediaType.MEDIA_IMAGE
                : MediaType.MEDIA_UNSPECIFIED,
              url,
            },
          },
        }),
      );
    });

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

  const handleStreamingEditSave = (content: string) => {
    const lastMsg = messages[messages.length - 1];
    const parentId = lastMsg?.id ?? "";

    const newMsgId = crypto.randomUUID();
    const chatExtension =
      data.extension?.case === "chat" ? data.extension.value : null;

    // Use current treeId or create new
    const treeId = chatExtension?.treeId || crypto.randomUUID();

    const newParts = [
      create(ChatMessagePartSchema, {
        part: { case: "text", value: content },
      }),
    ];

    // Sync to backend via chatSync signal or direct reducer if connected
    const conn = useFlowStore.getState().spacetimeConn;
    if (conn) {
      const partsJson = JSON.stringify(
        newParts.map((p) => toJson(ChatMessagePartSchema, p)),
      );
      conn.reducers.addChatMessage({
        id: newMsgId,
        modelId: selectedModel,
        nodeId: nodeId,
        parentId: parentId,
        partsJson,
        role: "assistant",
        timestamp: BigInt(Date.now()),
        treeId: treeId,
      });
    }

    // Move node's head to the newly created message
    updateNodeData(nodeId, {
      extension: {
        case: "chat",
        value: {
          conversationHeadId: newMsgId,
          isHistoryCleared: false,
          treeId: treeId,
        },
      },
    });

    toast.success("Branch created from partial content.");
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
        history={messages}
        isUploading={false}
        onDelete={handleDeleteBranch}
        onEdit={handleEdit}
        onRegenerate={handleRegenerate}
        onStreamingEditSave={handleStreamingEditSave}
        onSwitchBranch={handleSwitchBranch}
        status={status}
        streamingContent={partsToText(streamingMessage?.parts)}
        errorMessage={failedTask?.message}
      />

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
        selectedEndpoint={selectedEndpoint}
        selectedModel={selectedModel}
        setDroppedNodes={setDroppedNodes}
        status={status}
        useWebSearch={useWebSearch}
      />
    </div>
  );
};
