import React, { useState, useCallback, useEffect, useRef, useMemo } from "react";
import { Bot, CheckIcon, GlobeIcon, Edit2, RotateCcw, MessageSquare, Trash2, ImageIcon } from "lucide-react";
import { useNodeStream } from "../../hooks/useNodeStream";
import { type DynamicNodeData } from "../../types";
import { v4 as uuidv4 } from "uuid";
import { toast } from "react-hot-toast";
import { useFlowStore } from "../../store/flowStore";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "../ai-elements/conversation";
import {
  Message,
  MessageContent,
  MessageResponse,
  MessageActions,
  MessageAction,
  MessageBranch,
  MessageBranchContent,
  MessageBranchSelector,
  MessageBranchPrevious,
  MessageBranchNext,
  MessageBranchPage,
  MessageToolbar,
  MessageAttachments,
  MessageAttachment,
} from "../ai-elements/message";
import {
  PromptInput,
  PromptInputTextarea,
  PromptInputSubmit,
  PromptInputTools,
  PromptInputBody,
  PromptInputFooter,
  type PromptInputMessage,
  PromptInputButton,
  PromptInputAttachments,
  PromptInputAttachment,
  PromptInputActionAddAttachments,
  PromptInputActionMenu,
  PromptInputActionMenuTrigger,
  PromptInputActionMenuContent,
} from "../ai-elements/prompt-input";
import {
  ModelSelector,
  ModelSelectorContent,
  ModelSelectorEmpty,
  ModelSelectorGroup,
  ModelSelectorInput,
  ModelSelectorItem,
  ModelSelectorList,
  ModelSelectorLogo,
  ModelSelectorLogoGroup,
  ModelSelectorName,
  ModelSelectorTrigger,
} from "../ai-elements/model-selector";
import { Suggestion, Suggestions } from "../ai-elements/suggestion";

interface ContextNode {
  id: string;
  label: string;
  typeId?: string;
}

interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: number;
  attachments?: FileUIPart[];
  contextNodes?: ContextNode[];
}

interface ChatBotProps {
  nodeId: string;
}

import type { FileUIPart } from "ai";

const MODELS = [
  { id: "gpt-4o", name: "GPT-4o", chefSlug: "openai", providers: ["openai"] },
  { id: "gpt-4o-mini", name: "GPT-4o Mini", chefSlug: "openai", providers: ["openai"] },
  { id: "claude-3-5-sonnet", name: "Claude 3.5 Sonnet", chefSlug: "anthropic", providers: ["anthropic"] },
  { id: "gemini-1.5-pro", name: "Gemini 1.5 Pro", chefSlug: "google", providers: ["google"] },
];

const SUGGESTIONS = [
  "Explain how this graph works",
  "Summarize current results",
  "Optimize this workflow",
];

const getHistory = (data: DynamicNodeData): ChatMessage[] => {
  const meta = (data.metadata || {}) as Record<string, string>;
  if (meta.chat_history) {
    try {
      return JSON.parse(meta.chat_history);
    } catch {
      return [];
    }
  }
  return [];
};

export const ChatBot: React.FC<ChatBotProps> = ({ nodeId }) => {
  const node = useFlowStore(s => s.nodes.find(n => n.id === nodeId));
  const updateNodeData = useFlowStore(s => s.updateNodeData);
  
  const data = node?.data as DynamicNodeData;
  const messages = useMemo(() => (data ? getHistory(data) : []), [data]);

  const [status, setStatus] = useState<"ready" | "streaming" | "submitted" | "error">("ready");
  const [streamingContent, setStreamingContent] = useState("");
  const [inputText, setInputText] = useState("");
  const [selectedModel, setSelectedModel] = useState(MODELS[0]?.id || "");
  const [modelSelectorOpen, setModelSelectorOpen] = useState(false);
  const [useWebSearch, setUseWebSearch] = useState(false);
  const [droppedNodes, setDroppedNodes] = useState<ContextNode[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  const uploadFile = async (file: File): Promise<string | null> => {
    const formData = new FormData();
    formData.append("file", file);
    try {
      const response = await fetch("/api/upload", {
        method: "POST",
        body: formData,
      });
      const asset = await response.json();
      return asset.url;
    } catch (err) {
      console.error("Upload failed:", err);
      return null;
    }
  };

  useNodeStream(
    nodeId,
    useCallback((chunk, isDone) => {
      if (isDone) {
        setStatus("ready");
        setStreamingContent((finalContent) => {
          if (finalContent) {
            const newAssistantMsg: ChatMessage = {
              id: uuidv4(),
              role: "assistant",
              content: finalContent,
              createdAt: Date.now(),
            };
            const updated = [...getHistory(data), newAssistantMsg];
            updateNodeData(nodeId, {
              metadata: { ...(data.metadata || {}), chat_history: JSON.stringify(updated) },
            });
          }
          return "";
        });
      } else {
        setStatus("streaming");
        setStreamingContent((prev) => prev + chunk);
      }
    }, [nodeId, updateNodeData, data]),
  );

  const sendMessage = useCallback(async (content: string, files: FileUIPart[] = [], contextNodes: ContextNode[] = []) => {
    if (!content.trim() && files.length === 0 && contextNodes.length === 0 || status === "streaming") return;

    setStatus("submitted");
    setIsUploading(true);

    const finalAttachments: FileUIPart[] = [];
    for (const file of files) {
      if (file.url.startsWith("blob:")) {
        try {
          const response = await fetch(file.url);
          const blob = await response.blob();
          const permanentUrl = await uploadFile(new File([blob], file.filename || "image.png", { type: file.mediaType }));
          if (permanentUrl) {
            finalAttachments.push({ ...file, url: permanentUrl });
          }
        } catch (e) {
          console.error("Failed to process attachment", e);
        }
      } else {
        finalAttachments.push(file);
      }
    }

    const userMsg: ChatMessage = {
      id: uuidv4(),
      role: "user",
      content: content.trim(),
      createdAt: Date.now(),
      attachments: finalAttachments,
      contextNodes
    };

    const currentHistory = getHistory(data);
    const newMessages = [...currentHistory, userMsg];
    
    updateNodeData(nodeId, {
      metadata: { ...(data.metadata || {}), chat_history: JSON.stringify(newMessages) },
    });

    setIsUploading(false);
    setInputText("");
    setDroppedNodes([]);

    try {
      const { ActionExecutionRequestSchema } = await import("../../generated/flowcraft/v1/core/action_pb");
      const { create } = await import("@bufbuild/protobuf");
      const { socketClient } = await import("../../utils/SocketClient");

      await socketClient.send({
        payload: {
          case: "actionExecute",
          value: create(ActionExecutionRequestSchema, {
            actionId: "chat:generate",
            sourceNodeId: nodeId,
            contextNodeIds: contextNodes.map(n => n.id),
            paramsJson: JSON.stringify({
              model: selectedModel,
              webSearch: useWebSearch,
              messages: newMessages.map((m) => ({
                role: m.role,
                content: m.content,
                contextNodes: m.contextNodes,
                attachments: m.attachments
              })),
              stream: true,
            }),
          }),
        },
      });
    } catch (err) {
      console.error("Failed to send message", err);
      setStatus("ready");
      toast.error("Failed to send message");
    }
  }, [data, nodeId, selectedModel, status, useWebSearch, updateNodeData]);

  const handlePromptSubmit = (msg: PromptInputMessage) => {
    sendMessage(msg.text, msg.files, droppedNodes);
  };

  const handleDrop = (e: React.DragEvent) => {
    const dt = e.dataTransfer.getData("application/flowcraft-node");
    if (dt) {
      try {
        const n = JSON.parse(dt) as ContextNode;
        if (!droppedNodes.find(item => item.id === n.id)) {
          setDroppedNodes(prev => [...prev, n]);
        }
        e.preventDefault();
      } catch (err) {
        console.error("Failed to parse dropped node", err);
      }
    }
  };

  const handleRegenerate = (index: number) => {
    const historyUpTo = messages.slice(0, index);
    updateNodeData(nodeId, {
      metadata: { ...(data.metadata || {}), chat_history: JSON.stringify(historyUpTo) }
    });
    toast.success("Rolled back to this point.");
  };

  const handleDeleteMessage = (id: string) => {
    const updated = messages.filter(m => m.id !== id);
    updateNodeData(nodeId, {
      metadata: { ...(data.metadata || {}), chat_history: JSON.stringify(updated) }
    });
  };

  if (!node) return null;

  return (
    <div 
      className="flex flex-col h-full w-full overflow-hidden"
      onDragOver={(e) => e.preventDefault()}
      onDrop={handleDrop}
    >
      <Conversation className="flex-1 overflow-hidden">
        <ConversationContent>
          {messages.map((m, idx) => (
            <MessageBranch key={m.id}>
              <MessageBranchContent>
                <Message from={m.role}>
                  <MessageContent>
                    {m.contextNodes && m.contextNodes.length > 0 && (
                      <div className="flex flex-wrap gap-1 mb-2">
                        {m.contextNodes.map(cn => (
                          <div key={cn.id} className="bg-primary/10 text-[10px] px-1.5 py-0.5 rounded border border-primary/20 flex items-center gap-1">
                            <MessageSquare size={10} /> {cn.label}
                          </div>
                        ))}
                      </div>
                    )}
                    
                    {m.attachments && m.attachments.length > 0 && (
                      <MessageAttachments className="mb-2">
                        {m.attachments.map((file, i) => (
                          <MessageAttachment key={i} data={file} />
                        ))}
                      </MessageAttachments>
                    )}

                    <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">
                      {m.content}
                    </MessageResponse>
                    
                    <MessageToolbar>
                      <MessageActions>
                        <MessageAction tooltip="Edit" onClick={() => toast.info("Edit mode coming soon")}>
                          <Edit2 size={12} />
                        </MessageAction>
                        {m.role === "assistant" && (
                          <MessageAction tooltip="Regenerate" onClick={() => handleRegenerate(idx)}>
                            <RotateCcw size={12} />
                          </MessageAction>
                        )}
                        <MessageAction tooltip="Delete" onClick={() => handleDeleteMessage(m.id)} className="hover:text-destructive">
                          <Trash2 size={12} />
                        </MessageAction>
                      </MessageActions>
                      
                      <MessageBranchSelector from={m.role}>
                        <MessageBranchPrevious />
                        <MessageBranchPage />
                        <MessageBranchNext />
                      </MessageBranchSelector>
                    </MessageToolbar>
                  </MessageContent>
                </Message>
              </MessageBranchContent>
            </MessageBranch>
          ))}

          {(status === "streaming" || status === "submitted") && (
            <Message from="assistant">
              <MessageContent>
                <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">
                  {streamingContent || (isUploading ? "Uploading assets..." : "Thinking...")}
                </MessageResponse>
              </MessageContent>
            </Message>
          )}
        </ConversationContent>
        <ConversationScrollButton />
      </Conversation>

      <div className="shrink-0 bg-muted/5 border-t border-node-border">
        <div className="grid gap-2 pt-2">
          {/* Node Context Tags */}
          {droppedNodes.length > 0 && (
            <div className="flex flex-wrap gap-2 px-2">
              {droppedNodes.map(n => (
                <div key={n.id} className="group flex items-center gap-1.5 bg-primary/10 text-primary px-2 py-1 rounded-md text-xs border border-primary/20 animate-in fade-in zoom-in duration-200">
                  <MessageSquare size={12} />
                  <span>{n.label}</span>
                  <button onClick={() => setDroppedNodes(prev => prev.filter(item => item.id !== n.id))} className="hover:text-destructive transition-colors">
                    <Trash2 size={10} />
                  </button>
                </div>
              ))}
            </div>
          )}

          <Suggestions className="px-2">
            {SUGGESTIONS.map((s) => (
              <Suggestion key={s} onClick={() => sendMessage(s)} suggestion={s} />
            ))}
          </Suggestions>

          <div className="px-2 pb-2">
            <PromptInput onSubmit={handlePromptSubmit}>
              <PromptInputAttachments>
                {(file) => <PromptInputAttachment key={file.id} data={file} />}
              </PromptInputAttachments>
              <PromptInputBody>
                <PromptInputTextarea
                  placeholder={droppedNodes.length > 0 ? "Ask about these nodes..." : "Ask anything... (Drop nodes or images here)"}
                  onChange={(e) => setInputText(e.target.value)}
                  value={inputText}
                  className="min-h-[44px] max-h-[300px]"
                />
              </PromptInputBody>
              <PromptInputFooter>
                <PromptInputTools>
                  <PromptInputActionMenu>
                    <PromptInputActionMenuTrigger />
                    <PromptInputActionMenuContent>
                      <PromptInputActionAddAttachments label="Upload Images" />
                    </PromptInputActionMenuContent>
                  </PromptInputActionMenu>
                  
                  <PromptInputButton onClick={() => setUseWebSearch(!useWebSearch)} variant={useWebSearch ? "default" : "ghost"}>
                    <GlobeIcon size={14} />
                  </PromptInputButton>
                  
                  <ModelSelector open={modelSelectorOpen} onOpenChange={setModelSelectorOpen}>
                    <ModelSelectorTrigger asChild>
                      <PromptInputButton>
                        {MODELS.find(m => m.id === selectedModel)?.chefSlug && (
                          <ModelSelectorLogo provider={MODELS.find(m => m.id === selectedModel)!.chefSlug} />
                        )}
                        <span>{MODELS.find(m => m.id === selectedModel)?.name}</span>
                      </PromptInputButton>
                    </ModelSelectorTrigger>
                    <ModelSelectorContent>
                      <ModelSelectorInput placeholder="Search models..." />
                      <ModelSelectorList>
                        <ModelSelectorEmpty>No models found.</ModelSelectorEmpty>
                        <ModelSelectorGroup heading="Available Models">
                          {MODELS.map((m) => (
                            <ModelSelectorItem key={m.id} value={m.id} onSelect={() => { setSelectedModel(m.id); setModelSelectorOpen(false); }}>
                              <ModelSelectorLogo provider={m.chefSlug} />
                              <ModelSelectorName>{m.name}</ModelSelectorName>
                              <ModelSelectorLogoGroup>
                                {m.providers.map((p) => <ModelSelectorLogo key={p} provider={p} />)}
                              </ModelSelectorLogoGroup>
                              {selectedModel === m.id && <CheckIcon className="ml-auto size-3" />}
                            </ModelSelectorItem>
                          ))}
                        </ModelSelectorGroup>
                      </ModelSelectorList>
                    </ModelSelectorContent>
                  </ModelSelector>
                </PromptInputTools>
                <PromptInputSubmit
                  status={status === "streaming" ? "streaming" : status === "submitted" ? "submitted" : "ready"}
                  disabled={!inputText.trim() && droppedNodes.length === 0 && status !== "streaming"}
                />
              </PromptInputFooter>
            </PromptInput>
          </div>
        </div>
      </div>
    </div>
  );
};