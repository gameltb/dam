import { Check, Edit2, ImagePlus, MessageSquare, RotateCcw, X } from "lucide-react";
import React, { useMemo, useState } from "react";

import {
  Message,
  MessageAction,
  MessageActions,
  MessageAttachment,
  MessageAttachments,
  MessageBranch,
  MessageBranchContent,
  MessageBranchNext,
  MessageBranchPage,
  MessageBranchPrevious,
  MessageBranchSelector,
  MessageContent,
  MessageResponse,
  MessageToolbar,
} from "@/components/ai-elements/message";
import { Reasoning, ReasoningContent, ReasoningTrigger } from "@/components/ai-elements/reasoning";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";

import { type ChatMessage } from "./types";
import { parseMessageContent } from "./utils";

interface ChatConversationMessageProps {
  index: number;
  message: ChatMessage;
  onDelete: (id: string) => void;
  onEdit: (id: string, newContent: string, attachmentUrls?: string[]) => void;
  onRegenerate: (index: number) => void;
  onSwitchBranch: (messageId: string) => void;
}

export const ChatConversationMessage: React.FC<ChatConversationMessageProps> = ({
  index,
  message: m,
  onDelete,
  onEdit,
  onRegenerate,
  onSwitchBranch,
}) => {
  const [editingId, setEditingId] = useState<null | string>(null);
  const [editContent, setEditContent] = useState("");
  const [editAttachments, setEditAttachments] = useState<string[]>([]);
  const [isUploading, setIsUploading] = useState(false);

  // Simple parts extraction for now
  const { content, images, reasoning } = useMemo(() => {
    let text = "";
    let reasoningText = "";
    const imagesList: string[] = [];

    m.parts?.forEach((p) => {
      if (p.part.case === "text") {
        const { content: parsedContent, reasoning: parsedReasoning } = parseMessageContent(p.part.value);

        text += parsedContent;

        if (parsedReasoning) reasoningText += parsedReasoning;
      } else if (p.part.case === "media") {
        const media = p.part.value;

        if (media.type === MediaType.MEDIA_IMAGE) imagesList.push(media.url);
      }
    });

    // Fallback to content if parts missing

    if (!m.parts && m.content) {
      const parsed = parseMessageContent(m.content);

      text = parsed.content;

      reasoningText = parsed.reasoning ?? "";
    }

    return { content: text, images: imagesList, reasoning: reasoningText };
  }, [m.parts, m.content]);

  const handleStartEdit = () => {
    setEditingId(m.id);
    const rawText = m.parts?.map((p) => (p.part.case === "text" ? p.part.value : "")).join("\n") ?? m.content ?? "";
    setEditContent(rawText);
    setEditAttachments(
      m.parts?.filter((p) => p.part.case === "media").map((p) => (p.part.case === "media" ? p.part.value.url : "")) ??
        [],
    );
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditContent("");
    setEditAttachments([]);
  };

  const handleSaveEdit = () => {
    if (editContent.trim()) {
      onEdit(m.id, editContent, editAttachments);
      setEditingId(null);
      setEditContent("");
      setEditAttachments([]);
    }
  };

  const handleAddImage = () => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = "image/*";
    input.onchange = async (e) => {
      const file = (e.target as HTMLInputElement).files?.[0];
      if (file) {
        setIsUploading(true);
        const formData = new FormData();
        formData.append("file", file);
        try {
          const response = await fetch("/api/upload", {
            body: formData,
            method: "POST",
          });
          const asset = (await response.json()) as { url: string };
          if (asset.url) {
            setEditAttachments((prev) => [...prev, asset.url]);
          }
        } catch (err) {
          console.error("Upload failed:", err);
        } finally {
          setIsUploading(false);
        }
      }
    };
    input.click();
  };

  return (
    <MessageBranch>
      <MessageBranchContent>
        <Message from={m.role}>
          <MessageContent>
            {m.contextNodes && m.contextNodes.length > 0 && (
              <div className="flex flex-wrap gap-1 mb-2">
                {m.contextNodes.map((cn) => (
                  <div
                    className="bg-primary/10 text-[10px] px-1.5 py-0.5 rounded border border-primary/20 flex items-center gap-1"
                    key={cn.id}
                  >
                    <MessageSquare size={10} /> {cn.label}
                  </div>
                ))}
              </div>
            )}

            {m.attachments && m.attachments.length > 0 && (
              <MessageAttachments className="mb-2">
                {m.attachments.map((file, i) => (
                  <MessageAttachment data={file} key={i} />
                ))}
              </MessageAttachments>
            )}

            {editingId === m.id ? (
              <div className="flex flex-col gap-2 min-w-[300px]">
                {editAttachments.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {editAttachments.map((url, i) => (
                      <div className="relative group" key={i}>
                        <img
                          alt={`Attachment ${String(i)}`}
                          className="h-16 w-16 object-cover rounded border"
                          src={url}
                        />
                        <button
                          className="absolute -top-1 -right-1 bg-destructive text-destructive-foreground rounded-full p-0.5 opacity-0 group-hover:opacity-100 transition-opacity"
                          onClick={() => {
                            setEditAttachments((prev) => prev.filter((_, idx) => idx !== i));
                          }}
                        >
                          <X size={10} />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
                <Textarea
                  className="min-h-[100px] bg-background/50"
                  onChange={(e) => {
                    setEditContent(e.target.value);
                  }}
                  placeholder="Edit message..."
                  value={editContent}
                />
                <div className="flex justify-between items-center gap-2">
                  <Button
                    className="h-7 px-2"
                    disabled={isUploading}
                    onClick={handleAddImage}
                    size="sm"
                    variant="ghost"
                  >
                    <ImagePlus className="mr-1" size={14} /> {isUploading ? "Uploading..." : "Add Image"}
                  </Button>
                  <div className="flex gap-2">
                    <Button className="h-7 px-2" onClick={handleCancelEdit} size="sm" variant="ghost">
                      <X className="mr-1" size={14} /> Cancel
                    </Button>
                    <Button className="h-7 px-2" onClick={handleSaveEdit} size="sm">
                      <Check className="mr-1" size={14} /> Branch & Save
                    </Button>
                  </div>
                </div>
              </div>
            ) : (
              <>
                {reasoning && (
                  <Reasoning className="w-full">
                    <ReasoningTrigger />
                    <ReasoningContent>{reasoning}</ReasoningContent>
                  </Reasoning>
                )}
                {images.length > 0 && (
                  <div className="flex flex-wrap gap-2 mb-2">
                    {images.map((url, i) => (
                      <img alt={`Image ${String(i)}`} className="max-h-[200px] rounded border" key={i} src={url} />
                    ))}
                  </div>
                )}
                <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">{content}</MessageResponse>
              </>
            )}

            {!editingId && (
              <MessageToolbar>
                <MessageActions>
                  <MessageAction onClick={handleStartEdit} tooltip="Edit & Branch">
                    <Edit2 size={12} />
                  </MessageAction>
                  {m.role === "assistant" && (
                    <MessageAction
                      onClick={() => {
                        onRegenerate(index);
                      }}
                      tooltip="Regenerate Branch"
                    >
                      <RotateCcw size={12} />
                    </MessageAction>
                  )}
                  <MessageAction
                    onClick={() => {
                      onDelete(m.id);
                    }}
                    tooltip="Switch to Parent Branch"
                  >
                    <MessageSquare size={12} />
                  </MessageAction>
                </MessageActions>
                {m.siblingIds && m.siblingIds.length > 0 && (
                  <MessageBranchSelector from={m.role}>
                    <MessageBranchPrevious
                      onClick={() => {
                        const siblings = m.siblingIds;
                        if (!siblings) return;
                        const currentSiblings = [m.id, ...siblings].sort();
                        const currentIndex = currentSiblings.indexOf(m.id);
                        const prevIndex = (currentIndex - 1 + currentSiblings.length) % currentSiblings.length;
                        const target = currentSiblings[prevIndex];
                        if (target) onSwitchBranch(target);
                      }}
                    />
                    <MessageBranchPage>
                      {([m.id, ...m.siblingIds].sort().indexOf(m.id) + 1).toString()} /{" "}
                      {(m.siblingIds.length + 1).toString()}
                    </MessageBranchPage>
                    <MessageBranchNext
                      onClick={() => {
                        const siblings = m.siblingIds;
                        if (!siblings) return;
                        const currentSiblings = [m.id, ...siblings].sort();
                        const currentIndex = currentSiblings.indexOf(m.id);
                        const nextIndex = (currentIndex + 1) % currentSiblings.length;
                        const target = currentSiblings[nextIndex];
                        if (target) onSwitchBranch(target);
                      }}
                    />
                  </MessageBranchSelector>
                )}
              </MessageToolbar>
            )}
          </MessageContent>
        </Message>
      </MessageBranchContent>
    </MessageBranch>
  );
};
