import React, { useState } from "react";
import {
  MessageSquare,
  Edit2,
  RotateCcw,
  Trash2,
  X,
  Check,
} from "lucide-react";
import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "../../ai-elements/conversation";
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
} from "../../ai-elements/message";
import { type ChatMessage, type ChatStatus } from "./types";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

interface Props {
  history: ChatMessage[];
  streamingContent: string;
  status: ChatStatus;
  isUploading: boolean;
  onRegenerate: (index: number) => void;
  onDelete: (id: string) => void;
  onEdit: (id: string, newContent: string) => void;
}

export const ChatConversationArea: React.FC<Props> = ({
  history,
  streamingContent,
  status,
  isUploading,
  onRegenerate,
  onDelete,
  onEdit,
}) => {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [editContent, setEditContent] = useState("");

  const handleStartEdit = (msg: ChatMessage) => {
    setEditingId(msg.id);
    setEditContent(msg.content);
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setEditContent("");
  };

  const handleSaveEdit = (id: string) => {
    if (editContent.trim()) {
      onEdit(id, editContent);
      setEditingId(null);
      setEditContent("");
    }
  };

  return (
    <Conversation className="flex-1 overflow-hidden">
      <ConversationContent>
        {history.map((m, idx) => (
          <MessageBranch key={m.id}>
            <MessageBranchContent>
              <Message from={m.role}>
                <MessageContent>
                  {m.contextNodes && m.contextNodes.length > 0 && (
                    <div className="flex flex-wrap gap-1 mb-2">
                      {m.contextNodes.map((cn) => (
                        <div
                          key={cn.id}
                          className="bg-primary/10 text-[10px] px-1.5 py-0.5 rounded border border-primary/20 flex items-center gap-1"
                        >
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

                  {editingId === m.id ? (
                    <div className="flex flex-col gap-2 min-w-[300px]">
                      <Textarea
                        value={editContent}
                        onChange={(e) => {
                          setEditContent(e.target.value);
                        }}
                        className="min-h-[100px] bg-background/50"
                        placeholder="Edit message..."
                      />
                      <div className="flex justify-end gap-2">
                        <Button
                          size="sm"
                          variant="ghost"
                          onClick={handleCancelEdit}
                          className="h-7 px-2"
                        >
                          <X size={14} className="mr-1" /> Cancel
                        </Button>
                        <Button
                          size="sm"
                          onClick={() => {
                            handleSaveEdit(m.id);
                          }}
                          className="h-7 px-2"
                        >
                          <Check size={14} className="mr-1" /> Save & Regenerate
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">
                      {m.content}
                    </MessageResponse>
                  )}

                  {!editingId && (
                    <MessageToolbar>
                      <MessageActions>
                        <MessageAction
                          tooltip="Edit"
                          onClick={() => {
                            handleStartEdit(m);
                          }}
                        >
                          <Edit2 size={12} />
                        </MessageAction>
                        {m.role === "assistant" && (
                          <MessageAction
                            tooltip="Regenerate"
                            onClick={() => {
                              onRegenerate(idx);
                            }}
                          >
                            <RotateCcw size={12} />
                          </MessageAction>
                        )}
                        <MessageAction
                          tooltip="Delete"
                          onClick={() => {
                            onDelete(m.id);
                          }}
                          className="hover:text-destructive"
                        >
                          <Trash2 size={12} />
                        </MessageAction>
                      </MessageActions>
                      <MessageBranchSelector from={m.role}>
                        <MessageBranchPrevious />
                        <MessageBranchPage />
                        <MessageBranchNext />
                      </MessageBranchSelector>
                    </MessageToolbar>
                  )}
                </MessageContent>
              </Message>
            </MessageBranchContent>
          </MessageBranch>
        ))}

        {(status === "streaming" || status === "submitted") && (
          <Message from="assistant">
            <MessageContent>
              <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">
                {streamingContent ||
                  (isUploading ? "Uploading..." : "Thinking...")}
              </MessageResponse>
            </MessageContent>
          </Message>
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};
