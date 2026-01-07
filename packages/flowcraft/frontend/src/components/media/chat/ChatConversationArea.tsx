import {
  Check,
  Edit2,
  MessageSquare,
  RotateCcw,
  Trash2,
  X,
} from "lucide-react";
import React, { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "../../ai-elements/conversation";
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
} from "../../ai-elements/message";
import { type ChatMessage, type ChatStatus } from "./types";

interface Props {
  history: ChatMessage[];
  isUploading: boolean;
  onDelete: (id: string) => void;
  onEdit: (id: string, newContent: string) => void;
  onRegenerate: (index: number) => void;
  status: ChatStatus;
  streamingContent: string;
}

export const ChatConversationArea: React.FC<Props> = ({
  history,
  isUploading,
  onDelete,
  onEdit,
  onRegenerate,
  status,
  streamingContent,
}) => {
  const [editingId, setEditingId] = useState<null | string>(null);
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
                      <Textarea
                        className="min-h-[100px] bg-background/50"
                        onChange={(e) => {
                          setEditContent(e.target.value);
                        }}
                        placeholder="Edit message..."
                        value={editContent}
                      />
                      <div className="flex justify-end gap-2">
                        <Button
                          className="h-7 px-2"
                          onClick={handleCancelEdit}
                          size="sm"
                          variant="ghost"
                        >
                          <X className="mr-1" size={14} /> Cancel
                        </Button>
                        <Button
                          className="h-7 px-2"
                          onClick={() => {
                            handleSaveEdit(m.id);
                          }}
                          size="sm"
                        >
                          <Check className="mr-1" size={14} /> Save & Regenerate
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
                          onClick={() => {
                            handleStartEdit(m);
                          }}
                          tooltip="Edit"
                        >
                          <Edit2 size={12} />
                        </MessageAction>
                        {m.role === "assistant" && (
                          <MessageAction
                            onClick={() => {
                              onRegenerate(idx);
                            }}
                            tooltip="Regenerate"
                          >
                            <RotateCcw size={12} />
                          </MessageAction>
                        )}
                        <MessageAction
                          className="hover:text-destructive"
                          onClick={() => {
                            onDelete(m.id);
                          }}
                          tooltip="Delete"
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
