import { Bot, Check, Edit2, Play, RotateCcw, X } from "lucide-react";
import React, { useState } from "react";

import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";

import {
  Message,
  MessageAction,
  MessageActions,
  MessageContent,
  MessageResponse,
  MessageToolbar,
} from "../../ai-elements/message";
import { type ChatStatus } from "./types";

interface Props {
  errorMessage?: string;
  isUploading: boolean;
  onEditSave?: (content: string) => void;
  onRegenerate?: () => void;
  status: ChatStatus;
  streamingContent: string;
}

export const ChatStreamingMessage: React.FC<Props> = ({
  status,
  streamingContent,
  errorMessage,
  onRegenerate,
  onEditSave,
}) => {
  const [isEditing, setIsEditing] = useState(false);
  const [editValue, setEditValue] = useState("");

  const isPendingResponse =
    status === "ready" && !streamingContent && !errorMessage;

  const handleStartEdit = () => {
    setEditValue(streamingContent);
    setIsEditing(true);
  };

  const handleSave = () => {
    if (onEditSave) {
      onEditSave(editValue);
      setIsEditing(false);
    }
  };

  if (status === "ready" && !streamingContent && !errorMessage && !isEditing)
    return null;

  return (
    <div className="animate-in fade-in slide-in-from-bottom-2 duration-300">
      <Message from="assistant">
        <MessageContent>
          {/* Status Indicator */}
          {!isEditing && (status === "submitted" || status === "streaming") && (
            <div className="flex items-center gap-2 mb-2 opacity-50">
              <Bot className="animate-pulse" size={14} />
              <span className="text-[10px] uppercase tracking-wider font-bold">
                {status === "submitted" ? "Thinking..." : "Generating..."}
              </span>
            </div>
          )}

          {isEditing ? (
            <div className="flex flex-col gap-2 min-w-[300px]">
              <Textarea
                className="min-h-[100px] bg-background/50 text-xs font-sans"
                onChange={(e) => {
                  setEditValue(e.target.value);
                }}
                value={editValue}
              />
              <div className="flex justify-end gap-2">
                <Button
                  className="h-7 px-2"
                  onClick={() => {
                    setIsEditing(false);
                  }}
                  size="sm"
                  variant="ghost"
                >
                  <X className="mr-1" size={14} /> Cancel
                </Button>
                <Button className="h-7 px-2" onClick={handleSave} size="sm">
                  <Check className="mr-1" size={14} /> Save as Message
                </Button>
              </div>
            </div>
          ) : (
            <>
              {streamingContent && (
                <MessageResponse className="prose prose-sm dark:prose-invert max-w-none mb-2">
                  {streamingContent}
                </MessageResponse>
              )}

              {errorMessage && (
                <div className="my-2 border border-destructive/30 bg-destructive/5 rounded-md overflow-hidden">
                  <div className="px-3 py-1.5 bg-destructive/10 border-b border-destructive/20">
                    <span className="text-destructive text-[10px] font-bold uppercase tracking-tight">
                      Server Execution Error
                    </span>
                  </div>
                  <div className="p-3 max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-destructive/20">
                    <pre className="text-xs text-destructive-foreground/90 font-mono whitespace-pre-wrap break-all leading-relaxed">
                      {errorMessage.replace(/\\n/g, "\n")}
                    </pre>
                  </div>
                </div>
              )}

              {isPendingResponse && (
                <p className="text-xs text-muted-foreground italic mb-2">
                  Waiting to generate a response for your message...
                </p>
              )}
            </>
          )}

          {!isEditing && (
            <MessageToolbar>
              <MessageActions>
                {(errorMessage || isPendingResponse) && onRegenerate && (
                  <MessageAction
                    className={errorMessage ? "text-destructive" : "text-primary"}
                    onClick={onRegenerate}
                    tooltip={
                      errorMessage ? "Try Regenerating" : "Generate Response"
                    }
                  >
                    {errorMessage ? (
                      <RotateCcw size={12} />
                    ) : (
                      <Play size={12} fill="currentColor" />
                    )}
                  </MessageAction>
                )}
                {/* Allow editing if we have content OR an error */}
                {(streamingContent || errorMessage) && (
                  <MessageAction onClick={handleStartEdit} tooltip="Edit & Save">
                    <Edit2 size={12} />
                  </MessageAction>
                )}
              </MessageActions>
            </MessageToolbar>
          )}
        </MessageContent>
      </Message>
    </div>
  );
};