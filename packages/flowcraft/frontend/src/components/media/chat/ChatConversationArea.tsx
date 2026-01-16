import React from "react";

import { Conversation, ConversationContent, ConversationScrollButton } from "@/components/ai-elements/conversation";

import { ChatConversationMessage } from "./ChatConversationMessage";
import { ChatStreamingMessage } from "./ChatStreamingMessage";
import { type ChatMessage, ChatStatus } from "./types";

interface Props {
  errorMessage?: string;
  history: ChatMessage[];
  isUploading: boolean;
  onDelete: (id: string) => void;
  onEdit: (id: string, newContent: string, attachmentUrls?: string[]) => void;
  onRegenerate: (index: number) => void;
  onStreamingEditSave?: (content: string) => void;
  onSwitchBranch: (messageId: string) => void;
  status: ChatStatus;
  streamingContent: string;
}

export const ChatConversationArea: React.FC<Props> = ({
  errorMessage,
  history,
  isUploading,
  onDelete,
  onEdit,
  onRegenerate,
  onStreamingEditSave,
  onSwitchBranch,
  status,
  streamingContent,
}) => {
  const lastMessage = history[history.length - 1];
  const isLastMessageUser = lastMessage?.role === "user";

  const showStreamingOrPlaceholder =
    status === ChatStatus.STREAMING ||
    status === ChatStatus.SUBMITTED ||
    !!streamingContent ||
    !!errorMessage ||
    (status === ChatStatus.READY && isLastMessageUser);

  return (
    <Conversation className="flex-1 overflow-hidden">
      <ConversationContent>
        {history.map((m, idx) => (
          <ChatConversationMessage
            index={idx}
            key={m.id}
            message={m}
            onDelete={onDelete}
            onEdit={onEdit}
            onRegenerate={onRegenerate}
            onSwitchBranch={onSwitchBranch}
          />
        ))}

        {showStreamingOrPlaceholder && (
          <ChatStreamingMessage
            errorMessage={errorMessage}
            isUploading={isUploading}
            onEditSave={onStreamingEditSave}
            onRegenerate={() => {
              if (errorMessage || isLastMessageUser) {
                onRegenerate(history.length - 1);
              }
            }}
            status={status}
            streamingContent={streamingContent}
          />
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};
