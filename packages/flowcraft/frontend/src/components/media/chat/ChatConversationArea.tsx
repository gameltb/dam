import React from "react";

import {
  Conversation,
  ConversationContent,
  ConversationScrollButton,
} from "../../ai-elements/conversation";
import { ChatConversationMessage } from "./ChatConversationMessage";
import { ChatStreamingMessage } from "./ChatStreamingMessage";
import { type ChatMessage, type ChatStatus } from "./types";

interface Props {
  history: ChatMessage[];
  isUploading: boolean;
  onDelete: (id: string) => void;
  onEdit: (id: string, newContent: string, attachmentUrls?: string[]) => void;
  onRegenerate: (index: number) => void;
  onSwitchBranch: (messageId: string) => void;
  status: ChatStatus;
  streamingContent: string;
}

export const ChatConversationArea: React.FC<Props> = ({
  history,
  isUploading,
  onDelete,
  onEdit,
  onRegenerate,
  onSwitchBranch,
  status,
  streamingContent,
}) => {
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

        {(status === "streaming" ||
          status === "submitted" ||
          !!streamingContent) && (
          <ChatStreamingMessage
            isUploading={isUploading}
            status={status}
            streamingContent={streamingContent}
          />
        )}
      </ConversationContent>
      <ConversationScrollButton />
    </Conversation>
  );
};
