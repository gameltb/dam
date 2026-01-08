import React, { useMemo } from "react";

import {
  Message,
  MessageContent,
  MessageResponse,
} from "../../ai-elements/message";
import {
  Reasoning,
  ReasoningContent,
  ReasoningTrigger,
} from "../../ai-elements/reasoning";
import { type ChatStatus } from "./types";
import { parseMessageContent } from "./utils";

interface ChatStreamingMessageProps {
  isUploading: boolean;
  status: ChatStatus;
  streamingContent: string;
}

export const ChatStreamingMessage: React.FC<ChatStreamingMessageProps> = ({
  isUploading,
  status,
  streamingContent,
}) => {
  const { content, reasoning } = useMemo(
    () => parseMessageContent(streamingContent),
    [streamingContent],
  );

  // If status is submitted or uploading, show placeholder
  const displayText = content || (isUploading ? "Uploading..." : "Thinking...");
  const isStreaming = status === "streaming";

  return (
    <Message from="assistant">
      <MessageContent>
        {reasoning && (
          <Reasoning className="w-full" isStreaming={isStreaming}>
            <ReasoningTrigger />
            <ReasoningContent>{reasoning}</ReasoningContent>
          </Reasoning>
        )}
        <MessageResponse className="prose prose-sm dark:prose-invert max-w-none">
          {displayText}
        </MessageResponse>
      </MessageContent>
    </Message>
  );
};
