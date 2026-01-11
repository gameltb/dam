import OpenAI from "openai";

import { type ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";

/**

 * Common interface for ChatMessage across FE/BE that contains the essential parts.

 */

interface CommonChatMessage {
  parts?: ChatMessagePart[];

  role: string;
}

/**

 * Maps a list of common chat messages to OpenAI's format.

 */

export function mapHistoryToOpenAI(
  history: CommonChatMessage[],
): OpenAI.Chat.ChatCompletionMessageParam[] {
  return history.map((m) => {
    const role = (
      ["assistant", "system", "user"].includes(m.role) ? m.role : "user"
    ) as "assistant" | "system" | "user";

    return {
      content: mapPartsToOpenAI(m.parts ?? []),

      role,
    } as OpenAI.Chat.ChatCompletionMessageParam;
  });
}

/**
 * Shared utility to convert Protobuf message parts to OpenAI content parts.
 * This ensures multi-modal logic is identical on both Frontend and Backend.
 */
export function mapPartsToOpenAI(
  parts: ChatMessagePart[],
): OpenAI.Chat.ChatCompletionContentPart[] {
  return parts.map((p) => {
    if (p.part.case === "text") {
      return { text: p.part.value, type: "text" };
    } else if (p.part.case === "media") {
      // For images, we use the URL directly.
      // OpenAI expects { type: "image_url", image_url: { url: "..." } }
      return {
        image_url: { url: p.part.value.url },
        type: "image_url",
      };
    }
    return { text: "", type: "text" };
  });
}
