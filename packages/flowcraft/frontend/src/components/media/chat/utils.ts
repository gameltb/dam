import type { ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";

export interface ParsedContent {
  content: string;
  reasoning?: string;
}

export function parseMessageContent(text: string): ParsedContent {
  // Check for <think> tag
  const thinkMatch = /<think>(.*?)<\/think>/s.exec(text);

  if (thinkMatch) {
    const reasoning = thinkMatch[1]?.trim() ?? "";
    const content = text.replace(thinkMatch[0], "").trim();
    return { content, reasoning };
  }

  // Handle unclosed <think> during streaming
  const openThinkMatch = /<think>(.*)/s.exec(text);
  if (openThinkMatch) {
    return {
      content: "", // Content hasn't started yet if think is unclosed
      reasoning: openThinkMatch[1]?.trim() ?? "",
    };
  }

  return { content: text };
}

export function partsToText(parts: ChatMessagePart[] | undefined): string {
  if (!parts) return "";
  return parts
    .map((p) => {
      if (p.part.case === "text") return p.part.value;
      return "";
    })
    .filter(Boolean)
    .join("\n");
}
