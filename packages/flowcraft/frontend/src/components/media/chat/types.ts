import type { FileUIPart } from "ai";

import type { ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";

export interface ChatMessage {
  attachments?: FileUIPart[];
  content?: string; // Legacy field, should prefer parts
  contextNodes?: ContextNode[];
  createdAt?: number;
  id: string;
  metadata?: Record<string, unknown>;
  parentId?: string;
  parts?: ChatMessagePart[];
  role: "assistant" | "system" | "user";
  siblingIds?: string[];
  treeId?: string;
}

export type ChatStatus = "error" | "ready" | "streaming" | "submitted";

export interface ContextNode {
  id: string;
  label: string;
  typeId?: string;
}
