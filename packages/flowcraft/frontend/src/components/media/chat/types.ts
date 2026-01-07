import type { FileUIPart } from "ai";

export interface ChatMessage {
  attachments?: FileUIPart[];
  content: string;
  contextNodes?: ContextNode[];
  createdAt?: number;
  id: string;
  role: "assistant" | "system" | "user";
}

export type ChatStatus = "error" | "ready" | "streaming" | "submitted";

export interface ContextNode {
  id: string;
  label: string;
  typeId?: string;
}
