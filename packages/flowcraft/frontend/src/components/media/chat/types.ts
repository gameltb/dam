import type { FileUIPart } from "ai";

export interface ContextNode {
  id: string;
  label: string;
  typeId?: string;
}

export interface ChatMessage {
  id: string;
  role: "user" | "assistant" | "system";
  content: string;
  createdAt?: number;
  attachments?: FileUIPart[];
  contextNodes?: ContextNode[];
}

export type ChatStatus = "ready" | "streaming" | "submitted" | "error";
