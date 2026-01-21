import type { FileUIPart } from "ai";

import type { ChatMessagePart } from "@/generated/flowcraft/v1/actions/chat_actions_pb";

import { ChatStatus } from "@/types";

export enum ChatRole {
  ASSISTANT = "assistant",
  SYSTEM = "system",
  USER = "user",
}

export interface ChatMessage {
  attachments?: FileUIPart[];
  content?: string;
  contextNodes?: ContextNode[];
  createdAt?: number;
  id: string;
  metadata?: Record<string, unknown>;
  parentId?: string;
  parts?: ChatMessagePart[];
  role: ChatRole;
  siblingIds?: string[];
  timestamp?: bigint;
  treeId?: string;
}

export interface ContextNode {
  id: string;
  label: string;
  typeId?: string;
}

export { ChatStatus };
