export enum TaskQueue {
  CHAT_EDIT = "chat.edit",
  CHAT_GENERATE = "chat.generate",
}

export interface ChatEditPayload {
  messageId: string;
  newParts: unknown[]; // JSON representation of ChatMessagePart[]
}

export interface ChatGeneratePayload {
  endpointId?: string;
  // Let's standardize on modelId
  modelId: string;
  userContent: string;
  useWebSearch: boolean;

  // Legacy fields to support existing code if needed?
  // messages?: any[];
  // system?: string;
}

export interface TaskPayloads {
  [TaskQueue.CHAT_EDIT]: ChatEditPayload;
  [TaskQueue.CHAT_GENERATE]: ChatGeneratePayload;
}
