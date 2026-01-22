import { create } from "zustand";

import { type ChatMessage } from "@/generated/flowcraft/v1/core/service_pb";

export enum ChatStreamStatus {
  ERROR = "error",
  IDLE = "idle",
  STREAMING = "streaming",
  THINKING = "thinking",
}

interface ChatState {
  clearStreams: () => void;
  messages: ChatMessage[];

  setMessages: (messages: ChatMessage[]) => void;
  setStreams: (streams: ChatStreamState[]) => void;
  streams: Record<string, ChatStreamState>; // keyed by nodeId
  upsertMessage: (message: ChatMessage) => void;
}

interface ChatStreamState {
  content: string;
  nodeId: string;
  status: ChatStreamStatus;
}

export const useChatStore = create<ChatState>((set) => ({
  clearStreams: () => {
    set({ streams: {} });
  },
  messages: [],

  setMessages: (messages) => {
    set({ messages });
  },

  setStreams: (streamList) => {
    set(() => {
      const streams: Record<string, ChatStreamState> = {};
      streamList.forEach((s) => {
        streams[s.nodeId] = s;
      });
      return { streams };
    });
  },

  streams: {},

  upsertMessage: (message) => {
    set((state) => {
      const exists = state.messages.some((m) => m.id === message.id);
      if (exists) {
        return {
          messages: state.messages.map((m) => (m.id === message.id ? message : m)),
        };
      }
      return { messages: [...state.messages, message].sort((a, b) => Number(a.timestamp - b.timestamp)) };
    });
  },
}));
