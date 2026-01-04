import { create } from 'zustand';
import { nanoid } from 'nanoid';

export type MessageVersion = {
  id: string;
  content: string;
  timestamp: number;
};

export type ChatMessage = {
  key: string;
  from: 'user' | 'assistant';
  versions: MessageVersion[];
  activeVersionIndex: number;
  reasoning?: {
    content: string;
    duration: number;
  };
};

interface ChatState {
  // nodeId -> messages
  conversations: Record<string, ChatMessage[]>;
  
  // Actions
  addMessage: (nodeId: string, from: 'user' | 'assistant', content: string) => void;
  updateMessageVersion: (nodeId: string, messageKey: string, versionId: string, content: string) => void;
  switchBranch: (nodeId: string, messageKey: string, index: number) => void;
  regenerate: (nodeId: string, messageKey: string) => void;
  editMessage: (nodeId: string, messageKey: string, content: string) => void;
  
  // Internal
  setMessages: (nodeId: string, messages: ChatMessage[]) => void;
}

export const useChatStore = create<ChatState>((set, get) => ({
  conversations: {},

  setMessages: (nodeId, messages) => {
    set((state) => ({
      conversations: { ...state.conversations, [nodeId]: messages }
    }));
  },

  addMessage: (nodeId, from, content) => {
    const newVersion: MessageVersion = {
      id: nanoid(),
      content,
      timestamp: Date.now(),
    };
    
    const newMessage: ChatMessage = {
      key: nanoid(),
      from,
      versions: [newVersion],
      activeVersionIndex: 0,
    };

    set((state) => {
      const existing = state.conversations[nodeId] || [];
      return {
        conversations: {
          ...state.conversations,
          [nodeId]: [...existing, newMessage]
        }
      };
    });
  },

  updateMessageVersion: (nodeId, messageKey, versionId, content) => {
    set((state) => {
      const messages = state.conversations[nodeId] || [];
      const updated = messages.map(msg => {
        if (msg.key === messageKey) {
          return {
            ...msg,
            versions: msg.versions.map(v => v.id === versionId ? { ...v, content } : v)
          };
        }
        return msg;
      });
      return {
        conversations: { ...state.conversations, [nodeId]: updated }
      };
    });
  },

  switchBranch: (nodeId, messageKey, index) => {
    set((state) => {
      const messages = state.conversations[nodeId] || [];
      const updated = messages.map(msg => {
        if (msg.key === messageKey) {
          return { ...msg, activeVersionIndex: index };
        }
        return msg;
      });
      return {
        conversations: { ...state.conversations, [nodeId]: updated }
      };
    });
  },

  editMessage: (nodeId, messageKey, content) => {
    set((state) => {
      const messages = state.conversations[nodeId] || [];
      const msgIndex = messages.findIndex(m => m.key === messageKey);
      if (msgIndex === -1) return state;

      // When editing, we create a new version and truncate subsequent messages
      // (Common pattern in LLM chats)
      const existingMsg = messages[msgIndex];
      const newVersion: MessageVersion = {
        id: nanoid(),
        content,
        timestamp: Date.now(),
      };

      const updatedMsg = {
        ...existingMsg,
        versions: [...existingMsg.versions, newVersion],
        activeVersionIndex: existingMsg.versions.length,
      };

      const newMessages = [...messages.slice(0, msgIndex), updatedMsg];
      
      return {
        conversations: { ...state.conversations, [nodeId]: newMessages }
      };
    });
  },

  regenerate: (nodeId, messageKey) => {
    // Similar to edit but usually for assistant
    // Implementation would involve calling the backend again
  }
}));
