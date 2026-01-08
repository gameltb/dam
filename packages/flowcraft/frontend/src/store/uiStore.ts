import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import {
  type AppNode,
  ChatViewMode,
  DragMode,
  type Edge,
  type LocalLLMClientConfig,
  Theme,
} from "@/types";

export interface ShortcutConfig {
  autoLayout: string;
  copy: string;
  delete: string;
  duplicate: string;
  paste: string;
  redo: string;
  undo: string;
}

export interface UISettings {
  activeLocalClientId: null | string;
  hotkeys: ShortcutConfig;
  localClients: LocalLLMClientConfig[];
  serverAddress: string;
  showControls: boolean;
  showMinimap: boolean;
  theme: Theme;
}

interface UIState {
  activeChatNodeId: null | string;
  addLocalClient: (client: Omit<LocalLLMClientConfig, "id">) => void;
  chatViewMode: ChatViewMode;

  clipboard: null | { edges: Edge[]; nodes: AppNode[] };
  connectionStartHandle: null | {
    handleId: string;
    itemType: string;
    mainType: PortMainType;
    nodeId: string;
    type: string;
  };
  dragMode: DragMode;
  isChatFullscreen: boolean;
  // Transient state
  isSettingsOpen: boolean;
  isSidebarOpen: boolean;
  removeLocalClient: (id: string) => void;
  setActiveChat: (nodeId: null | string, mode?: ChatViewMode) => void;
  setActiveLocalClient: (id: null | string) => void;
  setChatFullscreen: (fullscreen: boolean) => void;

  setClipboard: (content: null | { edges: Edge[]; nodes: AppNode[] }) => void;
  setConnectionStartHandle: (handle: UIState["connectionStartHandle"]) => void;

  setDragMode: (mode: DragMode) => void;
  // Actions for persistence
  setSettings: (settings: Partial<UISettings>) => void;
  // Actions for transient state
  setSettingsOpen: (open: boolean) => void;
  setShortcut: (key: keyof ShortcutConfig, value: string) => void;
  setSidebarOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  // Persisted settings
  settings: UISettings;

  // Compatibility helpers for existing components
  shortcuts: ShortcutConfig;
  sidebarWidth: number;
  updateLocalClient: (
    id: string,
    client: Partial<LocalLLMClientConfig>,
  ) => void;
}

const DEFAULT_SHORTCUTS: ShortcutConfig = {
  autoLayout: "mod+l",
  copy: "mod+c",
  delete: "backspace",
  duplicate: "mod+d",
  paste: "mod+v",
  redo: "mod+shift+z",
  undo: "mod+z",
};

const DEFAULT_SETTINGS: UISettings = {
  activeLocalClientId: "default-local",
  hotkeys: DEFAULT_SHORTCUTS,
  localClients: [
    {
      apiKey: "lm-studio",
      baseUrl: "http://localhost:1234/v1",
      id: "default-local",
      model: "local-model",
      name: "Default Local",
    },
  ],
  serverAddress: "/", // Default to relative path (proxied)
  showControls: true,
  showMinimap: true,
  theme: Theme.DARK,
};

export const useUiStore = create<UIState>()(
  persist(
    (set, get) => ({
      activeChatNodeId: null,
      addLocalClient: (client) => {
        const id = crypto.randomUUID();
        set((state) => ({
          settings: {
            ...state.settings,
            localClients: [...state.settings.localClients, { ...client, id }],
          },
        }));
      },
      chatViewMode: ChatViewMode.INLINE,
      clipboard: null,
      connectionStartHandle: null,
      dragMode: DragMode.SELECT,
      isChatFullscreen: false,
      isSettingsOpen: false,
      isSidebarOpen: false,
      removeLocalClient: (id) => {
        set((state) => ({
          settings: {
            ...state.settings,
            activeLocalClientId:
              state.settings.activeLocalClientId === id
                ? null
                : state.settings.activeLocalClientId,
            localClients: state.settings.localClients.filter(
              (c) => c.id !== id,
            ),
          },
        }));
      },
      setActiveChat: (nodeId, mode = ChatViewMode.SIDEBAR) =>
        set({
          activeChatNodeId: nodeId,
          chatViewMode: nodeId ? mode : ChatViewMode.INLINE,
          isChatFullscreen: nodeId ? mode === ChatViewMode.FULLSCREEN : false,
          isSidebarOpen: nodeId ? mode === ChatViewMode.SIDEBAR : false,
        }),
      setActiveLocalClient: (id) => {
        set((state) => ({
          settings: { ...state.settings, activeLocalClientId: id },
        }));
      },
      setChatFullscreen: (fullscreen) => set({ isChatFullscreen: fullscreen }),
      setClipboard: (content) => set({ clipboard: content }),

      setConnectionStartHandle: (handle) =>
        set({ connectionStartHandle: handle }),

      setDragMode: (mode) => set({ dragMode: mode }),

      setSettings: (newSettings) =>
        set((state) => {
          const mergedSettings = { ...state.settings, ...newSettings };
          return {
            settings: mergedSettings,
            shortcuts: mergedSettings.hotkeys,
          };
        }),
      setSettingsOpen: (open) => set({ isSettingsOpen: open }),
      setShortcut: (key, value) => {
        const currentSettings = get().settings;
        const newHotkeys = { ...currentSettings.hotkeys, [key]: value };
        get().setSettings({ hotkeys: newHotkeys });
      },
      setSidebarOpen: (open) => set({ isSidebarOpen: open }),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      settings: DEFAULT_SETTINGS,
      shortcuts: DEFAULT_SHORTCUTS,
      sidebarWidth: 400,
      updateLocalClient: (id, client) => {
        set((state) => ({
          settings: {
            ...state.settings,
            localClients: state.settings.localClients.map((c) =>
              c.id === id ? { ...c, ...client } : c,
            ),
          },
        }));
      },
    }),
    {
      name: "flowcraft-ui-storage",
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.shortcuts = state.settings.hotkeys;

          // Migration for old localLLM settings
          const settings = state.settings as UISettings & {
            localLLM?: {
              apiKey: string;
              baseUrl: string;
              enabled: boolean;
              model: string;
            };
          };
          if (settings.localLLM) {
            if (
              settings.localLLM.enabled &&
              state.settings.localClients.length === 1 &&
              state.settings.localClients[0]?.id === "default-local"
            ) {
              state.settings.localClients[0] = {
                apiKey: settings.localLLM.apiKey,
                baseUrl: settings.localLLM.baseUrl,
                id: "default-local",
                model: settings.localLLM.model,
                name: "Imported Local",
              };
              state.settings.activeLocalClientId = "default-local";
            }
            delete settings.localLLM;
          }
        }
      },
      partialize: (state) => ({
        dragMode: state.dragMode,
        settings: state.settings,
      }),
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
