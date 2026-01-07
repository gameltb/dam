import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { PortMainType } from "../generated/flowcraft/v1/core/base_pb";
import { type AppNode, type Edge } from "../types";

export type ChatViewMode = "fullscreen" | "inline" | "sidebar";

export type DragMode = "pan" | "select";

export interface ShortcutConfig {
  autoLayout: string;
  copy: string;
  delete: string;
  duplicate: string;
  paste: string;
  redo: string;
  undo: string;
}

export interface UserSettings {
  hotkeys: ShortcutConfig;
  serverAddress: string;
  showControls: boolean;
  showMinimap: boolean;
  theme: "dark" | "light";
}

interface UIState {
  activeChatNodeId: null | string;
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
  setActiveChat: (nodeId: null | string, mode?: ChatViewMode) => void;
  setChatFullscreen: (fullscreen: boolean) => void;

  setClipboard: (content: null | { edges: Edge[]; nodes: AppNode[] }) => void;
  setConnectionStartHandle: (handle: UIState["connectionStartHandle"]) => void;

  setDragMode: (mode: DragMode) => void;
  // Actions for persistence
  setSettings: (settings: Partial<UserSettings>) => void;
  // Actions for transient state
  setSettingsOpen: (open: boolean) => void;
  setShortcut: (key: keyof ShortcutConfig, value: string) => void;
  setSidebarOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  // Persisted settings
  settings: UserSettings;

  // Compatibility helpers for existing components
  shortcuts: ShortcutConfig;
  sidebarWidth: number;
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

const DEFAULT_SETTINGS: UserSettings = {
  hotkeys: DEFAULT_SHORTCUTS,
  serverAddress: "/", // Default to relative path (proxied)
  showControls: true,
  showMinimap: true,
  theme: "dark",
};

export const useUiStore = create<UIState>()(
  persist(
    (set, get) => ({
      activeChatNodeId: null,
      chatViewMode: "inline",
      clipboard: null,
      connectionStartHandle: null,
      dragMode: "select",
      isChatFullscreen: false,
      isSettingsOpen: false,
      isSidebarOpen: false,
      setActiveChat: (nodeId, mode = "sidebar") =>
        set({
          activeChatNodeId: nodeId,
          chatViewMode: nodeId ? mode : "inline",
          isChatFullscreen: nodeId ? mode === "fullscreen" : false,
          isSidebarOpen: nodeId ? mode === "sidebar" : false,
        }),
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
    }),
    {
      name: "flowcraft-ui-storage",
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.shortcuts = state.settings.hotkeys;
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
