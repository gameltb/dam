import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { type AppNode, type Edge } from "../types";
import { PortMainType } from "../generated/flowcraft/v1/core/base_pb";

export type DragMode = "pan" | "select";

export interface ShortcutConfig {
  copy: string;
  paste: string;
  delete: string;
  duplicate: string;
  undo: string;
  redo: string;
  autoLayout: string;
}

export interface UserSettings {
  hotkeys: ShortcutConfig;
  theme: "dark" | "light";
  showMinimap: boolean;
  showControls: boolean;
  serverAddress: string;
}

export type ChatViewMode = "inline" | "sidebar" | "fullscreen";

interface UIState {
  // Persisted settings
  settings: UserSettings;
  dragMode: DragMode;

  // Transient state
  isSettingsOpen: boolean;
  isSidebarOpen: boolean;
  isChatFullscreen: boolean;
  sidebarWidth: number;
  activeChatNodeId: string | null;
  chatViewMode: ChatViewMode;
  clipboard: { nodes: AppNode[]; edges: Edge[] } | null;
  connectionStartHandle: {
    nodeId: string;
    handleId: string;
    type: string;
    mainType: PortMainType;
    itemType: string;
  } | null;

  // Actions for persistence
  setSettings: (settings: Partial<UserSettings>) => void;
  setDragMode: (mode: DragMode) => void;

  // Actions for transient state
  setSettingsOpen: (open: boolean) => void;
  setSidebarOpen: (open: boolean) => void;
  setChatFullscreen: (fullscreen: boolean) => void;
  setSidebarWidth: (width: number) => void;
  setActiveChat: (nodeId: string | null, mode?: ChatViewMode) => void;
  setClipboard: (content: { nodes: AppNode[]; edges: Edge[] } | null) => void;
  setConnectionStartHandle: (handle: UIState["connectionStartHandle"]) => void;

  // Compatibility helpers for existing components
  shortcuts: ShortcutConfig;
  setShortcut: (key: keyof ShortcutConfig, value: string) => void;
}

const DEFAULT_SHORTCUTS: ShortcutConfig = {
  copy: "mod+c",
  paste: "mod+v",
  delete: "backspace",
  duplicate: "mod+d",
  undo: "mod+z",
  redo: "mod+shift+z",
  autoLayout: "mod+l",
};

const DEFAULT_SETTINGS: UserSettings = {
  hotkeys: DEFAULT_SHORTCUTS,
  theme: "dark",
  showMinimap: true,
  showControls: true,
  serverAddress: "/", // Default to relative path (proxied)
};

export const useUiStore = create<UIState>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,
      shortcuts: DEFAULT_SHORTCUTS,
      dragMode: "select",
      isSettingsOpen: false,
      isSidebarOpen: false,
      isChatFullscreen: false,
      sidebarWidth: 400,
      activeChatNodeId: null,
      chatViewMode: "inline",
      clipboard: null,
      connectionStartHandle: null,

      setSettings: (newSettings) =>
        set((state) => {
          const mergedSettings = { ...state.settings, ...newSettings };
          return {
            settings: mergedSettings,
            shortcuts: mergedSettings.hotkeys,
          };
        }),

      setShortcut: (key, value) => {
        const currentSettings = get().settings;
        const newHotkeys = { ...currentSettings.hotkeys, [key]: value };
        get().setSettings({ hotkeys: newHotkeys });
      },

      setDragMode: (mode) => set({ dragMode: mode }),
      setSettingsOpen: (open) => set({ isSettingsOpen: open }),
      setSidebarOpen: (open) => set({ isSidebarOpen: open }),
      setChatFullscreen: (fullscreen) => set({ isChatFullscreen: fullscreen }),
      setSidebarWidth: (width) => set({ sidebarWidth: width }),
      setActiveChat: (nodeId, mode = "sidebar") =>
        set({
          activeChatNodeId: nodeId,
          chatViewMode: nodeId ? mode : "inline",
          isSidebarOpen: nodeId ? mode === "sidebar" : false,
          isChatFullscreen: nodeId ? mode === "fullscreen" : false,
        }),
      setClipboard: (content) => set({ clipboard: content }),
      setConnectionStartHandle: (handle) =>
        set({ connectionStartHandle: handle }),
    }),
    {
      name: "flowcraft-ui-storage",
      storage: createJSONStorage(() => localStorage),
      partialize: (state) => ({
        settings: state.settings,
        dragMode: state.dragMode,
      }),
      onRehydrateStorage: () => (state) => {
        if (state) {
          state.shortcuts = state.settings.hotkeys;
        }
      },
    },
  ),
);
