import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { type AppNode, type Edge } from "../types";

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
}

interface UIState {
  // Persisted settings
  settings: UserSettings;
  dragMode: DragMode;

  // Transient state
  isSettingsOpen: boolean;
  clipboard: { nodes: AppNode[]; edges: Edge[] } | null;
  connectionStartHandle: {
    nodeId: string;
    handleId: string;
    type: string;
    mainType: string;
    itemType: string;
  } | null;

  // Actions for persistence
  setSettings: (settings: Partial<UserSettings>) => void;
  setDragMode: (mode: DragMode) => void;

  // Actions for transient state
  setSettingsOpen: (open: boolean) => void;
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
};

export const useUiStore = create<UIState>()(
  persist(
    (set, get) => ({
      settings: DEFAULT_SETTINGS,
      shortcuts: DEFAULT_SHORTCUTS, // Keep in sync with settings.hotkeys
      dragMode: "select",
      isSettingsOpen: false,
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
        // Ensure shortcuts helper is synced after rehydration
        if (state) {
          state.shortcuts = state.settings.hotkeys;
        }
      },
    },
  ),
);
