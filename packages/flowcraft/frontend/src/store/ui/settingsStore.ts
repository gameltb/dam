import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { type LocalLLMClientConfig, Theme } from "@/types";

export interface ShortcutConfig {
  autoLayout: string;
  copy: string;
  delete: string;
  duplicate: string;
  paste: string;
  redo: string;
  undo: string;
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

export interface SettingsState {
  activeLocalClientId: null | string;
  addLocalClient: (client: Omit<LocalLLMClientConfig, "id">) => void;
  hotkeys: ShortcutConfig;
  localClients: LocalLLMClientConfig[];
  removeLocalClient: (id: string) => void;
  serverAddress: string;
  setSettings: (settings: Partial<Omit<SettingsState, "hotkeys" | "localClients">>) => void;

  // Actions
  setTheme: (theme: Theme) => void;
  showControls: boolean;
  showMinimap: boolean;
  theme: Theme;
  updateHotkeys: (hotkeys: Partial<ShortcutConfig>) => void;
  updateLocalClient: (id: string, client: Partial<LocalLLMClientConfig>) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      activeLocalClientId: "default-local",
      addLocalClient: (client) => {
        const id = crypto.randomUUID();
        set((state) => ({
          localClients: [...state.localClients, { ...client, id }],
        }));
      },
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
      removeLocalClient: (id) =>
        set((state) => ({
          activeLocalClientId: state.activeLocalClientId === id ? null : state.activeLocalClientId,
          localClients: state.localClients.filter((c) => c.id !== id),
        })),
      serverAddress: "/spacetime/",
      setSettings: (settings) => set((state) => ({ ...state, ...settings })),

      setTheme: (theme) => set({ theme }),
      showControls: true,
      showMinimap: true,
      theme: Theme.DARK,
      updateHotkeys: (hotkeys) => set((state) => ({ hotkeys: { ...state.hotkeys, ...hotkeys } })),
      updateLocalClient: (id, client) =>
        set((state) => ({
          localClients: state.localClients.map((c) => (c.id === id ? { ...c, ...client } : c)),
        })),
    }),
    {
      name: "flowcraft-settings",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
