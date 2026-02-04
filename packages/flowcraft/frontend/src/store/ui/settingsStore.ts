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
  theme: Theme;
  serverAddress: string;
  showControls: boolean;
  showMinimap: boolean;
  hotkeys: ShortcutConfig;
  localClients: LocalLLMClientConfig[];
  activeLocalClientId: string | null;

  // Actions
  setTheme: (theme: Theme) => void;
  setSettings: (settings: Partial<Omit<SettingsState, "hotkeys" | "localClients">>) => void;
  updateHotkeys: (hotkeys: Partial<ShortcutConfig>) => void;
  addLocalClient: (client: Omit<LocalLLMClientConfig, "id">) => void;
  removeLocalClient: (id: string) => void;
  updateLocalClient: (id: string, client: Partial<LocalLLMClientConfig>) => void;
}

export const useSettingsStore = create<SettingsState>()(
  persist(
    (set) => ({
      theme: Theme.DARK,
      serverAddress: "/spacetime/",
      showControls: true,
      showMinimap: true,
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
      activeLocalClientId: "default-local",

      setTheme: (theme) => set({ theme }),
      setSettings: (settings) => set((state) => ({ ...state, ...settings })),
      updateHotkeys: (hotkeys) => set((state) => ({ hotkeys: { ...state.hotkeys, ...hotkeys } })),
      addLocalClient: (client) => {
        const id = crypto.randomUUID();
        set((state) => ({
          localClients: [...state.localClients, { ...client, id }],
        }));
      },
      removeLocalClient: (id) => set((state) => ({
        localClients: state.localClients.filter((c) => c.id !== id),
        activeLocalClientId: state.activeLocalClientId === id ? null : state.activeLocalClientId,
      })),
      updateLocalClient: (id, client) => set((state) => ({
        localClients: state.localClients.map((c) => (c.id === id ? { ...c, ...client } : c)),
      })),
    }),
    {
      name: "flowcraft-settings",
      storage: createJSONStorage(() => localStorage),
    }
  )
);
