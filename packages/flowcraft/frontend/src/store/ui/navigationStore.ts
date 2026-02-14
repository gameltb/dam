import { create } from "zustand";
import { createJSONStorage, persist } from "zustand/middleware";

import { Scope } from "@/types";

export enum NavigationStatus {
  IDLE = "idle",
  SWAPPING = "swapping",
  TRANSITIONING = "transitioning",
}

export interface NavigationState {
  activeScopeId: null | string;
  getViewportForScope: (scopeId: null | string) => null | { x: number; y: number; zoom: number };
  navigationStatus: NavigationStatus;

  saveViewportForScope: (scopeId: null | string, viewport: { x: number; y: number; zoom: number }) => void;
  scopedViewports: Record<string, { x: number; y: number; zoom: number }>;
  // Actions
  setActiveScope: (id: null | string) => void;
  setNavigationStatus: (status: NavigationStatus) => void;
}

export const useNavigationStore = create<NavigationState>()(
  persist(
    (set, get) => ({
      activeScopeId: null,
      getViewportForScope: (scopeId) => get().scopedViewports[scopeId ?? Scope.ROOT] || null,
      navigationStatus: NavigationStatus.IDLE,

      saveViewportForScope: (scopeId, viewport) =>
        set((state) => ({
          scopedViewports: {
            ...state.scopedViewports,
            [scopeId ?? Scope.ROOT]: viewport,
          },
        })),
      scopedViewports: {},
      setActiveScope: (id) => set({ activeScopeId: id }),
      setNavigationStatus: (status) => set({ navigationStatus: status }),
    }),
    {
      name: "flowcraft-navigation",
      storage: createJSONStorage(() => localStorage),
    },
  ),
);
