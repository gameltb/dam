import { create } from "zustand";
import { persist, createJSONStorage } from "zustand/middleware";
import { Scope } from "@/types";

export enum NavigationStatus {
  IDLE = "idle",
  SWAPPING = "swapping",
  TRANSITIONING = "transitioning",
}

export interface NavigationState {
  activeScopeId: string | null;
  navigationStatus: NavigationStatus;
  scopedViewports: Record<string, { x: number; y: number; zoom: number }>;

  // Actions
  setActiveScope: (id: string | null) => void;
  setNavigationStatus: (status: NavigationStatus) => void;
  saveViewportForScope: (scopeId: string | null, viewport: { x: number; y: number; zoom: number }) => void;
  getViewportForScope: (scopeId: string | null) => { x: number; y: number; zoom: number } | null;
}

export const useNavigationStore = create<NavigationState>()(
  persist(
    (set, get) => ({
      activeScopeId: null,
      navigationStatus: NavigationStatus.IDLE,
      scopedViewports: {},

      setActiveScope: (id) => set({ activeScopeId: id }),
      setNavigationStatus: (status) => set({ navigationStatus: status }),
      saveViewportForScope: (scopeId, viewport) => set((state) => ({
        scopedViewports: {
          ...state.scopedViewports,
          [scopeId ?? Scope.ROOT]: viewport,
        },
      })),
      getViewportForScope: (scopeId) => get().scopedViewports[scopeId ?? Scope.ROOT] || null,
    }),
    {
      name: "flowcraft-navigation",
      storage: createJSONStorage(() => localStorage),
    }
  )
);
