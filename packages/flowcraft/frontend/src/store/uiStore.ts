import { create } from "zustand";
import { type AppNode } from "../types";
import { type Edge } from "@xyflow/react";

export interface UiState {
  clipboard: { nodes: AppNode[]; edges: Edge[] } | null;
  setClipboard: (data: { nodes: AppNode[]; edges: Edge[] } | null) => void;

  connectionStartHandle: {
    nodeId: string;
    handleId: string;
    type: "source" | "target";
    portType?: string;
    mainType?: string;
    itemType?: string;
  } | null;
  setConnectionStartHandle: (
    h: {
      nodeId: string;
      handleId: string;
      type: "source" | "target";
      portType?: string;
      mainType?: string;
      itemType?: string;
    } | null,
  ) => void;
}

export const useUiStore = create<UiState>((set) => ({
  clipboard: null,
  setClipboard: (c) => {
    set({ clipboard: c });
  },

  connectionStartHandle: null,
  setConnectionStartHandle: (h) => {
    set({ connectionStartHandle: h });
  },
}));
