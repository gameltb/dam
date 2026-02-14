import { create } from "zustand";

import { type PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { ChatViewMode, DragMode } from "@/types";

export interface LayoutState {
  // Feature Context
  activeChatNodeId: null | string;
  chatViewMode: ChatViewMode;
  connectionStartHandle: null | {
    handleId: string;
    itemType: string;
    mainType: PortMainType;
    nodeId: string;
    type: string;
  };
  // Interaction States
  dragMode: DragMode;

  isChatFullscreen: boolean;
  isNavigatingViaKeyboard: boolean;
  isSettingsOpen: boolean;
  // Panel States
  isSidebarOpen: boolean;
  kernelExplorerHeight: number;

  lastMousePos: { x: number; y: number };
  navigatingNodeId: null | string;
  resetNavigatingNode: (id: string) => void;

  setActiveChat: (nodeId: null | string, mode?: ChatViewMode) => void;
  setChatFullscreen: (fullscreen: boolean) => void;
  setConnectionStartHandle: (handle: LayoutState["connectionStartHandle"]) => void;
  setDragMode: (mode: DragMode) => void;
  setKernelExplorerHeight: (height: number) => void;
  setLastMousePos: (pos: { x: number; y: number }) => void;
  setNavigatingNode: (id: null | string, viaKeyboard?: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  // Actions
  setSidebarOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  sidebarWidth: number;
}

export const useUiStore = create<LayoutState>((set) => ({
  activeChatNodeId: null,
  chatViewMode: ChatViewMode.INLINE,
  connectionStartHandle: null,
  dragMode: DragMode.SELECT,

  isChatFullscreen: false,
  isNavigatingViaKeyboard: false,
  isSettingsOpen: false,
  isSidebarOpen: false,
  kernelExplorerHeight: 500,

  lastMousePos: { x: 0, y: 0 },
  navigatingNodeId: null,
  resetNavigatingNode: (id) => {
    set((state) => (state.navigatingNodeId === id ? { isNavigatingViaKeyboard: false, navigatingNodeId: null } : {}));
  },

  setActiveChat: (nodeId, mode = ChatViewMode.SIDEBAR) => {
    set({
      activeChatNodeId: nodeId,
      chatViewMode: nodeId ? mode : ChatViewMode.INLINE,
      isChatFullscreen: nodeId ? mode === ChatViewMode.FULLSCREEN : false,
      isSidebarOpen: nodeId ? mode === ChatViewMode.SIDEBAR : false,
    });
  },
  setChatFullscreen: (fullscreen) => {
    set({ isChatFullscreen: fullscreen });
  },
  setConnectionStartHandle: (handle) => {
    set({ connectionStartHandle: handle });
  },
  setDragMode: (mode) => {
    set({ dragMode: mode });
  },
  setKernelExplorerHeight: (height) => {
    set({ kernelExplorerHeight: height });
  },
  setLastMousePos: (pos) => {
    set({ lastMousePos: pos });
  },
  setNavigatingNode: (id, viaKeyboard = false) => {
    set({
      isNavigatingViaKeyboard: viaKeyboard,
      navigatingNodeId: id,
    });
  },
  setSettingsOpen: (open) => {
    set({ isSettingsOpen: open });
  },
  setSidebarOpen: (open) => {
    set({ isSidebarOpen: open });
  },
  setSidebarWidth: (width) => {
    set({ sidebarWidth: width });
  },
  sidebarWidth: 400,
}));
