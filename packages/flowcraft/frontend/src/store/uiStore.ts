import { create } from "zustand";
import { type PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { ChatViewMode, DragMode } from "@/types";

export interface LayoutState {
  // Panel States
  isSidebarOpen: boolean;
  isSettingsOpen: boolean;
  sidebarWidth: number;
  kernelExplorerHeight: number;
  
  // Interaction States
  dragMode: DragMode;
  lastMousePos: { x: number; y: number };
  navigatingNodeId: string | null;
  isNavigatingViaKeyboard: boolean;
  connectionStartHandle: null | {
    handleId: string;
    itemType: string;
    mainType: PortMainType;
    nodeId: string;
    type: string;
  };

  // Feature Context
  activeChatNodeId: string | null;
  chatViewMode: ChatViewMode;
  isChatFullscreen: boolean;

  // Actions
  setSidebarOpen: (open: boolean) => void;
  setSettingsOpen: (open: boolean) => void;
  setSidebarWidth: (width: number) => void;
  setKernelExplorerHeight: (height: number) => void;
  setDragMode: (mode: DragMode) => void;
  setLastMousePos: (pos: { x: number; y: number }) => void;
  setNavigatingNode: (id: string | null, viaKeyboard?: boolean) => void;
  resetNavigatingNode: (id: string) => void;
  setConnectionStartHandle: (handle: LayoutState["connectionStartHandle"]) => void;
  setActiveChat: (nodeId: string | null, mode?: ChatViewMode) => void;
  setChatFullscreen: (fullscreen: boolean) => void;
}

export const useUiStore = create<LayoutState>((set) => ({
  isSidebarOpen: false,
  isSettingsOpen: false,
  sidebarWidth: 400,
  kernelExplorerHeight: 500,
  
  dragMode: DragMode.SELECT,
  lastMousePos: { x: 0, y: 0 },
  navigatingNodeId: null,
  isNavigatingViaKeyboard: false,
  connectionStartHandle: null,

  activeChatNodeId: null,
  chatViewMode: ChatViewMode.INLINE,
  isChatFullscreen: false,

  setSidebarOpen: (open) => set({ isSidebarOpen: open }),
  setSettingsOpen: (open) => set({ isSettingsOpen: open }),
  setSidebarWidth: (width) => set({ sidebarWidth: width }),
  setKernelExplorerHeight: (height) => set({ kernelExplorerHeight: height }),
  setDragMode: (mode) => set({ dragMode: mode }),
  setLastMousePos: (pos) => set({ lastMousePos: pos }),
  setNavigatingNode: (id, viaKeyboard = false) => set({ 
    isNavigatingViaKeyboard: viaKeyboard, 
    navigatingNodeId: id 
  }),
  resetNavigatingNode: (id) => set((state) => 
    state.navigatingNodeId === id ? { isNavigatingViaKeyboard: false, navigatingNodeId: null } : {}
  ),
  setConnectionStartHandle: (handle) => set({ connectionStartHandle: handle }),
  setActiveChat: (nodeId, mode = ChatViewMode.SIDEBAR) => set({
    activeChatNodeId: nodeId,
    chatViewMode: nodeId ? mode : ChatViewMode.INLINE,
    isChatFullscreen: nodeId ? mode === ChatViewMode.FULLSCREEN : false,
    isSidebarOpen: nodeId ? mode === ChatViewMode.SIDEBAR : false,
  }),
  setChatFullscreen: (fullscreen) => set({ isChatFullscreen: fullscreen }),
}));
