import { Bot, Minimize2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Toaster } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import { useFlowHandlers } from "@/hooks/graph/useFlowHandlers";
import { useGraphOperations } from "@/hooks/graph/useGraphOperations";
import { type HelperLines, useHelperLines } from "@/hooks/graph/useHelperLines";
import { useRecursiveNavigation } from "@/hooks/graph/useRecursiveNavigation";
import { initChatMaterializer } from "@/hooks/integration/useChatMaterializer";
import { useFlowSocket } from "@/hooks/integration/useFlowSocket";
import { useGenericMaterializer } from "@/hooks/integration/useGenericMaterializer";
import { useSpacetimeChat } from "@/hooks/integration/useSpacetimeChat";
import { useSpacetimeSync } from "@/hooks/integration/useSpacetimeSync";
import { type PreviewData, useNodeEventListener } from "@/hooks/nodes/useNodeEventListener";
import { useAppActions } from "@/hooks/ux/useAppActions";
import { useAppHotkeys } from "@/hooks/ux/useAppHotkeys";
import { useContextMenu } from "@/hooks/ux/useContextMenu";
import { useTheme } from "@/hooks/ux/useTheme";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { type RFState } from "@/store/types";
import { NavigationStatus, useNavigationStore } from "@/store/ui/navigationStore";
import { useSettingsStore } from "@/store/ui/settingsStore";
import { useUiStore } from "@/store/uiStore";
import { ChatViewMode } from "@/types";
import { SocketStatus } from "@/utils/SocketClient";

import { Breadcrumbs } from "./components/canvas/Breadcrumbs";
import { FlowCanvas } from "./components/canvas/FlowCanvas";
import { ChatBot } from "./components/media/ChatBot";
import { ContextMenuOverlay } from "./components/menus/ContextMenuOverlay";
import { Sidebar } from "./components/Sidebar";
import { AppOverlays } from "./components/ui/AppOverlays";
import { Button } from "./components/ui/button";

function App() {
  useSpacetimeSync();
  useRecursiveNavigation();
  useTheme();
  useSpacetimeChat();

  useEffect(() => {
    initChatMaterializer();
    window.flowStore = useFlowStore;
  }, []);

  useGenericMaterializer();

  const { edges, nodes, onConnect, onEdgesChange, onNodesChange } = useFlowStore(
    useShallow((s: RFState) => ({
      edges: s.edges,
      nodes: s.nodes,
      onConnect: s.onConnect,
      onEdgesChange: s.onEdgesChange,
      onNodesChange: s.onNodesChange,
    })),
  );

  const { activeChatNodeId, chatViewMode, dragMode, isSidebarOpen, setActiveChat, setChatFullscreen } = useUiStore(
    useShallow((s) => ({
      activeChatNodeId: s.activeChatNodeId,
      chatViewMode: s.chatViewMode,
      dragMode: s.dragMode,
      isSidebarOpen: s.isSidebarOpen,
      setActiveChat: s.setActiveChat,
      setChatFullscreen: s.setChatFullscreen,
    })),
  );

  const { navigationStatus, setActiveScope } = useNavigationStore(
    useShallow((s) => ({
      navigationStatus: s.navigationStatus,
      setActiveScope: s.setActiveScope,
    })),
  );

  const { theme } = useSettingsStore();

  const [activeEditorId, setActiveEditorId] = useState<null | string>(null);
  const [previewData, setPreviewData] = useState<null | PreviewData>(null);
  const [pendingAction, setPendingAction] = useState<{ actionId: string; nodeId: string } | null>(null);
  const [helperLines, setHelperLines] = useState<HelperLines>({});

  const { cancelTask, executeTask, streamAction, templates, updateViewport } = useFlowSocket();
  const { calculateLines } = useHelperLines();
  const { autoLayout, copySelected, deleteEdge, deleteNode, duplicateSelected, groupSelected, paste } =
    useGraphOperations();

  const {
    closeContextMenu,
    contextMenu,
    onEdgeContextMenu,
    onNodeContextMenu,
    onPaneContextMenu,
    onSelectionContextMenu,
    setContextMenu,
  } = useContextMenu();

  const {
    handleMove: onMove,
    handleNodeDragStop: onNodeDragStop,
    onConnectEnd,
    onConnectStart,
    onInit,
    onNodesChangeWithSnapping,
  } = useFlowHandlers({
    calculateLines,
    contextMenuDragStop: closeContextMenu,
    nodes,
    onNodeContextMenuHook: onNodeContextMenu,
    onNodesChange: onNodesChange,
    setHelperLines: setHelperLines,
    updateViewport,
  });

  const onMoveEnd = useCallback(
    (_: unknown, viewport: { x: number; y: number; zoom: number }) => {
      const scopeId = useNavigationStore.getState().activeScopeId ?? "root";
      updateViewport(scopeId, viewport.x, viewport.y, viewport.zoom);
    },
    [updateViewport],
  );

  const onNodeDragReset = useCallback(() => {
    setHelperLines({ horizontal: undefined, vertical: undefined });
  }, []);

  useAppHotkeys();

  useNodeEventListener({
    cancelTask,
    executeTask,
    nodes,
    setActiveEditorId,
    setContextMenu,
    setPreviewData,
    streamAction,
  });

  const { exportBranch, handleAddNode, handleExecuteAction } = useAppActions(
    setPendingAction,
    contextMenu,
    closeContextMenu,
  );

  // --- Global Navigation Lock Logic ---
  useEffect(() => {
    // Pre-initialize window property
    window.lastProcessedMousePos ??= { x: 0, y: 0 };

    const handleGlobalMouseMove = (e: MouseEvent) => {
      // Store current pos for jump logic to capture
      window.lastProcessedMousePos = { x: e.clientX, y: e.clientY };

      const uiState = useUiStore.getState();
      if (!uiState.navigatingNodeId || !uiState.isNavigatingViaKeyboard) return;

      const dx = Math.abs(e.clientX - uiState.lastMousePos.x);
      const dy = Math.abs(e.clientY - uiState.lastMousePos.y);

      // If mouse moves more than 20px, it's a real intentional movement.
      // Unlock the keyboard navigation focus so standard hover can take over.
      if (dx > 20 || dy > 20) {
        uiState.resetNavigatingNode(uiState.navigatingNodeId);
      }
    };

    window.addEventListener("mousemove", handleGlobalMouseMove, { passive: true });
    return () => {
      window.removeEventListener("mousemove", handleGlobalMouseMove);
    };
  }, []);

  return (
    <div className="flex h-screen w-full flex-col bg-background overflow-hidden relative">
      <Breadcrumbs />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar />

        <main className="flex-1 relative overflow-hidden bg-muted/10">
          <FlowCanvas
            dragMode={dragMode}
            edges={edges}
            helperLines={helperLines}
            nodes={nodes}
            onConnect={onConnect}
            onConnectEnd={onConnectEnd}
            onConnectStart={onConnectStart}
            onEdgeContextMenu={onEdgeContextMenu}
            onEdgesChange={onEdgesChange}
            onInit={onInit}
            onMove={onMove}
            onMoveEnd={onMoveEnd}
            onNodeContextMenu={onNodeContextMenu}
            onNodeDragStart={onNodeDragReset}
            onNodeDragStop={onNodeDragStop}
            onNodesChange={onNodesChangeWithSnapping}
            onPaneContextMenu={onPaneContextMenu}
            onSelectionContextMenu={onSelectionContextMenu}
            theme={theme}
          />

          {/* Scope Transition Overlay */}
          <div
            className={cn(
              "absolute inset-0 z-[1000] pointer-events-none bg-background/40 backdrop-blur-sm transition-opacity duration-300",
              navigationStatus !== NavigationStatus.IDLE ? "opacity-100" : "opacity-0",
            )}
          />

          <AppOverlays
            activeEditorId={activeEditorId}
            connectionStatus={SocketStatus.CONNECTED}
            nodes={nodes}
            onExecuteAction={handleExecuteAction}
            pendingAction={pendingAction}
            previewData={previewData}
            setActiveEditorId={setActiveEditorId}
            setPendingAction={setPendingAction}
            setPreviewData={setPreviewData}
          />

          {activeChatNodeId && chatViewMode === ChatViewMode.SIDEBAR && (
            <div
              className={cn(
                "absolute right-0 top-0 bottom-0 w-[450px] bg-background border-l border-border z-50 shadow-2xl transition-transform duration-300",
                isSidebarOpen ? "translate-x-0" : "translate-x-full",
              )}
            >
              <div className="flex flex-col h-full">
                <div className="p-4 border-b border-border flex items-center justify-between bg-muted/30">
                  <div className="flex items-center gap-2">
                    <Bot className="text-primary" size={18} />
                    <span className="font-semibold text-sm">Assistant</span>
                  </div>
                  <div className="flex items-center gap-1">
                    <Button
                      aria-label="Fullscreen Assistant"
                      className="h-8 w-8"
                      onClick={() => {
                        setChatFullscreen(true);
                      }}
                      size="icon"
                      variant="ghost"
                    >
                      <Minimize2 className="rotate-45" size={14} />
                    </Button>
                    <Button
                      aria-label="Close Assistant"
                      className="h-8 w-8"
                      onClick={() => {
                        setActiveChat(null);
                      }}
                      size="icon"
                      variant="ghost"
                    >
                      <X size={14} />
                    </Button>
                  </div>
                </div>
                <div className="flex-1 overflow-hidden">
                  <ChatBot nodeId={activeChatNodeId} />
                </div>
              </div>
            </div>
          )}
        </main>
      </div>

      <ContextMenuOverlay
        availableActions={[]}
        contextMenu={contextMenu}
        edges={edges}
        nodes={nodes}
        onAddNode={handleAddNode}
        onAutoLayout={autoLayout}
        onClose={closeContextMenu}
        onCopy={copySelected}
        onDeleteEdge={deleteEdge}
        onDeleteNode={deleteNode}
        onDuplicate={duplicateSelected}
        onEnterScope={setActiveScope}
        onExecuteAction={handleExecuteAction}
        onExportBranch={exportBranch}
        onGroup={groupSelected}
        onOpenEditor={(id) => {
          setActiveEditorId(id);
        }}
        onPaste={paste}
        templates={templates}
      />

      <Toaster position="bottom-right" toastOptions={{ className: "dark-toast" }} />
    </div>
  );
}

export default App;
