import { Bot, Minimize2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Toaster } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import { type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { useAppActions } from "@/hooks/useAppActions";
import { useAppHotkeys } from "@/hooks/useAppHotkeys";
import { initChatMaterializer } from "@/hooks/useChatMaterializer";
import { useContextMenu } from "@/hooks/useContextMenu";
import { useFlowHandlers } from "@/hooks/useFlowHandlers";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useGenericMaterializer } from "@/hooks/useGenericMaterializer";
import { useGraphOperations } from "@/hooks/useGraphOperations";
import { type HelperLines, useHelperLines } from "@/hooks/useHelperLines";
import { type PreviewData, useNodeEventListener } from "@/hooks/useNodeEventListener";
import { useSpacetimeChat } from "@/hooks/useSpacetimeChat";
import { useSpacetimeSync } from "@/hooks/useSpacetimeSync";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { type RFState } from "@/store/types";
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
import { useRecursiveNavigation } from "./hooks/useRecursiveNavigation";

function App() {
  useSpacetimeSync();
  useRecursiveNavigation();
  useTheme();
  useSpacetimeChat();

  useEffect(() => {
    initChatMaterializer();
    (window as any).flowStore = useFlowStore;
  }, []);

  useGenericMaterializer();

  const {
    addNode: _addNodeToStore,
    edges,
    nodes,
    onConnect,
    onEdgesChange,
    onNodesChange,
  } = useFlowStore(
    useShallow((s: RFState) => ({
      addNode: s.addNode,
      edges: s.edges,
      nodes: s.nodes,
      onConnect: s.onConnect,
      onEdgesChange: s.onEdgesChange,
      onNodesChange: s.onNodesChange,
    })),
  );

  const {
    activeChatNodeId,
    chatViewMode,
    dragMode,
    isSidebarOpen,
    setActiveChat,
    setActiveScope,
    setChatFullscreen,
    settings,
  } = useUiStore(
    useShallow((s) => ({
      activeChatNodeId: s.activeChatNodeId,
      chatViewMode: s.chatViewMode,
      dragMode: s.dragMode,
      isSidebarOpen: s.isSidebarOpen,
      setActiveChat: s.setActiveChat,
      setActiveScope: s.setActiveScope,
      setChatFullscreen: s.setChatFullscreen,
      settings: s.settings,
    })),
  );

  const [activeEditorId, setActiveEditorId] = useState<null | string>(null);
  const [previewData, setPreviewData] = useState<null | PreviewData>(null);
  const [pendingAction, setPendingAction] = useState<ActionTemplate | null>(null);
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

  const { onConnectEnd, onConnectStart, onInit, onNodesChangeWithSnapping } = useFlowHandlers({
    calculateLines,
    contextMenuDragStop: closeContextMenu,
    nodes,
    onNodeContextMenuHook: onNodeContextMenu,
    onNodesChange: onNodesChange,
    setHelperLines: setHelperLines,
    updateViewport,
  });

  const onMoveEnd = useCallback(
    (_: any, viewport: { x: number; y: number; zoom: number }) => {
      updateViewport(viewport.x, viewport.y, viewport.zoom);
    },
    [updateViewport],
  );

  const onNodeDragReset = useCallback(() => {
    setHelperLines({ horizontal: undefined, vertical: undefined });
  }, []);

  useAppHotkeys();

  useNodeEventListener({
    addNodeToStore: _addNodeToStore,
    cancelTask,
    executeTask,
    nodes,
    setActiveEditorId,
    setContextMenu,
    setPreviewData,
    streamAction,
  });

  const { handleAddNode, handleExecuteAction } = useAppActions(setPendingAction, contextMenu, closeContextMenu);

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
            onMoveEnd={onMoveEnd}
            onNodeContextMenu={onNodeContextMenu}
            onNodeDragStart={onNodeDragReset}
            onNodeDragStop={onNodeDragReset}
            onNodesChange={onNodesChangeWithSnapping}
            onPaneContextMenu={onPaneContextMenu}
            onSelectionContextMenu={onSelectionContextMenu}
            theme={settings.theme}
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
