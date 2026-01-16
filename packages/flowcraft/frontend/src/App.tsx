import { type Node as RFNode, useReactFlow } from "@xyflow/react";
import { Bot, Minimize2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Toaster } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import { type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import { useAppActions } from "@/hooks/useAppActions";
import { useAppHotkeys } from "@/hooks/useAppHotkeys";
import { useContextMenu } from "@/hooks/useContextMenu";
import { useFlowHandlers } from "@/hooks/useFlowHandlers";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useGraphOperations } from "@/hooks/useGraphOperations";
import { useHelperLines } from "@/hooks/useHelperLines";
import { type PreviewData, useNodeEventListener } from "@/hooks/useNodeEventListener";
import { useSpacetimeSync } from "@/hooks/useSpacetimeSync";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { type RFState } from "@/store/types";
import { useUiStore } from "@/store/uiStore";
import { ChatViewMode } from "@/types";
import { socketClient, SocketStatus } from "@/utils/SocketClient";

import { FlowCanvas } from "./components/canvas/FlowCanvas";
import { ContextMenuOverlay } from "./components/menus/ContextMenuOverlay";
import { ChatBot } from "./components/media/ChatBot";
import { Sidebar } from "./components/Sidebar";
import { AppOverlays } from "./components/ui/AppOverlays";
import { Button } from "./components/ui/button";

function App() {
  useSpacetimeSync();
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
  const { activeChatNodeId, dragMode, isChatFullscreen, setActiveChat, settings } = useUiStore();
  const { cancelTask, executeTask, streamAction, templates, updateViewport } = useFlowSocket();
  const { calculateLines, helperLines, setHelperLines } = useHelperLines();
  const { screenToFlowPosition: _screenToFlowPosition } = useReactFlow();
  useTheme();

  const [connectionStatus, setConnectionStatus] = useState<SocketStatus>(socketClient.getStatus());
  const [availableActions, setAvailableActions] = useState<ActionTemplate[]>([]);
  const [pendingAction, setPendingAction] = useState<ActionTemplate | null>(null);
  const [previewData, setPreviewData] = useState<null | PreviewData>(null);
  const [activeEditorId, setActiveEditorId] = useState<null | string>(null);

  const {
    closeContextMenuAndClear,
    contextMenu,
    onEdgeContextMenu,
    onNodeContextMenu,
    onNodeDragStop: contextMenuDragStop,
    onPaneContextMenu,
    onSelectionContextMenu,
    setContextMenu,
  } = useContextMenu();

  const {
    handleAddNode: _handleAddNode,
    handleDrop,
    handleExecuteAction,
  } = useAppActions(setPendingAction, contextMenu, closeContextMenuAndClear);

  const {
    addNode: _addNode,
    autoLayout: _autoLayout,
    copySelected: _copySelected,
    deleteEdge: _deleteEdge,
    deleteNode: _deleteNode,
    duplicateSelected: _duplicateSelected,
    groupSelected: _groupSelected,
    paste: _paste,
  } = useGraphOperations();

  const {
    handleMoveEnd,
    handleNodeContextMenu,
    handleNodeDragStop,
    onConnectEnd,
    onConnectStart,
    onInit,
    onNodesChangeWithSnapping,
  } = useFlowHandlers({
    calculateLines,
    contextMenuDragStop,
    nodes,
    onNodeContextMenuHook: onNodeContextMenu,
    onNodesChange,
    setHelperLines,
    updateViewport,
  });

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

  useEffect(() => {
    const s = (status: SocketStatus) => {
      setConnectionStatus(status);
    };
    socketClient.on("statusChange", s);
    socketClient.on("actions", (actions) => {
      setAvailableActions(actions);
    });
    return () => {
      socketClient.off("statusChange", s);
    };
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      handleDrop(event, templates);
    },
    [handleDrop, templates],
  );

  const onNodeDragStart = useCallback((_e: React.MouseEvent, node: RFNode) => {
    const nativeEvent = _e.nativeEvent as DragEvent | undefined;
    if (nativeEvent?.dataTransfer) {
      nativeEvent.dataTransfer.setData(
        "application/flowcraft-node",
        JSON.stringify({ id: node.id, label: node.data.displayName }),
      );
      nativeEvent.dataTransfer.effectAllowed = "move";
    }
  }, []);

  return (
    <div
      className="flex w-screen h-screen overflow-hidden bg-background text-foreground"
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = "move";
      }}
      onDrop={onDrop}
    >
      <div
        className={cn(
          "flex-1 relative h-full min-w-0 fc-canvas-area",
          isChatFullscreen && "pointer-events-none opacity-50 grayscale-[0.5]",
        )}
        inert={isChatFullscreen ? true : undefined}
      >
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
          onMoveEnd={handleMoveEnd}
          onNodeContextMenu={handleNodeContextMenu}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={handleNodeDragStop}
          onNodesChange={onNodesChangeWithSnapping}
          onPaneContextMenu={onPaneContextMenu}
          onSelectionContextMenu={onSelectionContextMenu}
          theme={settings.theme}
        />
        <AppOverlays
          activeEditorId={activeEditorId}
          connectionStatus={connectionStatus}
          nodes={nodes}
          onExecuteAction={handleExecuteAction}
          pendingAction={pendingAction}
          previewData={previewData}
          setActiveEditorId={setActiveEditorId}
          setPendingAction={setPendingAction}
          setPreviewData={setPreviewData}
        />
        <ContextMenuOverlay
          availableActions={availableActions}
          contextMenu={contextMenu}
          edges={edges}
          nodes={nodes}
          onAddNode={_handleAddNode}
          onAutoLayout={_autoLayout}
          onClose={closeContextMenuAndClear}
          onCopy={_copySelected}
          onDeleteEdge={_deleteEdge}
          onDeleteNode={_deleteNode}
          onDuplicate={_duplicateSelected}
          onExecuteAction={handleExecuteAction}
          onGroup={_groupSelected}
          onOpenEditor={setActiveEditorId}
          onPaste={_paste}
          templates={templates}
        />
      </div>
      <div className={cn(isChatFullscreen && "pointer-events-none")} inert={isChatFullscreen ? true : undefined}>
        <Sidebar />
      </div>
      {isChatFullscreen && activeChatNodeId && (
        <div className="fixed inset-0 z-[10000] bg-background flex flex-col animate-in fade-in zoom-in-95 duration-200">
          <div className="shrink-0 p-4 border-b border-node-border flex justify-between items-center bg-muted/20">
            <div className="flex items-center gap-2">
              <Bot className="text-primary-color" size={20} />
              <h2 className="font-bold">Full Conversation Mode</h2>
            </div>
            <div className="flex gap-2">
              <Button
                onClick={() => {
                  setActiveChat(activeChatNodeId, ChatViewMode.INLINE);
                }}
                size="icon"
                variant="ghost"
              >
                <Minimize2 size={18} />
              </Button>
              <Button
                onClick={() => {
                  setActiveChat(null);
                }}
                size="icon"
                variant="ghost"
              >
                <X size={18} />
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden relative">
            <ChatBot nodeId={activeChatNodeId} />
          </div>
        </div>
      )}
      <Toaster position="bottom-right" />
    </div>
  );
}

export default App;
