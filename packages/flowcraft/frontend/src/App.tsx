/* eslint-disable @typescript-eslint/no-unsafe-call */
/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useCallback, useEffect } from "react";
import { useTheme } from "./hooks/useTheme";
import { useReactFlow } from "@xyflow/react";
import { useFlowStore } from "./store/flowStore";
import { useUiStore } from "./store/uiStore";
import { useTaskStore } from "./store/taskStore";
import "@xyflow/react/dist/style.css";
import { useFlowSocket } from "./hooks/useFlowSocket";
import { MutationSource, type AppNode } from "./types";
import { type ActionTemplate } from "./generated/flowcraft/v1/core/action_pb";
import { Toaster } from "react-hot-toast";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import { useHelperLines } from "./hooks/useHelperLines";
import { socketClient, SocketStatus } from "./utils/SocketClient";
import { Sidebar } from "./components/Sidebar";
import { ChatBot } from "./components/media/ChatBot";
import { X, Minimize2, Bot } from "lucide-react";
import { Button } from "./components/ui/button";
import { useShallow } from "zustand/react/shallow";
import { useAppHotkeys } from "./hooks/useAppHotkeys";
import {
  useNodeEventListener,
  type PreviewData,
} from "./hooks/useNodeEventListener";
import { useFlowHandlers } from "./hooks/useFlowHandlers";
import { fromProtoNodeData } from "./utils/protoAdapter";
import { cn } from "./lib/utils";
import { FlowCanvas } from "./components/canvas/FlowCanvas";
import { AppOverlays } from "./components/ui/AppOverlays";
import { ContextMenuOverlay } from "./components/menus/ContextMenuOverlay";

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    version: clientVersion,
    addNode: addNodeToStore,
  } = useFlowStore(
    useShallow((s) => ({
      nodes: s.nodes,
      edges: s.edges,
      onNodesChange: s.onNodesChange,
      onEdgesChange: s.onEdgesChange,
      onConnect: s.onConnect,
      version: s.version,
      addNode: s.addNode,
    })),
  );
  const {
    dragMode,
    settings,
    isChatFullscreen,
    activeChatNodeId,
    setActiveChat,
  } = useUiStore();
  const { templates, updateViewport, streamAction, cancelTask, executeTask } =
    useFlowSocket();
  const { helperLines, setHelperLines, calculateLines } = useHelperLines();
  const { screenToFlowPosition } = useReactFlow();
  useTheme();

  const [connectionStatus, setConnectionStatus] = useState<SocketStatus>(
    socketClient.getStatus(),
  );
  const [availableActions, setAvailableActions] = useState<ActionTemplate[]>(
    [],
  );
  const [pendingAction, setPendingAction] = useState<ActionTemplate | null>(
    null,
  );
  const [previewData, setPreviewData] = useState<PreviewData | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);

  const {
    contextMenu,
    onNodeContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu,
    onPaneContextMenu,
    closeContextMenuAndClear,
    setContextMenu,
    onNodeDragStop: contextMenuDragStop,
  } = useContextMenu();

  const {
    addNode,
    autoLayout,
    groupSelected,
    copySelected,
    duplicateSelected,
    paste,
    deleteNode,
    deleteEdge,
  } = useGraphOperations({ clientVersion });

  const {
    onInit,
    handleNodeDragStop,
    onConnectStart,
    onConnectEnd,
    onNodesChangeWithSnapping,
    handleMoveEnd,
    handleNodeContextMenu,
  } = useFlowHandlers({
    nodes,
    onNodesChange,
    updateViewport,
    calculateLines,
    setHelperLines,
    onNodeContextMenuHook: onNodeContextMenu,
    contextMenuDragStop,
  });

  useAppHotkeys();
  useNodeEventListener({
    nodes,
    setContextMenu,
    setPreviewData,
    setActiveEditorId,
    streamAction,
    addNodeToStore,
    cancelTask,
    executeTask,
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

  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, any> = {}) => {
      if (action.paramsSchemaJson && Object.keys(params).length === 0) {
        setPendingAction(action);
        closeContextMenuAndClear();
        return;
      }
      const effectiveNodeId =
        contextMenu?.nodeId || nodes.find((n) => n.selected)?.id || "";
      if (!effectiveNodeId && nodes.filter((n) => n.selected).length === 0)
        return;

      const taskId = crypto.randomUUID();
      useTaskStore.getState().registerTask({
        taskId,
        label: action.label,
        source: MutationSource.SOURCE_REMOTE_TASK,
      });

      void socketClient.send({
        payload: {
          case: "actionExecute",
          value: {
            actionId: action.id,
            sourceNodeId: effectiveNodeId,
            contextNodeIds: nodes.filter((n) => n.selected).map((n) => n.id),
            paramsJson: JSON.stringify({ ...params, taskId }),
          } as any,
        },
      });
      setPendingAction(null);
      closeContextMenuAndClear();
    },
    [closeContextMenuAndClear, contextMenu, nodes],
  );

  const handleAddNode = useCallback(
    (template: any) => {
      if (!contextMenu) return;
      const pos = screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });
      addNode(
        template.templateId,
        pos,
        {},
        template.defaultWidth,
        template.defaultHeight,
      );
    },
    [addNode, contextMenu, screenToFlowPosition],
  );

  const onDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      const dropLogic = async () => {
        const files = event.dataTransfer?.files;
        if (!files || files.length === 0) return;
        const position = screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });
        for (const file of Array.from(files)) {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/upload", {
            method: "POST",
            body: formData,
          });
          const asset = await res.json();
          let tpl = "tpl-media-md";
          if (asset.mimeType.startsWith("image/")) tpl = "tpl-media-image";
          const template = templates.find((t) => t.templateId === tpl);
          addNode(tpl, position, {
            ...(template?.defaultState
              ? fromProtoNodeData(template.defaultState)
              : {}),
            label: asset.name,
            widgetsValues: { url: asset.url, mimeType: asset.mimeType },
          });
        }
      };
      void dropLogic().catch((e: unknown) => {
        console.error(e);
      });
    },
    [addNode, templates, screenToFlowPosition],
  );

  const onNodeDragStart = useCallback((_e: any, node: AppNode) => {
    const nativeEvent = _e?.nativeEvent as DragEvent | undefined;
    if (nativeEvent?.dataTransfer) {
      nativeEvent.dataTransfer.setData(
        "application/flowcraft-node",
        JSON.stringify({ label: node.data.label, id: node.id }),
      );
      nativeEvent.dataTransfer.effectAllowed = "move";
    }
  }, []);

  return (
    <div
      className="flex w-screen h-screen overflow-hidden bg-background text-foreground"
      onDrop={onDrop}
      onDragOver={(e) => {
        e.preventDefault();
        if (e.dataTransfer) e.dataTransfer.dropEffect = "move";
      }}
    >
      <div
        className={cn(
          "flex-1 relative h-full min-w-0 fc-canvas-area",
          isChatFullscreen && "pointer-events-none opacity-50 grayscale-[0.5]",
        )}
        inert={isChatFullscreen || undefined}
      >
        <FlowCanvas
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChangeWithSnapping}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={onInit}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={handleNodeDragStop}
          onConnectStart={onConnectStart}
          onConnectEnd={onConnectEnd}
          onNodeContextMenu={handleNodeContextMenu}
          onEdgeContextMenu={onEdgeContextMenu}
          onSelectionContextMenu={onSelectionContextMenu}
          onPaneContextMenu={onPaneContextMenu}
          onMoveEnd={handleMoveEnd}
          theme={settings.theme}
          dragMode={dragMode}
          helperLines={helperLines}
        />
        <AppOverlays
          nodes={nodes}
          previewData={previewData}
          setPreviewData={setPreviewData}
          activeEditorId={activeEditorId}
          setActiveEditorId={setActiveEditorId}
          connectionStatus={connectionStatus}
          pendingAction={pendingAction}
          setPendingAction={setPendingAction}
          onExecuteAction={handleExecuteAction}
        />
      </div>
      <div
        className={cn(isChatFullscreen && "pointer-events-none")}
        inert={isChatFullscreen || undefined}
      >
        <Sidebar />
      </div>
      {isChatFullscreen && activeChatNodeId && (
        <div className="fixed inset-0 z-[10000] bg-background flex flex-col animate-in fade-in zoom-in-95 duration-200">
          <div className="shrink-0 p-4 border-b border-node-border flex justify-between items-center bg-muted/20">
            <div className="flex items-center gap-2">
              <Bot size={20} className="text-primary-color" />
              <h2 className="font-bold">Full Conversation Mode</h2>
            </div>
            <div className="flex gap-2">
              <Button
                variant="outline"
                size="sm"
                onClick={() => {
                  setActiveChat(activeChatNodeId, "inline");
                }}
              >
                <Minimize2 size={16} className="mr-2" /> Dock to Node
              </Button>
              <Button
                variant="ghost"
                size="icon"
                onClick={() => {
                  setActiveChat(null);
                }}
              >
                <X size={20} />
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden ai-theme-container">
            <ChatBot nodeId={activeChatNodeId} />
          </div>
        </div>
      )}
      <ContextMenuOverlay
        contextMenu={contextMenu}
        nodes={nodes}
        edges={edges}
        templates={templates}
        availableActions={availableActions}
        onClose={closeContextMenuAndClear}
        onDeleteNode={deleteNode}
        onDeleteEdge={deleteEdge}
        onOpenEditor={setActiveEditorId}
        onCopy={copySelected}
        onDuplicate={duplicateSelected}
        onGroup={groupSelected}
        onAutoLayout={autoLayout}
        onPaste={paste}
        onAddNode={handleAddNode}
        onExecuteAction={handleExecuteAction}
      />
      <Toaster position="bottom-right" />
    </div>
  );
}

export default App;
