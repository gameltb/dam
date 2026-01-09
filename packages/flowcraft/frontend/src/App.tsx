import { create, type JsonObject } from "@bufbuild/protobuf";
import { type Node as RFNode, useReactFlow } from "@xyflow/react";
import { Bot, Minimize2, X } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { Toaster } from "react-hot-toast";
import { useShallow } from "zustand/react/shallow";

import {
  ActionExecutionRequestSchema,
  type ActionTemplate,
} from "@/generated/flowcraft/v1/core/action_pb";
import { useAppHotkeys } from "@/hooks/useAppHotkeys";
import { useContextMenu } from "@/hooks/useContextMenu";
import { useFlowHandlers } from "@/hooks/useFlowHandlers";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { useGraphOperations } from "@/hooks/useGraphOperations";
import { useHelperLines } from "@/hooks/useHelperLines";
import {
  type PreviewData,
  useNodeEventListener,
} from "@/hooks/useNodeEventListener";
import { useTheme } from "@/hooks/useTheme";
import { cn } from "@/lib/utils";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { type RFState } from "@/store/types";
import { useUiStore } from "@/store/uiStore";
import {
  type AppNode,
  ChatViewMode,
  MutationSource,
  type NodeTemplate,
} from "@/types";
import { fromProtoNodeData } from "@/utils/protoAdapter";
import { socketClient, SocketStatus } from "@/utils/SocketClient";

import { FlowCanvas } from "./components/canvas/FlowCanvas";
import { ChatBot } from "./components/media/ChatBot";
import { ContextMenuOverlay } from "./components/menus/ContextMenuOverlay";
import { Sidebar } from "./components/Sidebar";
import { AppOverlays } from "./components/ui/AppOverlays";
import { Button } from "./components/ui/button";

function App() {
  const {
    addNode: addNodeToStore,
    edges,
    nodes,
    onConnect,
    onEdgesChange,
    onNodesChange,
    version: clientVersion,
  } = useFlowStore(
    useShallow((s: RFState) => ({
      addNode: s.addNode,
      edges: s.edges,
      nodes: s.nodes,
      onConnect: s.onConnect,
      onEdgesChange: s.onEdgesChange,
      onNodesChange: s.onNodesChange,
      version: s.version,
    })),
  );
  const {
    activeChatNodeId,
    dragMode,
    isChatFullscreen,
    setActiveChat,
    settings,
  } = useUiStore();
  const { cancelTask, executeTask, streamAction, templates, updateViewport } =
    useFlowSocket();
  const { calculateLines, helperLines, setHelperLines } = useHelperLines();
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
    addNode,
    autoLayout,
    copySelected,
    deleteEdge,
    deleteNode,
    duplicateSelected,
    groupSelected,
    paste,
  } = useGraphOperations({ clientVersion });

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
    addNodeToStore,
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

  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, unknown> = {}) => {
      if (action.paramsSchema && Object.keys(params).length === 0) {
        setPendingAction(action);
        closeContextMenuAndClear();
        return;
      }
      const effectiveNodeId =
        contextMenu?.nodeId ?? nodes.find((n: AppNode) => n.selected)?.id ?? "";
      if (
        !effectiveNodeId &&
        nodes.filter((n: AppNode) => n.selected).length === 0
      )
        return;

      const taskId = crypto.randomUUID();
      useTaskStore.getState().registerTask({
        label: action.label,
        source: MutationSource.SOURCE_REMOTE_TASK,
        taskId,
      });

      void socketClient.send({
        payload: {
          case: "actionExecute",
          value: create(ActionExecutionRequestSchema, {
            actionId: action.id,
            contextNodeIds: nodes
              .filter((n: AppNode) => n.selected)
              .map((n: AppNode) => n.id),
            params: {
              case: "paramsStruct",
              value: { ...params, taskId } as JsonObject,
            },
            sourceNodeId: effectiveNodeId,
          }),
        },
      });
      setPendingAction(null);
      closeContextMenuAndClear();
    },
    [closeContextMenuAndClear, contextMenu, nodes],
  );

  const handleAddNode = useCallback(
    (template: NodeTemplate) => {
      if (!contextMenu) return;
      const pos = screenToFlowPosition({ x: contextMenu.x, y: contextMenu.y });
      addNode(
        template.templateId,
        pos,
        {
          ...(template.defaultState
            ? fromProtoNodeData(template.defaultState)
            : {}),
          label: template.displayName,
        },
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
        const files = event.dataTransfer.files;
        if (files.length === 0) return;
        const position = screenToFlowPosition({
          x: event.clientX,
          y: event.clientY,
        });
        for (const file of Array.from(files)) {
          const formData = new FormData();
          formData.append("file", file);
          const res = await fetch("/api/upload", {
            body: formData,
            method: "POST",
          });
          const asset = (await res.json()) as {
            mimeType: string;
            name: string;
            url: string;
          };
          let tpl = "flowcraft.node.media.document";
          if (asset.mimeType.startsWith("image/"))
            tpl = "flowcraft.node.media.visual";
          const template = templates.find((t) => t.templateId === tpl);
          addNode(tpl, position, {
            ...(template?.defaultState
              ? fromProtoNodeData(template.defaultState)
              : {}),
            label: asset.name
              ? asset.name
              : (template?.displayName ?? "New Asset"),
            widgetsValues: { mimeType: asset.mimeType, url: asset.url },
          });
        }
      };
      void dropLogic().catch((e: unknown) => {
        console.error(e);
      });
    },
    [addNode, templates, screenToFlowPosition],
  );

  const onNodeDragStart = useCallback((_e: React.MouseEvent, node: RFNode) => {
    const nativeEvent = _e.nativeEvent as DragEvent | undefined;
    if (nativeEvent?.dataTransfer) {
      nativeEvent.dataTransfer.setData(
        "application/flowcraft-node",
        JSON.stringify({ id: node.id, label: node.data.label }),
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
      </div>
      <div
        className={cn(isChatFullscreen && "pointer-events-none")}
        inert={isChatFullscreen ? true : undefined}
      >
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
                size="sm"
                variant="outline"
              >
                <Minimize2 className="mr-2" size={16} /> Dock to Node
              </Button>
              <Button
                onClick={() => {
                  setActiveChat(null);
                }}
                size="icon"
                variant="ghost"
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
        availableActions={availableActions}
        contextMenu={contextMenu}
        edges={edges}
        nodes={nodes}
        onAddNode={handleAddNode}
        onAutoLayout={autoLayout}
        onClose={closeContextMenuAndClear}
        onCopy={copySelected}
        onDeleteEdge={deleteEdge}
        onDeleteNode={deleteNode}
        onDuplicate={duplicateSelected}
        onExecuteAction={handleExecuteAction}
        onGroup={groupSelected}
        onOpenEditor={setActiveEditorId}
        onPaste={paste}
        templates={templates}
      />
      <Toaster position="bottom-right" />
    </div>
  );
}

export default App;
