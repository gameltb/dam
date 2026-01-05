import { useState, memo, useCallback, useEffect } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import { useTheme } from "./hooks/useTheme";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  useReactFlow,
} from "@xyflow/react";
import { useFlowStore } from "./store/flowStore";
import { useUiStore } from "./store/uiStore";
import { useTaskStore } from "./store/taskStore";
import "@xyflow/react/dist/style.css";
import { useFlowSocket } from "./hooks/useFlowSocket";
import { PaneContextMenu } from "./components/menus/PaneContextMenu";
import { NodeContextMenu } from "./components/menus/NodeContextMenu";
import { EdgeContextMenu } from "./components/menus/EdgeContextMenu";
import { GalleryItemContextMenu } from "./components/menus/GalleryItemContextMenu";
import { MutationSource } from "./types";
import type { NodeTemplate } from "./types";
import {
  type ActionTemplate,
  ActionExecutionRequestSchema,
  ActionDiscoveryRequestSchema,
} from "./generated/flowcraft/v1/core/action_pb";
import { create } from "@bufbuild/protobuf";
import { Toaster } from "react-hot-toast";
import { Notifications } from "./components/Notifications";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import { useHelperLines } from "./hooks/useHelperLines";
import { HelperLinesRenderer } from "./components/HelperLinesRenderer";
import { MediaPreview } from "./components/media/MediaPreview";
import { EditorPlaceholder } from "./components/media/EditorPlaceholder";
import { TaskHistoryDrawer } from "./components/TaskHistoryDrawer";
import { socketClient, SocketStatus } from "./utils/SocketClient";
import { SideToolbar } from "./components/SideToolbar";
import { SettingsModal } from "./components/SettingsModal";
import { ActionParamsModal } from "./components/ActionParamsModal";
import { Sidebar } from "./components/Sidebar";
import { ChatBot } from "./components/media/ChatBot";
import { X, Minimize2, Bot } from "lucide-react";
import { Button } from "./components/ui/button";

import { useShallow } from "zustand/react/shallow";
import { SelectionMode } from "@xyflow/react";
import {
  nodeTypes,
  edgeTypes,
  defaultEdgeOptions,
  snapGrid,
} from "./flowConfig";
import { useAppHotkeys } from "./hooks/useAppHotkeys";
import { useNodeEventListener } from "./hooks/useNodeEventListener";
import { useFlowHandlers } from "./hooks/useFlowHandlers";
import { fromProtoNodeData } from "./utils/protoAdapter";
import type { DynamicNodeData, AppNode } from "./types";
import { cn } from "./lib/utils";

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode: addNodeToStore,
    version: clientVersion,
  } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addNode: state.addNode,
      version: state.version,
    })),
  );

  const { dragMode, settings, isChatFullscreen, activeChatNodeId, setActiveChat } = useUiStore();

  const flowSocket = useFlowSocket();
  const { cancelTask, executeTask, streamAction, templates, updateViewport } =
    flowSocket;
  useTheme();
  const { helperLines, setHelperLines, calculateLines } = useHelperLines();

  const [connectionStatus, setConnectionStatus] = useState<SocketStatus>(
    socketClient.getStatus(),
  );
  const [availableActions, setAvailableActions] = useState<ActionTemplate[]>(
    [],
  );
  const [previewData, setPreviewData] = useState<{
    nodeId: string;
    index: number;
  } | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);
  const [pendingAction, setPendingAction] = useState<ActionTemplate | null>(
    null,
  );

  useEffect(() => {
    socketClient.updateBaseUrl(settings.serverAddress);
  }, [settings.serverAddress]);

  useEffect(() => {
    const onActions = (actions: unknown) => {
      setAvailableActions(actions as ActionTemplate[]);
    };
    const onStatusChange = (status: SocketStatus) => {
      setConnectionStatus(status);
    };

    socketClient.on("actions", onActions);
    socketClient.on("statusChange", onStatusChange);

    return () => {
      socketClient.off("actions", onActions);
      socketClient.off("statusChange", onStatusChange);
    };
  }, []);

  const {
    onNodeContextMenu: onNodeContextMenuHook,
    onPaneContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu: onSelectionContextMenuBase,
    closeContextMenuAndClear,
    contextMenu,
    setContextMenu,
    onNodeDragStop: contextMenuDragStop,
  } = useContextMenu();

  const onSelectionContextMenu = useCallback(
    (event: ReactMouseEvent) => {
      onSelectionContextMenuBase(event);
      const selectedNodeIds = nodes.filter((n) => n.selected).map((n) => n.id);
      if (selectedNodeIds.length > 0) {
        void socketClient.send({
          payload: {
            case: "actionDiscovery",
            value: create(ActionDiscoveryRequestSchema, {
              nodeId: "",
              selectedNodeIds,
            }),
          },
        });
      }
    },
    [onSelectionContextMenuBase, nodes],
  );

  // --- Logic Extraction Hooks ---
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

  const {
    handleMoveEnd,
    handleNodeDragStop,
    onNodesChangeWithSnapping,
    onConnectStart,
    onConnectEnd,
    handleNodeContextMenu,
    onInit,
  } = useFlowHandlers({
    nodes,
    onNodesChange,
    updateViewport,
    calculateLines,
    setHelperLines,
    onNodeContextMenuHook,
    contextMenuDragStop,
  });

  const {
    autoLayout,
    addNode: addNodeOp,
    groupSelected,
    copySelected,
    paste,
    duplicateSelected,
    deleteNode,
    deleteEdge,
  } = useGraphOperations({
    clientVersion,
  });

  const { screenToFlowPosition } = useReactFlow();

  const handleAddNode = useCallback(
    (template: NodeTemplate) => {
      if (!contextMenu) return;
      const flowPosition = screenToFlowPosition({
        x: contextMenu.x,
        y: contextMenu.y,
      });
      addNodeOp(
        template.templateId,
        flowPosition,
        {},
        template.defaultWidth,
        template.defaultHeight,
      );
    },
    [addNodeOp, contextMenu, screenToFlowPosition],
  );

  const contextNodeId = contextMenu?.nodeId;
  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, any> = {}) => {
      if (action.paramsSchemaJson && Object.keys(params).length === 0) {
        setPendingAction(action);
        closeContextMenuAndClear();
        return;
      }

      const effectiveNodeId =
        contextNodeId || nodes.find((n) => n.selected)?.id || "";
      if (!effectiveNodeId && nodes.filter((n) => n.selected).length === 0)
        return;

      const selectedIds = nodes.filter((n) => n.selected).map((n) => n.id);

      const taskId = crypto.randomUUID();
      useTaskStore.getState().registerTask({
        taskId,
        label: action.label,
        source: MutationSource.SOURCE_REMOTE_TASK,
      });

      void socketClient.send({
        payload: {
          case: "actionExecute",
          value: create(ActionExecutionRequestSchema, {
            actionId: action.id,
            sourceNodeId: effectiveNodeId,
            contextNodeIds: selectedIds,
            paramsJson: JSON.stringify({ ...params, taskId }),
          }),
        },
      });

      setPendingAction(null);
      closeContextMenuAndClear();
    },
    [closeContextMenuAndClear, contextNodeId, nodes],
  );

  const onDrop = useCallback(
    async (event: React.DragEvent) => {
      event.preventDefault();

      const files = event.dataTransfer.files;
      if (files.length === 0) return;

      const reactFlowBounds = document
        .querySelector(".react-flow")
        ?.getBoundingClientRect();
      if (!reactFlowBounds) return;

      const position = screenToFlowPosition({
        x: event.clientX,
        y: event.clientY,
      });

      for (const file of Array.from(files)) {
        const formData = new FormData();
        formData.append("file", file);

        try {
          const response = await fetch("/api/upload", {
            method: "POST",
            body: formData,
          });
          const asset = await response.json();

          let templateId = "tpl-media-md";
          if (asset.mimeType.startsWith("image/"))
            templateId = "tpl-media-image";
          else if (asset.mimeType.startsWith("video/"))
            templateId = "tpl-media-video";
          else if (asset.mimeType.startsWith("audio/"))
            templateId = "tpl-media-audio";

          const template = templates.find((t) => t.templateId === templateId);
          const defaultData = (
            template?.defaultState
              ? fromProtoNodeData(template.defaultState)
              : {}
          ) as Partial<DynamicNodeData>;

          addNodeOp(templateId, position, {
            ...defaultData,
            label: asset.name,
            widgetsValues: {
              ...(defaultData.widgetsValues || {}),
              url: asset.url,
              mimeType: asset.mimeType,
            },
          });
        } catch (err) {
          console.error("Upload failed:", err);
        }
      }
    },
    [screenToFlowPosition, addNodeOp, templates],
  );

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault();
    event.dataTransfer.dropEffect = "move";
  }, []);

  const onNodeDragStart = useCallback((_event: ReactMouseEvent, node: AppNode) => {
    const nodeData = {
      type: "flow-node",
      id: node.id,
      label: node.data.label || node.id,
      typeId: (node.data as any).typeId
    };
    
    if (_event.nativeEvent instanceof MouseEvent && (_event.nativeEvent as any).dataTransfer) {
       (_event.nativeEvent as any).dataTransfer.setData("application/flowcraft-node", JSON.stringify(nodeData));
       (_event.nativeEvent as any).dataTransfer.effectAllowed = "copyMove";
    }
  }, []);

  return (
    <div
      className="flex w-screen h-screen overflow-hidden bg-background text-foreground"
      onDrop={onDrop}
      onDragOver={onDragOver}
    >
      {/* Main Flow Area */}
      <div 
        className={cn(
          "flex-1 relative h-full min-w-0 fc-canvas-area",
          isChatFullscreen && "pointer-events-none opacity-50 grayscale-[0.5] transition-all duration-500"
        )}
        {...(isChatFullscreen ? { inert: "" } : {})}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChangeWithSnapping}
          onEdgesChange={onEdgesChange}
          onConnect={onConnect}
          onInit={onInit}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          onNodeDragStart={onNodeDragStart}
          onNodeDragStop={handleNodeDragStop}
          onConnectStart={onConnectStart}
          onConnectEnd={onConnectEnd}
          onNodeContextMenu={handleNodeContextMenu}
          onEdgeContextMenu={onEdgeContextMenu}
          onSelectionContextMenu={onSelectionContextMenu}
          onPaneContextMenu={onPaneContextMenu}
          onMoveEnd={handleMoveEnd}
          fitView
          colorMode={settings.theme}
          selectionMode={SelectionMode.Partial}
          panOnDrag={dragMode === "pan" ? [0, 1] : [1]}
          selectionOnDrag={dragMode === "select"}
          selectNodesOnDrag={false}
          snapToGrid={false}
          snapGrid={snapGrid}
          defaultEdgeOptions={defaultEdgeOptions}
        >
          <Background variant={BackgroundVariant.Dots} gap={15} size={1} />
          <Controls />
          <MiniMap
            style={{
              borderRadius: "8px",
              overflow: "hidden",
            }}
            maskColor="var(--xy-minimap-mask-background-color)"
          />
          <Notifications />
          {contextMenu && (
            <>
              {(contextMenu.nodeId || nodes.some((n) => n.selected)) && (
                <NodeContextMenu
                  x={contextMenu.x}
                  y={contextMenu.y}
                  nodeId={contextMenu.nodeId || ""}
                  onClose={closeContextMenuAndClear}
                  onDelete={() => {
                    const nodeId = contextMenu.nodeId;
                    if (nodeId) {
                      const node = nodes.find((n) => n.id === nodeId);
                      if (node?.selected) {
                        const selectedNodes = nodes.filter((n) => n.selected);
                        const selectedEdges = edges.filter((e) => e.selected);
                        selectedNodes.forEach((n) => {
                          deleteNode(n.id);
                        });
                        selectedEdges.forEach((e) => {
                          deleteEdge(e.id);
                        });
                      } else {
                        deleteNode(nodeId);
                      }
                    } else {
                      const selectedNodes = nodes.filter((n) => n.selected);
                      selectedNodes.forEach((n) => {
                        deleteNode(n.id);
                      });
                    }
                    closeContextMenuAndClear();
                  }}
                  onFocus={() => {}}
                  onOpenEditor={() => {
                    if (contextMenu.nodeId) {
                      setActiveEditorId(contextMenu.nodeId);
                    }
                    closeContextMenuAndClear();
                  }}
                  onCopy={copySelected}
                  onDuplicate={duplicateSelected}
                  onGroupSelected={
                    nodes.some((n) => n.selected) ? groupSelected : undefined
                  }
                  onLayoutGroup={
                    nodes.find((n) => n.id === contextMenu.nodeId)?.type ===
                    "groupNode"
                      ? autoLayout
                      : undefined
                  }
                  dynamicActions={availableActions.map((action) => ({
                    id: action.id,
                    name: action.label,
                    path: action.path,
                    onClick: () => {
                      handleExecuteAction(action);
                    },
                  }))}
                />
              )}
              {contextMenu.edgeId && !nodes.some((n) => n.selected) && (
                <EdgeContextMenu
                  x={contextMenu.x}
                  y={contextMenu.y}
                  edgeId={contextMenu.edgeId}
                  onClose={closeContextMenuAndClear}
                  onDelete={() => {
                    const edgeId = contextMenu.edgeId;
                    if (edgeId) {
                      const edge = edges.find((e) => e.id === edgeId);
                      if (edge?.selected) {
                        const selectedNodes = nodes.filter((n) => n.selected);
                        const selectedEdges = edges.filter((e) => e.selected);
                        selectedNodes.forEach((n) => {
                          deleteNode(n.id);
                        });
                        selectedEdges.forEach((e) => {
                          deleteEdge(e.id);
                        });
                      } else {
                        deleteEdge(edgeId);
                      }
                    }
                    closeContextMenuAndClear();
                  }}
                />
              )}
              {contextMenu.galleryItemUrl && (
                <GalleryItemContextMenu
                  x={contextMenu.x}
                  y={contextMenu.y}
                  url={contextMenu.galleryItemUrl}
                  onClose={closeContextMenuAndClear}
                  onExtract={(url: string) => {
                    console.log("Extracting", url);
                    closeContextMenuAndClear();
                  }}
                />
              )}
              {!contextMenu.nodeId &&
                !contextMenu.edgeId &&
                !contextMenu.galleryItemUrl &&
                !nodes.some((n) => n.selected) && (
                  <PaneContextMenu
                    x={contextMenu.x}
                    y={contextMenu.y}
                    templates={templates}
                    onAddNode={handleAddNode}
                    onAutoLayout={autoLayout}
                    onClose={closeContextMenuAndClear}
                    onPaste={paste}
                    onCopy={
                      nodes.some((n) => n.selected) ? copySelected : undefined
                    }
                    onDuplicate={
                      nodes.some((n) => n.selected)
                        ? duplicateSelected
                        : undefined
                    }
                    onGroupSelected={
                      nodes.some((n) => n.selected) ? groupSelected : undefined
                    }
                    onDeleteSelected={
                      nodes.some((n) => n.selected) ||
                      edges.some((e) => e.selected)
                        ? () => {
                            const selectedNodes = nodes.filter((n) => n.selected);
                            const selectedEdges = edges.filter((e) => e.selected);
                            selectedNodes.forEach((n) => {
                              deleteNode(n.id);
                            });
                            selectedEdges.forEach((e) => {
                              deleteEdge(e.id);
                            });
                            closeContextMenuAndClear();
                          }
                        : undefined
                    }
                  />
                )}
            </>
          )}
          <HelperLinesRenderer lines={helperLines} />
        </ReactFlow>

        {previewData &&
          nodes
            .filter((n) => n.id === previewData.nodeId)
            .map((node) => (
              <MediaPreview
                key={node.id}
                node={node}
                initialIndex={previewData.index}
                onClose={() => {
                  setPreviewData(null);
                }}
              />
            ))}

        {activeEditorId &&
          nodes
            .filter((n) => n.id === activeEditorId)
            .map((node) => (
              <EditorPlaceholder
                key={node.id}
                node={node}
                onClose={() => {
                  setActiveEditorId(null);
                }}
              />
            ))}
        <TaskHistoryDrawer />
        <SideToolbar connectionStatus={connectionStatus} />
        <SettingsModal />
        {pendingAction && (
          <ActionParamsModal
            action={pendingAction}
            onConfirm={(params) => {
              handleExecuteAction(pendingAction, params);
            }}
            onCancel={() => {
              setPendingAction(null);
            }}
          />
        )}
      </div>

      <div 
        className={cn(isChatFullscreen && "pointer-events-none")}
        {...(isChatFullscreen ? { inert: "" } : {})}
      >
        <Sidebar />
      </div>

      {/* Global Fullscreen Chat Overlay */}
      {isChatFullscreen && activeChatNodeId && (
        <div className="fixed inset-0 z-[10000] bg-background flex flex-col animate-in fade-in zoom-in-95 duration-200">
          <div className="shrink-0 p-4 border-b border-node-border flex justify-between items-center bg-muted/20">
            <div className="flex items-center gap-2">
              <Bot size={20} className="text-primary-color" />
              <h2 className="font-bold">Full Conversation Mode</h2>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm" onClick={() => setActiveChat(activeChatNodeId, "inline")}>
                <Minimize2 size={16} className="mr-2" /> Dock to Node
              </Button>
              <Button variant="ghost" size="icon" onClick={() => setActiveChat(null)}>
                <X size={20} />
              </Button>
            </div>
          </div>
          <div className="flex-1 overflow-hidden shadcn-lookup ai-theme-container">
            <ChatBot nodeId={activeChatNodeId} />
          </div>
        </div>
      )}
      
      <Toaster position="bottom-right" />
    </div>
  );
}

export default memo(App);
