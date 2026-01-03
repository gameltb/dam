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

  const { dragMode, settings } = useUiStore(
    useShallow((state) => ({
      dragMode: state.dragMode,
      settings: state.settings,
    })),
  );

  const flowSocket = useFlowSocket();
  const { cancelTask, executeTask, streamAction, templates, updateViewport } =
    flowSocket;
  const { theme } = useTheme();
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
      // Trigger discovery for selection
      const selectedNodeIds = nodes.filter((n) => n.selected).map((n) => n.id);
      if (selectedNodeIds.length > 0) {
        void socketClient.send({
          payload: {
            case: "actionDiscovery",
            value: create(ActionDiscoveryRequestSchema, {
              nodeId: "", // Generic selection
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
      const defaultData = template.defaultState
        ? fromProtoNodeData(template.defaultState)
        : {};
      const flowPosition = screenToFlowPosition({
        x: contextMenu.x,
        y: contextMenu.y,
      });
      addNodeOp(
        "dynamic",
        defaultData,
        flowPosition,
        template.templateId,
        template.defaultWidth,
        template.defaultHeight,
      );
    },
    [addNodeOp, contextMenu, screenToFlowPosition],
  );

  const contextNodeId = contextMenu?.nodeId;
  const handleExecuteAction = useCallback(
    (action: ActionTemplate, params: Record<string, any> = {}) => {
      // If action has schema and no params provided yet, show modal
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

  return (
    <div style={{ width: "100vw", height: "100vh" }} className={theme}>
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChangeWithSnapping}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        nodeTypes={nodeTypes}
        edgeTypes={edgeTypes}
        onNodeDragStop={handleNodeDragStop}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        onNodeContextMenu={handleNodeContextMenu}
        onEdgeContextMenu={onEdgeContextMenu}
        onSelectionContextMenu={onSelectionContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onMoveEnd={handleMoveEnd}
        fitView
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
            backgroundColor: "var(--panel-bg)",
            borderRadius: "8px",
            overflow: "hidden",
            border: "1px solid var(--node-border)",
          }}
          maskColor="rgba(0, 0, 0, 0.1)"
        />
        <Notifications />
        {contextMenu && (
          <>
            {/* Handle Node/Selection Context Menu */}
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
                    // Just selection
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
                      // Delete all selected nodes and edges
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
                  /* extract logic */
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
      <Toaster position="bottom-right" />
    </div>
  );
}

export default memo(App);
