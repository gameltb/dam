import { useState, memo, useCallback, useEffect } from "react";
import { useTheme } from "./hooks/useTheme";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
} from "@xyflow/react";
import { useFlowStore } from "./store/flowStore";
import { useUiStore } from "./store/uiStore";
import { useTaskStore } from "./store/taskStore";
import "@xyflow/react/dist/style.css";
import { useMockSocket } from "./hooks/useMockSocket";
import { ContextMenu } from "./components/ContextMenu";
import { StatusPanel } from "./components/StatusPanel";
import { EditUrlModal } from "./components/EditUrlModal";
import { MutationSource, type NodeTemplate } from "./types";
import { type ActionTemplate } from "./generated/action_pb";
import { ActionExecutionRequestSchema } from "./generated/action_pb";
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
import { socketClient } from "./utils/SocketClient";
import { SideToolbar } from "./components/SideToolbar";
import { SettingsModal } from "./components/SettingsModal";

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

  const { dragMode } = useUiStore(
    useShallow((state) => ({
      dragMode: state.dragMode,
    })),
  );

  const mockSocket = useMockSocket();
  const { cancelTask, executeTask, streamAction, templates, updateViewport } =
    mockSocket;
  const { theme } = useTheme();
  const { helperLines, setHelperLines, calculateLines } = useHelperLines();

  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws (mocked)");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [availableActions, setAvailableActions] = useState<ActionTemplate[]>(
    [],
  );
  const [previewData, setPreviewData] = useState<{
    nodeId: string;
    index: number;
  } | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);

  useEffect(() => {
    const onActions = (actions: unknown) => {
      setAvailableActions(actions as ActionTemplate[]);
    };
    socketClient.on("actions", onActions);
    return () => {
      socketClient.off("actions", onActions);
    };
  }, []);

  const {
    onNodeContextMenu: onNodeContextMenuHook,
    onPaneContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu,
    closeContextMenuAndClear,
    contextMenu,
    setContextMenu,
    onNodeDragStop: contextMenuDragStop,
  } = useContextMenu();

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
  } = useGraphOperations({
    clientVersion,
  });

  const handleAddNode = useCallback(
    (template: NodeTemplate) => {
      if (!contextMenu) return;
      addNodeOp(
        "dynamic",
        template.defaultData,
        { x: contextMenu.x, y: contextMenu.y },
        template.id,
        template.defaultWidth,
        template.defaultHeight,
      );
    },
    [addNodeOp, contextMenu],
  );

  const handleExecuteAction = (action: ActionTemplate) => {
    const nodeId = contextMenu?.nodeId;
    if (!nodeId) return;
    const selectedIds = nodes.filter((n) => n.selected).map((n) => n.id);

    const taskId = crypto.randomUUID();
    useTaskStore.getState().registerTask({
      taskId,
      label: action.label,
      source: MutationSource.REMOTE_TASK,
    });

    void socketClient.send({
      payload: {
        case: "actionExecute",
        value: create(ActionExecutionRequestSchema, {
          actionId: action.id,
          sourceNodeId: nodeId,
          contextNodeIds: selectedIds,
          paramsJson: JSON.stringify({ taskId }),
        }),
      },
    });

    closeContextMenuAndClear();
  };

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
        <StatusPanel
          url={wsUrl}
          status="Connected"
          onClick={() => {
            setIsModalOpen(true);
          }}
        />
        <Notifications />
        {contextMenu && (
          <ContextMenu
            {...contextMenu}
            dynamicActions={availableActions.map((action) => ({
              id: action.id,
              name: action.label,
              onClick: () => {
                handleExecuteAction(action);
              },
            }))}
            onClose={closeContextMenuAndClear}
            templates={templates}
            onAddNode={handleAddNode}
            onAutoLayout={autoLayout}
            onGroupSelected={groupSelected}
          />
        )}
        <HelperLinesRenderer lines={helperLines} />
      </ReactFlow>

      {isModalOpen && (
        <EditUrlModal
          currentUrl={wsUrl}
          onClose={() => {
            setIsModalOpen(false);
          }}
          onSave={(newUrl) => {
            setWsUrl(newUrl);
          }}
        />
      )}

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
      <SideToolbar />
      <SettingsModal />
      <Toaster position="bottom-right" />
    </div>
  );
}

export default memo(App);
