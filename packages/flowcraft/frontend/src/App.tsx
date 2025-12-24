import { useEffect, useMemo, useState, memo, useRef } from "react";
import { useTheme } from "./hooks/useTheme";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  type NodeTypes,
  type ReactFlowInstance,
  type Edge,
} from "@xyflow/react";
import { useFlowStore, useTemporalStore } from "./store/flowStore";
import { useNotificationStore } from "./store/notificationStore";
import toast from "react-hot-toast";
import "@xyflow/react/dist/style.css";
import { useMockSocket, type WebSocketMessage } from "./hooks/useMockSocket";
import GroupNode from "./components/GroupNode";
import { DynamicNode } from "./components/DynamicNode";
import { ContextMenu } from "./components/ContextMenu";
import { StatusPanel } from "./components/StatusPanel";
import { EditUrlModal } from "./components/EditUrlModal";
import { type AppNode, type GraphState, type NodeTemplate } from "./types";
import { Toaster } from "react-hot-toast";
import { Notifications } from "./components/Notifications";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import SystemEdge from "./components/edges/SystemEdge";
import { MediaPreview } from "./components/media/MediaPreview";
import { EditorPlaceholder } from "./components/media/EditorPlaceholder";

import { useShallow } from "zustand/react/shallow";
import { SelectionMode } from "@xyflow/react";

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    setNodes,
    setEdges,
    addNode: addNodeToStore,
    version: clientVersion,
    setGraph,
    updateNodeData,
    lastNodeEvent,
  } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      setNodes: state.setNodes,
      setEdges: state.setEdges,
      addNode: state.addNode,
      version: state.version,
      setGraph: state.setGraph,
      updateNodeData: state.updateNodeData,
      lastNodeEvent: state.lastNodeEvent,
    })),
  );

  const { undo, redo } = useTemporalStore((state) => ({
    undo: state.undo,
    redo: state.redo,
  }));

  const { addNotification } = useNotificationStore();
  const { sendJsonMessage, lastJsonMessage, templates, mockServerState } =
    useMockSocket();
  const { toggleTheme } = useTheme();

  const [isOutOfSync, setIsOutOfSync] = useState(false);
  const [serverGraphState, setServerGraphState] = useState<GraphState | null>(
    null,
  );
  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws (mocked)");
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Modal states
  const [previewData, setPreviewData] = useState<{
    nodeId: string;
    index: number;
  } | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);

  const lastProcessedMessageRef = useRef<WebSocketMessage | null>(null);
  const lastProcessedEventTimeRef = useRef<number>(0);
  const rfInstanceRef = useRef<ReactFlowInstance<AppNode, Edge> | null>(null);
  const isInitialLoadRef = useRef(true);

  const connectionStatus = "Connected (Mock)";

  // Keyboard Shortcuts for Undo/Redo
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if ((event.ctrlKey || event.metaKey) && event.key === "z") {
        if (event.shiftKey) {
          redo();
        } else {
          undo();
        }
      } else if ((event.ctrlKey || event.metaKey) && event.key === "y") {
        redo();
      }
    };

    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo]);

  // Custom Hooks
  const {
    contextMenu,
    onPaneContextMenu,
    onNodeContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu,
    onPaneClick,
    closeContextMenu,
    setContextMenu,
  } = useContextMenu();

  useEffect(() => {
    if (
      !lastNodeEvent ||
      lastNodeEvent.timestamp <= lastProcessedEventTimeRef.current
    )
      return;

    lastProcessedEventTimeRef.current = lastNodeEvent.timestamp;

    if (lastNodeEvent.type === "gallery-context") {
      const { x, y, nodeId, url } = lastNodeEvent.payload as {
        x: number;
        y: number;
        nodeId: string;
        url: string;
      };
      setContextMenu({ x, y, nodeId, galleryItemUrl: url });
    } else if (lastNodeEvent.type === "open-preview") {
      setTimeout(
        () =>
          setPreviewData(
            lastNodeEvent.payload as { nodeId: string; index: number },
          ),
        0,
      );
    } else if (lastNodeEvent.type === "open-editor") {
      setTimeout(
        () =>
          setActiveEditorId(
            (lastNodeEvent.payload as { nodeId: string }).nodeId,
          ),
        0,
      );
    }
  }, [lastNodeEvent, setContextMenu]);

  const {
    addNode,
    deleteNode,
    deleteEdge,
    focusNode,
    exitFocusView,
    autoLayout,
    incrementalLayout,
    groupSelectedNodes,
    layoutGroup,
    isFocusView,
  } = useGraphOperations({ clientVersion, sendJsonMessage });

  const memoizedNodeTypes: NodeTypes = useMemo(
    () => ({
      groupNode: GroupNode,
      dynamic: DynamicNode,
    }),
    [],
  );

  const memoizedEdgeTypes = useMemo(
    () => ({
      system: SystemEdge,
    }),
    [],
  );

  const handleManualSync = () => {
    if (serverGraphState) {
      setGraph(serverGraphState.graph, serverGraphState.version);
      if (serverGraphState.graph.viewport && rfInstanceRef.current) {
        rfInstanceRef.current.setViewport(serverGraphState.graph.viewport);
      }
      setIsOutOfSync(false);
      setServerGraphState(null);
      const message = "Graph successfully synced with the server.";
      toast.success(message);
      addNotification({ message, type: "success" });
    }
  };

  const handleAddNodeFromTemplate = (template: NodeTemplate) => {
    if (!rfInstanceRef.current || !contextMenu) return;

    const position = rfInstanceRef.current.screenToFlowPosition({
      x: contextMenu.x,
      y: contextMenu.y,
    });

    addNode(
      "dynamic",
      {
        ...template.defaultData,
        onChange: (id, data) => updateNodeData(id, data),
        onWidgetClick: (nodeId, widgetId) => {
          useFlowStore
            .getState()
            .dispatchNodeEvent("widget-click", { nodeId, widgetId });
        },
        onGalleryItemContext: (nodeId, url, x, y) => {
          useFlowStore
            .getState()
            .dispatchNodeEvent("gallery-context", { nodeId, url, x, y });
        },
      } as AppNode["data"],
      position,
    );
  };

  const handleCreateFromGallery = (url: string) => {
    if (!rfInstanceRef.current || !contextMenu) return;

    const position = rfInstanceRef.current.screenToFlowPosition({
      x: contextMenu.x,
      y: contextMenu.y,
    });

    addNode(
      "dynamic",
      {
        label: "Extracted Image",
        modes: ["media"],
        activeMode: "media",
        media: { type: "image", url, aspectRatio: 1 },
        inputType: "any",
        outputType: "image",
        onChange: (id, data) => updateNodeData(id, data),
      } as AppNode["data"],
      position,
    );
  };

  useEffect(() => {
    if (!lastJsonMessage || lastJsonMessage === lastProcessedMessageRef.current)
      return;

    lastProcessedMessageRef.current = lastJsonMessage;

    if (lastJsonMessage.type === "sync_graph") {
      const { graph, version } = lastJsonMessage.payload;
      if (lastJsonMessage.error === "version_mismatch") {
        setIsOutOfSync(true); // eslint-disable-line react-hooks/set-state-in-effect
        setServerGraphState({ graph, version });
        const message =
          "Your graph is out of sync. Click the status panel to update.";
        toast.error(message);
        addNotification({ message, type: "error" });
      } else {
        // Successful sync or initial load
        if (isInitialLoadRef.current) {
          setGraph(graph, version);
          if (graph.viewport && rfInstanceRef.current) {
            rfInstanceRef.current.setViewport(graph.viewport);
          }
          isInitialLoadRef.current = false;
        } else if (version > clientVersion) {
          // Background sync: update state only if server has a NEWER version
          setGraph(graph, version);
        }
      }
    } else if (lastJsonMessage.type === "apply_changes") {
      const { add = [], addEdges = [] } = lastJsonMessage.payload;

      const updatedNodes = [...nodes, ...add];
      const updatedEdges = [...edges, ...addEdges];
      const newNodeIds = add.map((n: AppNode) => n.id);

      // Update store with new elements first
      setNodes(updatedNodes);
      setEdges(updatedEdges);

      // Perform incremental layout on the new nodes only
      setTimeout(
        () => incrementalLayout(updatedNodes, updatedEdges, newNodeIds),
        0,
      );
    }
  }, [
    lastJsonMessage,
    nodes,
    edges,
    setNodes,
    setEdges,
    addNodeToStore,
    mockServerState,
    addNotification,
    setGraph,
    autoLayout,
    incrementalLayout,
    clientVersion,
  ]);

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Toaster />
      <Notifications />
      {isFocusView && (
        <div style={{ position: "absolute", top: 10, left: 10, zIndex: 4 }}>
          <button onClick={exitFocusView}>Back to Global View</button>
          <button onClick={() => undo()}>Undo</button>
          <button onClick={() => redo()}>Redo</button>
        </div>
      )}
      <ReactFlow<AppNode, Edge>
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onNodeContextMenu={onNodeContextMenu}
        onEdgeContextMenu={onEdgeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        onSelectionContextMenu={onSelectionContextMenu}
        onPaneClick={onPaneClick}
        nodeTypes={memoizedNodeTypes}
        edgeTypes={memoizedEdgeTypes}
        onInit={(instance) => {
          rfInstanceRef.current = instance;
        }}
        selectionOnDrag
        panOnDrag={[1]}
        selectionMode={SelectionMode.Partial}
        multiSelectionKeyCode="Control"
        proOptions={{ hideAttribution: true }}
      >
        <Controls />
        <MiniMap />
        <Background variant={BackgroundVariant.Dots} gap={12} size={1} />
      </ReactFlow>
      {contextMenu && (
        <ContextMenu
          x={contextMenu.x}
          y={contextMenu.y}
          onClose={closeContextMenu}
          onDelete={
            contextMenu.nodeId
              ? () => {
                  deleteNode(contextMenu.nodeId!);
                  closeContextMenu();
                }
              : undefined
          }
          onDeleteEdge={
            contextMenu.edgeId
              ? () => {
                  deleteEdge(contextMenu.edgeId!);
                  closeContextMenu();
                }
              : undefined
          }
          onFocus={
            contextMenu.nodeId
              ? () => {
                  focusNode(contextMenu.nodeId!);
                  closeContextMenu();
                }
              : undefined
          }
          onOpenEditor={
            contextMenu.nodeId
              ? () => {
                  useFlowStore.getState().dispatchNodeEvent("open-editor", {
                    nodeId: contextMenu.nodeId,
                  });
                  closeContextMenu();
                }
              : undefined
          }
          dynamicActions={
            contextMenu.nodeId &&
            nodes.find((n) => n.id === contextMenu.nodeId)?.type === "dynamic"
              ? mockServerState.availableActions.text.map((action) => ({
                  ...action,
                  onClick: async () => {
                    const viewport = rfInstanceRef.current?.getViewport();
                    await sendJsonMessage({
                      type: "sync_graph",
                      payload: {
                        version: clientVersion,
                        graph: { nodes, edges, viewport },
                      },
                    });
                    sendJsonMessage({
                      type: "execute_action",
                      payload: {
                        actionId: action.id,
                        nodeId: contextMenu.nodeId,
                      },
                    });
                  },
                }))
              : []
          }
          onToggleTheme={toggleTheme}
          templates={templates}
          onAddNode={handleAddNodeFromTemplate}
          onAutoLayout={() => {
            autoLayout();
            closeContextMenu();
          }}
          onGroupSelected={
            nodes.filter((n) => n.selected).length > 1
              ? () => {
                  groupSelectedNodes();
                  closeContextMenu();
                }
              : undefined
          }
          onLayoutGroup={
            contextMenu.nodeId &&
            nodes.find((n) => n.id === contextMenu.nodeId)?.type === "groupNode"
              ? () => {
                  layoutGroup(contextMenu.nodeId!);
                  closeContextMenu();
                }
              : undefined
          }
          onGalleryAction={handleCreateFromGallery}
          galleryItemUrl={contextMenu.galleryItemUrl}
          isPaneMenu={!contextMenu.nodeId && !contextMenu.edgeId}
        />
      )}
      <StatusPanel
        status={
          isOutOfSync ? "Out of Sync. Click to update." : connectionStatus
        }
        url={wsUrl}
        isOutOfSync={isOutOfSync}
        onClick={isOutOfSync ? handleManualSync : () => setIsModalOpen(true)}
      />
      {isModalOpen && (
        <EditUrlModal
          currentUrl={wsUrl}
          onClose={() => setIsModalOpen(false)}
          onSave={(newUrl) => setWsUrl(newUrl)}
        />
      )}

      {/* Preview Modal */}
      {previewData && (
        <MediaPreview
          node={nodes.find((n) => n.id === previewData.nodeId)!}
          initialIndex={previewData.index}
          onClose={() => setPreviewData(null)}
        />
      )}

      {/* Editor Placeholder Modal */}
      {activeEditorId && (
        <EditorPlaceholder
          node={nodes.find((n) => n.id === activeEditorId)!}
          onClose={() => setActiveEditorId(null)}
        />
      )}
    </div>
  );
}

export default memo(App);
