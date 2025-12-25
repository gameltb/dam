import { useEffect, useMemo, useState, memo, useRef, useCallback } from "react";
import { useTheme } from "./hooks/useTheme";
import { flowcraft } from "./generated/flowcraft";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  type NodeTypes,
  type ReactFlowInstance,
  type Edge,
  type OnConnectStartParams,
} from "@xyflow/react";
import { useFlowStore, useTemporalStore } from "./store/flowStore";
import { useTaskStore } from "./store/taskStore";
import "@xyflow/react/dist/style.css";
import { useMockSocket } from "./hooks/useMockSocket";
import GroupNode from "./components/GroupNode";
import { DynamicNode } from "./components/DynamicNode";
import ProcessingNode from "./components/ProcessingNode";
import { ContextMenu } from "./components/ContextMenu";
import { StatusPanel } from "./components/StatusPanel";
import { EditUrlModal } from "./components/EditUrlModal";
import {
  type AppNode,
  type NodeTemplate,
  MediaType,
  RenderMode,
} from "./types";
import { Toaster } from "react-hot-toast";
import { Notifications } from "./components/Notifications";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import SystemEdge from "./components/edges/SystemEdge";
import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";
import { MediaPreview } from "./components/media/MediaPreview";
import { EditorPlaceholder } from "./components/media/EditorPlaceholder";

import { useShallow } from "zustand/react/shallow";
import { v4 as uuidv4 } from "uuid";
import { SelectionMode } from "@xyflow/react";

function App() {
  const {
    nodes,
    edges,
    onNodesChange,
    onEdgesChange,
    onConnect,
    addNode: addNodeToStore,
    version: clientVersion,
    updateNodeData,
    lastNodeEvent,
    setConnectionStartHandle,
  } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addNode: state.addNode,
      version: state.version,
      updateNodeData: state.updateNodeData,
      lastNodeEvent: state.lastNodeEvent,
      setConnectionStartHandle: state.setConnectionStartHandle,
    })),
  );

  const { undo, redo } = useTemporalStore((state) => ({
    undo: state.undo,
    redo: state.redo,
  }));

  const { templates, discoverActions, executeAction, executeTask, cancelTask } =
    useMockSocket();
  const { theme, toggleTheme } = useTheme();

  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws (mocked)");
  const [isModalOpen, setIsModalOpen] = useState(false);

  const [availableActions, setAvailableActions] = useState<
    flowcraft.v1.IActionTemplate[]
  >([]);

  const [previewData, setPreviewData] = useState<{
    nodeId: string;
    index: number;
  } | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);

  const lastProcessedEventTimeRef = useRef<number>(0);
  const rfInstanceRef = useRef<ReactFlowInstance<AppNode, Edge> | null>(null);

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

  const {
    addNode,
    deleteNode,
    deleteEdge,
    autoLayout,
    copySelected,
    paste,
    duplicateSelected,
  } = useGraphOperations({ clientVersion });

  const onConnectStart = useCallback(
    (_: unknown, { nodeId, handleId, handleType }: OnConnectStartParams) => {
      // Defer to avoid render warning
      setTimeout(() => {
        if (nodeId && handleId && handleType) {
          setConnectionStartHandle({ nodeId, handleId, type: handleType });
        }
      }, 0);
    },
    [setConnectionStartHandle],
  );

  const onConnectEnd = useCallback(() => {
    setTimeout(() => {
      setConnectionStartHandle(null);
    }, 0);
  }, [setConnectionStartHandle]);

  const handleNodeContextMenu = useCallback(
    async (event: React.MouseEvent, node: AppNode) => {
      onNodeContextMenu(event, node);
      const actions = await discoverActions(node.id);
      setAvailableActions(actions);
    },
    [onNodeContextMenu, discoverActions],
  );

  const handleExecuteAction = async (action: flowcraft.v1.IActionTemplate) => {
    if (!contextMenu?.nodeId) return;
    const nodeId = contextMenu.nodeId;

    const result = await executeAction(action.id!, nodeId);

    if (
      result &&
      result.type === "task" &&
      (result as unknown as { taskId: string }).taskId
    ) {
      const taskId = (result as unknown as { taskId: string }).taskId;
      const parentNode = nodes.find((n) => n.id === nodeId);
      const position = parentNode
        ? { x: parentNode.position.x + 300, y: parentNode.position.y }
        : { x: 0, y: 0 };

      const placeholderNode: AppNode = {
        id: `task-${taskId}`,
        type: "processing",
        position,
        data: {
          label: `Running ${action.label}...`,
          taskId,
          onCancel: (tid: string) => cancelTask(tid),
        },
      };
      addNodeToStore(placeholderNode);
      useTaskStore.getState().registerTask(taskId);
      executeTask(taskId, action.id!, { sourceNodeId: nodeId });
    }
    closeContextMenu();
  };

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      if (
        document.activeElement?.tagName === "INPUT" ||
        document.activeElement?.tagName === "TEXTAREA"
      )
        return;

      if ((event.ctrlKey || event.metaKey) && event.key === "z") {
        if (event.shiftKey) redo();
        else undo();
      } else if ((event.ctrlKey || event.metaKey) && event.key === "y") {
        redo();
      } else if ((event.ctrlKey || event.metaKey) && event.key === "c") {
        event.preventDefault();
        copySelected();
      } else if ((event.ctrlKey || event.metaKey) && event.key === "v") {
        event.preventDefault();
        paste();
      } else if ((event.ctrlKey || event.metaKey) && event.key === "d") {
        event.preventDefault();
        duplicateSelected();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [undo, redo, copySelected, paste, duplicateSelected]);

  useEffect(() => {
    if (
      !lastNodeEvent ||
      lastNodeEvent.timestamp <= lastProcessedEventTimeRef.current
    )
      return;
    lastProcessedEventTimeRef.current = lastNodeEvent.timestamp;

    if (lastNodeEvent.type === "gallery-context") {
      const payload = lastNodeEvent.payload as {
        x: number;
        y: number;
        nodeId: string;
        url: string;
        mediaType: MediaType;
      };
      setContextMenu({
        x: payload.x,
        y: payload.y,
        nodeId: payload.nodeId,
        galleryItemUrl: payload.url,
        galleryItemType: payload.mediaType,
      });
    } else if (lastNodeEvent.type === "open-preview") {
      const payload = lastNodeEvent.payload as {
        nodeId: string;
        index: number;
      };
      setTimeout(() => setPreviewData(payload), 0);
    } else if (lastNodeEvent.type === "open-editor") {
      const payload = lastNodeEvent.payload as { nodeId: string };
      setTimeout(() => setActiveEditorId(payload.nodeId), 0);
    }
  }, [lastNodeEvent, setContextMenu]);

  const memoizedNodeTypes: NodeTypes = useMemo(
    () => ({
      groupNode: GroupNode,
      dynamic: DynamicNode,
      processing: ProcessingNode,
    }),
    [],
  );

  const memoizedEdgeTypes = useMemo(
    () => ({
      system: SystemEdge,
      default: BaseFlowEdge,
    }),
    [],
  );

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
        onChange: (id: string, data: Partial<AppNode["data"]>) =>
          updateNodeData(id, data),
      },
      position,
    );
  };

  const handleCreateFromGallery = (url: string) => {
    if (!rfInstanceRef.current || !contextMenu) return;
    const position = rfInstanceRef.current.screenToFlowPosition({
      x: contextMenu.x,
      y: contextMenu.y,
    });
    const mediaType = contextMenu.galleryItemType || MediaType.MEDIA_IMAGE;
    const newNodeId = uuidv4();
    const newNode: AppNode = {
      id: newNodeId,
      type: "dynamic",
      position,
      data: {
        label: "Extracted Media",
        modes: [RenderMode.MODE_MEDIA],
        activeMode: RenderMode.MODE_MEDIA,
        media: { type: mediaType, url, aspectRatio: 1, galleryUrls: [] },
        onChange: (id: string, data: Partial<AppNode["data"]>) =>
          updateNodeData(id, data),
      },
    } as AppNode;
    addNodeToStore(newNode);
    closeContextMenu();
  };

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Toaster />
      <Notifications />
      <ReactFlow<AppNode, Edge>
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChange}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        onNodeContextMenu={handleNodeContextMenu}
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
        colorMode={theme}
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
          onCopy={copySelected}
          onPaste={() => {
            paste(
              rfInstanceRef.current?.screenToFlowPosition({
                x: contextMenu.x,
                y: contextMenu.y,
              }),
            );
            closeContextMenu();
          }}
          onDuplicate={duplicateSelected}
          dynamicActions={availableActions.map((action) => ({
            id: action.id!,
            name: action.label!,
            onClick: () => handleExecuteAction(action),
          }))}
          onToggleTheme={toggleTheme}
          templates={templates}
          onAddNode={handleAddNodeFromTemplate}
          onAutoLayout={() => {
            autoLayout();
            closeContextMenu();
          }}
          onGalleryAction={handleCreateFromGallery}
          galleryItemUrl={contextMenu.galleryItemUrl}
          isPaneMenu={!contextMenu.nodeId && !contextMenu.edgeId}
        />
      )}
      <StatusPanel
        status={`Connected (Mock WS) - ${theme}`}
        url={wsUrl}
        onClick={() => setIsModalOpen(true)}
      />
      {isModalOpen && (
        <EditUrlModal
          currentUrl={wsUrl}
          onClose={() => setIsModalOpen(false)}
          onSave={(newUrl) => setWsUrl(newUrl)}
        />
      )}
      {previewData && (
        <MediaPreview
          node={nodes.find((n) => n.id === previewData.nodeId)!}
          initialIndex={previewData.index}
          onClose={() => setPreviewData(null)}
        />
      )}
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
