import { useEffect, useMemo, useState, memo, useRef, useCallback } from "react";
import { useTheme } from "./hooks/useTheme";
import { flowcraft_proto } from "./generated/flowcraft_proto";
import {
  ReactFlow,
  MiniMap,
  Controls,
  Background,
  BackgroundVariant,
  type NodeTypes,
  type ReactFlowInstance,
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
  MutationSource,
} from "./types";
import { Toaster } from "react-hot-toast";
import { Notifications } from "./components/Notifications";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import SystemEdge from "./components/edges/SystemEdge";
import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";
import { MediaPreview } from "./components/media/MediaPreview";
import { EditorPlaceholder } from "./components/media/EditorPlaceholder";
import { TaskHistoryDrawer } from "./components/TaskHistoryDrawer";
import { socketClient } from "./utils/SocketClient";

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

  const mockSocket = useMockSocket();
  const { templates, executeTask, cancelTask, streamAction } = mockSocket;
  const { theme, toggleTheme } = useTheme();

  const [wsUrl, setWsUrl] = useState("ws://127.0.0.1:8000/ws (mocked)");
  const [isModalOpen, setIsModalOpen] = useState(false);

  const [availableActions, setAvailableActions] = useState<
    flowcraft_proto.v1.IActionTemplate[]
  >([]);

  const [previewData, setPreviewData] = useState<{
    nodeId: string;
    index: number;
  } | null>(null);
  const [activeEditorId, setActiveEditorId] = useState<string | null>(null);

  const lastProcessedEventTimeRef = useRef<number>(0);
  const rfInstanceRef = useRef<ReactFlowInstance<AppNode> | null>(null);

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

  const closeContextMenuAndClear = useCallback(() => {
    closeContextMenu();
    setAvailableActions([]);
  }, [closeContextMenu]);

  const {
    addNode,
    deleteNode,
    deleteEdge,
    autoLayout,
    groupSelected,
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
    (event: React.MouseEvent, node: AppNode) => {
      onNodeContextMenu(event, node);
      // Discover actions via Unified Protocol
      void socketClient
        .send({
          actionDiscovery: {
            nodeId: node.id,
            selectedNodeIds: nodes.filter((n) => n.selected).map((n) => n.id),
          },
        })
        .catch((e: unknown) => {
          console.error("Failed to send action discovery", e);
        });
    },
    [onNodeContextMenu, nodes],
  );

  const handleExecuteAction = (action: flowcraft_proto.v1.IActionTemplate) => {
    const nodeId = contextMenu?.nodeId ?? "";
    const selectedIds = nodes.filter((n) => n.selected).map((n) => n.id);

    // Track as a task even if backend hasn't replied yet (Optimistic UI)
    const taskId = uuidv4();
    useTaskStore.getState().registerTask({
      taskId,

      label: action.label ?? "Backend Action",
      source: MutationSource.REMOTE_TASK,
    });

    void socketClient
      .send({
        actionExecute: {
          actionId: action.id ?? "",
          sourceNodeId: nodeId,
          contextNodeIds: selectedIds,
          paramsJson: JSON.stringify({ taskId }),
        },
      })
      .catch((e: unknown) => {
        console.error("Failed to execute action", e);
      });

    closeContextMenuAndClear();
  };

  useEffect(() => {
    const onActions = (actions: flowcraft_proto.v1.IActionTemplate[]) => {
      setAvailableActions(actions);
    };
    socketClient.on("actions", onActions);
    return () => {
      socketClient.off("actions", onActions);
    };
  }, []);

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
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
    };
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
      setTimeout(() => {
        setPreviewData(payload);
      }, 0);
    } else if (lastNodeEvent.type === "open-editor") {
      const payload = lastNodeEvent.payload as { nodeId: string };
      setTimeout(() => {
        setActiveEditorId(payload.nodeId);
      }, 0);
    } else if (lastNodeEvent.type === "widget-click") {
      const { nodeId, widgetId } = lastNodeEvent.payload as {
        nodeId: string;
        widgetId: string;
      };
      const node = nodes.find((n) => n.id === nodeId);
      if (node && node.type === "dynamic" && node.data.widgets) {
        const widget = node.data.widgets.find((w) => w.id === widgetId);
        if (widget && typeof widget.value === "string") {
          const val = widget.value;
          if (val.startsWith("stream-to:")) {
            const targetWidgetId = val.split(":")[1] ?? "";
            let currentBuffer = "";
            streamAction(nodeId, widgetId, (chunk) => {
              currentBuffer += chunk;
              const store = useFlowStore.getState();
              const currentNode = store.nodes.find((n) => n.id === nodeId);
              if (
                currentNode &&
                currentNode.type === "dynamic" &&
                currentNode.data.widgets
              ) {
                const updatedWidgets = currentNode.data.widgets.map((w) =>
                  w.id === targetWidgetId ? { ...w, value: currentBuffer } : w,
                );
                store.updateNodeData(nodeId, { widgets: updatedWidgets });
              }
            });
          } else if (val.startsWith("task:")) {
            const taskType = val.split(":")[1] ?? "";
            const taskId = uuidv4();
            const position = {
              x: node.position.x + 300,
              y: node.position.y,
            };
            const placeholderNode: AppNode = {
              id: `task-${taskId}`,
              type: "processing",
              position,
              data: {
                label: `Running ${taskType ?? ""}...`,
                taskId,
                onCancel: (tid: string) => {
                  cancelTask(tid);
                },
              },
            } as AppNode;
            addNodeToStore(placeholderNode);
            useTaskStore.getState().registerTask({
              taskId,
              label: `Running ${taskType ?? ""}...`,
              source: MutationSource.REMOTE_TASK,
            });
            executeTask(taskId, taskType, { sourceNodeId: nodeId });
          }
        }
      }
    }
  }, [
    lastNodeEvent,
    setContextMenu,
    nodes,
    streamAction,
    addNodeToStore,
    cancelTask,
    executeTask,
  ]);

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
        onChange: (id: string, data: Partial<AppNode["data"]>) => {
          updateNodeData(id, data);
        },
      },
      position,
      template.id,
    );
  };

  const handleCreateFromGallery = (url: string) => {
    if (!rfInstanceRef.current || !contextMenu) return;
    const position = rfInstanceRef.current.screenToFlowPosition({
      x: contextMenu.x,
      y: contextMenu.y,
    });
    const mediaType = contextMenu.galleryItemType ?? MediaType.MEDIA_IMAGE;
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
        onChange: (id: string, data: Partial<AppNode["data"]>) => {
          updateNodeData(id, data);
        },
      },
    } as AppNode;
    addNodeToStore(newNode);
    closeContextMenuAndClear();
  };

  return (
    <div style={{ width: "100vw", height: "100vh" }}>
      <Toaster />
      <Notifications />
      <ReactFlow<AppNode>
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
          onClose={closeContextMenuAndClear}
          onDelete={
            contextMenu.nodeId
              ? () => {
                  deleteNode(contextMenu.nodeId!);
                  closeContextMenuAndClear();
                }
              : undefined
          }
          onDeleteEdge={
            contextMenu.edgeId
              ? () => {
                  deleteEdge(contextMenu.edgeId!);
                  closeContextMenuAndClear();
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
            closeContextMenuAndClear();
          }}
          onDuplicate={duplicateSelected}
          dynamicActions={availableActions.map((action) => ({
            id: action.id ?? "",
            name: action.label ?? "Action",
            onClick: () => {
              handleExecuteAction(action);
            },
          }))}
          onToggleTheme={toggleTheme}
          templates={templates}
          onAddNode={handleAddNodeFromTemplate}
          onAutoLayout={() => {
            autoLayout();
            closeContextMenuAndClear();
          }}
          onGroupSelected={
            nodes.filter((n) => n.selected).length >= 2
              ? () => {
                  groupSelected();
                  closeContextMenuAndClear();
                }
              : undefined
          }
          onGalleryAction={handleCreateFromGallery}
          galleryItemUrl={contextMenu.galleryItemUrl}
          isPaneMenu={!contextMenu.nodeId && !contextMenu.edgeId}
        />
      )}
      <StatusPanel
        status={`Connected (Mock WS) - ${theme}`}
        url={wsUrl}
        onClick={() => {
          setIsModalOpen(true);
        }}
      />
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
      {previewData && nodes.find((n) => n.id === previewData.nodeId) && (
        <MediaPreview
          node={nodes.find((n) => n.id === previewData.nodeId) ?? nodes[0]}
          initialIndex={previewData.index}
          onClose={() => {
            setPreviewData(null);
          }}
        />
      )}
      {activeEditorId && nodes.find((n) => n.id === activeEditorId) && (
        <EditorPlaceholder
          node={nodes.find((n) => n.id === activeEditorId) ?? nodes[0]}
          onClose={() => {
            setActiveEditorId(null);
          }}
        />
      )}
      <TaskHistoryDrawer />
    </div>
  );
}

const MemoizedApp = memo(App);
export default MemoizedApp;
