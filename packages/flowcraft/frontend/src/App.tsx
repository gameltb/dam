import {
  useEffect,
  useMemo,
  useState,
  memo,
  useCallback,
  useRef,
} from "react";
import { useTheme } from "./hooks/useTheme";
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
import { useUiStore } from "./store/uiStore";
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
  MediaType,
  MutationSource,
  type Port,
  type NodeTemplate,
} from "./types";
import { type ActionTemplate } from "./generated/action_pb";
import { Toaster } from "react-hot-toast";
import { Notifications } from "./components/Notifications";
import { useContextMenu } from "./hooks/useContextMenu";
import { useGraphOperations } from "./hooks/useGraphOperations";
import { useHelperLines } from "./hooks/useHelperLines";
import { HelperLinesRenderer } from "./components/HelperLinesRenderer";
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
    lastNodeEvent,
  } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      onNodesChange: state.onNodesChange,
      onEdgesChange: state.onEdgesChange,
      onConnect: state.onConnect,
      addNode: state.addNode,
      version: state.version,
      lastNodeEvent: state.lastNodeEvent,
    })),
  );

  const setConnectionStartHandle = useUiStore(
    (state) => state.setConnectionStartHandle,
  );

  const { undo, redo } = useTemporalStore(
    useShallow((state) => ({
      undo: state.undo,
      redo: state.redo,
    })),
  );

  const mockSocket = useMockSocket();
  const { cancelTask, executeTask, streamAction, templates } = mockSocket;
  const { theme, toggleTheme } = useTheme();
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

  const {
    onNodeContextMenu,
    onPaneContextMenu,
    closeContextMenuAndClear,
    contextMenu,
    setContextMenu,
    onNodeDragStop: contextMenuDragStop,
  } = useContextMenu();

  const handleNodeDragStop = useCallback(() => {
    setHelperLines({});
    contextMenuDragStop();
  }, [setHelperLines, contextMenuDragStop]);

  const onNodesChangeWithSnapping = useCallback(
    (changes: NodeChange[]) => {
      const snappedChanges = changes.map((change) => {
        if (change.type === "position" && change.position) {
          const node = nodes.find((n) => n.id === change.id);
          if (node) {
            const { snappedPosition } = calculateLines(
              node,
              nodes,
              true,
              change.position,
            );
            return {
              ...change,
              position: snappedPosition,
            };
          }
        }
        return change;
      });
      onNodesChange(snappedChanges);
    },
    [onNodesChange, nodes, calculateLines],
  );

  const {
    copySelected,
    paste,
    duplicateSelected,
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
      );
    },
    [addNodeOp, contextMenu],
  );

  const onConnectStart = useCallback(
    (_: unknown, { nodeId, handleId, handleType }: OnConnectStartParams) => {
      setTimeout(() => {
        const store = useFlowStore.getState();
        const node = store.nodes.find((n) => n.id === nodeId);
        let portInfo = {};

        if (node?.type === "dynamic") {
          const data = node.data;
          const port = (data.outputPorts?.find((p) => p.id === handleId) ??
            data.inputPorts?.find((p) => p.id === handleId) ??
            data.widgets?.find((w) => w.inputPortId === handleId)) as
            | Port
            | undefined;

          if (port?.type) {
            portInfo = {
              portType: port.type.mainType,
              mainType: port.type.mainType,
              itemType: port.type.itemType,
            };
          }
        }

        if (handleType) {
          setConnectionStartHandle({
            nodeId: nodeId ?? "",
            handleId: handleId ?? "",
            type: handleType,
            ...portInfo,
          });
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
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      onNodeContextMenu(event, node);
      void socketClient
        .send({
          payload: {
            case: "actionDiscovery",
            value: {
              nodeId: node.id,
              selectedNodeIds: nodes.filter((n) => n.selected).map((n) => n.id),
            },
          },
        })
        .catch((e: unknown) => {
          console.error("Failed to send action discovery", e);
        });
    },
    [onNodeContextMenu, nodes],
  );

  const handleExecuteAction = (action: ActionTemplate) => {
    const nodeId = contextMenu?.nodeId;
    if (!nodeId) return;
    const selectedIds = nodes.filter((n) => n.selected).map((n) => n.id);

    const taskId = uuidv4();
    useTaskStore.getState().registerTask({
      taskId,
      label: action.label,
      source: MutationSource.REMOTE_TASK,
    });

    void socketClient
      .send({
        payload: {
          case: "actionExecute",
          value: {
            actionId: action.id,
            sourceNodeId: nodeId,
            contextNodeIds: selectedIds,
            paramsJson: JSON.stringify({ taskId }),
          },
        },
      })
      .catch((e: unknown) => {
        console.error("Failed to execute action", e);
      });

    closeContextMenuAndClear();
  };

  useEffect(() => {
    const onActions = (actions: unknown) => {
      setAvailableActions(actions as ActionTemplate[]);
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

  const lastProcessedEventTimestamp = useRef<number>(0);

  useEffect(() => {
    if (
      !lastNodeEvent ||
      lastNodeEvent.timestamp === lastProcessedEventTimestamp.current
    )
      return;

    lastProcessedEventTimestamp.current = lastNodeEvent.timestamp;

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
            const targetWidgetId = val.split(":")[1];
            if (!targetWidgetId) return;
            let currentBuffer = "";
            socketClient.registerStreamHandler(
              nodeId,
              widgetId,
              (chunk: string) => {
                currentBuffer += chunk;
                const store = useFlowStore.getState();
                const currentNode = store.nodes.find((n) => n.id === nodeId);
                if (
                  currentNode &&
                  currentNode.type === "dynamic" &&
                  currentNode.data.widgets
                ) {
                  const updatedWidgets = currentNode.data.widgets.map((w) =>
                    w.id === targetWidgetId
                      ? { ...w, value: currentBuffer }
                      : w,
                  );
                  store.updateNodeData(nodeId, { widgets: updatedWidgets });
                }
              },
            );
            streamAction(nodeId, widgetId);
          } else if (val.startsWith("task:")) {
            const taskType = val.split(":")[1];
            if (!taskType) return;
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
                label: `Running ${taskType}...`,
                taskId,
                onCancel: (tid: string) => {
                  cancelTask(tid);
                },
              },
            } as AppNode;
            addNodeToStore(placeholderNode);
            useTaskStore.getState().registerTask({
              taskId,
              label: `Running ${taskType}...`,
              source: MutationSource.REMOTE_TASK,
            });
            executeTask(
              {
                id: taskType,
                label: taskType,
                strategy: 0,
                path: [],
              } as unknown as ActionTemplate,
              nodeId,
            );
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

  const onInit = useCallback((instance: ReactFlowInstance<AppNode>) => {
    console.log("React Flow Instance Ready", instance);
  }, []);

  const defaultEdgeOptions = useMemo(
    () => ({
      type: "default",
      animated: true,
      style: { strokeWidth: 2 },
    }),
    [],
  );

  const snapGrid: [number, number] = useMemo(() => [15, 15], []);

  return (
    <div
      style={{ width: "100vw", height: "100vh", backgroundColor: "var(--bg)" }}
      className={theme}
    >
      <ReactFlow
        nodes={nodes}
        edges={edges}
        onNodesChange={onNodesChangeWithSnapping}
        onEdgesChange={onEdgesChange}
        onConnect={onConnect}
        onInit={onInit}
        nodeTypes={memoizedNodeTypes}
        edgeTypes={memoizedEdgeTypes}
        onNodeDragStop={handleNodeDragStop}
        onConnectStart={onConnectStart}
        onConnectEnd={onConnectEnd}
        onNodeContextMenu={handleNodeContextMenu}
        onPaneContextMenu={onPaneContextMenu}
        fitView
        selectionMode={SelectionMode.Partial}
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
            onToggleTheme={toggleTheme}
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
      <Toaster position="bottom-right" />
    </div>
  );
}

const MemoizedApp = memo(App);
export default MemoizedApp;