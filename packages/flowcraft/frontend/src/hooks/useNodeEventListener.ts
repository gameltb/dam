import { useEffect, useRef } from "react";
import { useFlowStore } from "../store/flowStore";
import { useTaskStore } from "../store/taskStore";
import { socketClient } from "../utils/SocketClient";
import { v4 as uuidv4 } from "uuid";
import { type AppNode, MediaType, MutationSource } from "../types";
import { type ActionTemplate } from "../generated/action_pb";

interface NodeEventListenerProps {
  nodes: AppNode[];
  setContextMenu: (menu: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any
  setPreviewData: (data: any) => void; // eslint-disable-line @typescript-eslint/no-explicit-any
  setActiveEditorId: (id: string | null) => void;
  streamAction: (nodeId: string, widgetId: string) => void;
  addNodeToStore: (node: AppNode) => void;
  cancelTask: (taskId: string) => void;
  executeTask: (action: ActionTemplate, nodeId: string) => void;
}

export function useNodeEventListener({
  nodes,
  setContextMenu,
  setPreviewData,
  setActiveEditorId,
  streamAction,
  addNodeToStore,
  cancelTask,
  executeTask,
}: NodeEventListenerProps) {
  const lastNodeEvent = useFlowStore((state) => state.lastNodeEvent);
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
    setPreviewData,
    setActiveEditorId,
  ]);
}
