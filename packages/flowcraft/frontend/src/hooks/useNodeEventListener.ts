import { useEffect, useRef } from "react";
import { v4 as uuidv4 } from "uuid";

import { type ActionTemplate } from "../generated/flowcraft/v1/core/action_pb";
import { useFlowStore } from "../store/flowStore";
import { useTaskStore } from "../store/taskStore";
import {
  type AppNode,
  AppNodeType,
  FlowEvent,
  isDynamicNode,
  MediaType,
  MutationSource,
} from "../types";

export interface ContextMenuData {
  galleryItemType?: MediaType;
  galleryItemUrl?: string;
  nodeId: string;
  x: number;
  y: number;
}

export interface PreviewData {
  index: number;
  nodeId: string;
}

interface NodeEventListenerProps {
  addNodeToStore: (node: AppNode) => void;
  cancelTask: (taskId: string) => void;
  executeTask: (action: ActionTemplate, nodeId: string) => void;
  nodes: AppNode[];
  setActiveEditorId: (id: null | string) => void;
  setContextMenu: (menu: ContextMenuData | null) => void;
  setPreviewData: (data: null | PreviewData) => void;
  streamAction: (nodeId: string, widgetId: string) => void;
}

export function useNodeEventListener({
  addNodeToStore,
  cancelTask,
  executeTask,
  nodes,
  setActiveEditorId,
  setContextMenu,
  setPreviewData,
  streamAction,
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

    if (lastNodeEvent.type === FlowEvent.GALLERY_ITEM_CONTEXT) {
      const payload = lastNodeEvent.payload as {
        mediaType: MediaType;
        nodeId: string;
        url: string;
        x: number;
        y: number;
      };
      setContextMenu({
        galleryItemType: payload.mediaType,
        galleryItemUrl: payload.url,
        nodeId: payload.nodeId,
        x: payload.x,
        y: payload.y,
      });
    } else if (lastNodeEvent.type === FlowEvent.OPEN_PREVIEW) {
      const payload = lastNodeEvent.payload as {
        index: number;
        nodeId: string;
      };
      setTimeout(() => {
        setPreviewData(payload);
      }, 0);
    } else if (lastNodeEvent.type === FlowEvent.OPEN_EDITOR) {
      const payload = lastNodeEvent.payload as { nodeId: string };
      setTimeout(() => {
        setActiveEditorId(payload.nodeId);
      }, 0);
    } else if (lastNodeEvent.type === FlowEvent.WIDGET_CLICK) {
      const { nodeId, widgetId } = lastNodeEvent.payload as {
        nodeId: string;
        widgetId: string;
      };
      const node = nodes.find((n) => n.id === nodeId);
      if (node && isDynamicNode(node) && node.data.widgets) {
        const widget = node.data.widgets.find((w) => w.id === widgetId);
        if (widget && typeof widget.value === "string") {
          const val = widget.value;
          if (val.startsWith("task:")) {
            const taskType = val.split(":")[1];
            if (!taskType) return;
            const taskId = uuidv4();
            const position = {
              x: node.position.x + 300,
              y: node.position.y,
            };
            const placeholderNode: AppNode = {
              data: {
                label: `Running ${taskType}...`,
                onCancel: (tid: string) => {
                  cancelTask(tid);
                },
                taskId,
              },
              id: `task-${taskId}`,
              position,
              type: AppNodeType.PROCESSING,
            } as AppNode;
            addNodeToStore(placeholderNode);
            useTaskStore.getState().registerTask({
              label: `Running ${taskType}...`,
              source: MutationSource.SOURCE_REMOTE_TASK,
              taskId,
            });
            executeTask(
              {
                id: taskType,
                label: taskType,
                path: [],
                strategy: 0,
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
