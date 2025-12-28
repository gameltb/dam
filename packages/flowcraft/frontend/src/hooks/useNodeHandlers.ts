import { useCallback } from "react";
import { useFlowStore } from "../store/flowStore";
import { type MediaType } from "../types";

export function useNodeHandlers() {
  const updateNodeData = useFlowStore((state) => state.updateNodeData);
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

  const onChange = useCallback(
    (id: string, data: Record<string, unknown>) => {
      updateNodeData(id, data);
    },
    [updateNodeData],
  );

  const onWidgetClick = useCallback(
    (nodeId: string, widgetId: string) => {
      dispatchNodeEvent("widget-click", { nodeId, widgetId });
    },
    [dispatchNodeEvent],
  );

  const onGalleryItemContext = useCallback(
    (
      nodeId: string,
      url: string,
      mediaType: MediaType,
      x: number,
      y: number,
    ) => {
      dispatchNodeEvent("gallery-context", { nodeId, url, mediaType, x, y });
    },
    [dispatchNodeEvent],
  );

  return {
    onChange,
    onWidgetClick,
    onGalleryItemContext,
  };
}
