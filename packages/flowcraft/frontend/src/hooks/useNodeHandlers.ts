import { useMemo, useCallback } from "react";
import { MediaType } from "../generated/flowcraft/v1/core/node_pb";
import type { DynamicNodeData } from "../types";
import { useNodeLayout } from "./useNodeLayout";
import { useFlowStore } from "../store/flowStore";

export interface NodeHandlersResult {
  minHeight: number;
  minWidth: number;
  isMedia: boolean;
  isAudio: boolean;
  shouldLockAspectRatio: boolean;
  containerStyle: React.CSSProperties;
  onChange: (id: string, data: Partial<DynamicNodeData>) => void;
  onWidgetClick: (nodeId: string, widgetId: string) => void;
  onGalleryItemContext: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
}

/**
 * Hook that provides layout, styling, and common event handlers for nodes.
 * Replaces the withNodeHandlers HOC and provides access to the flow store.
 */
export function useNodeHandlers(
  data?: DynamicNodeData,
  selected?: boolean,
  _positionAbsoluteX?: number,
  _positionAbsoluteY?: number,
): NodeHandlersResult {
  const updateNodeData = useFlowStore((state) => state.updateNodeData);
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);

  const layout = useNodeLayout(data ?? ({} as DynamicNodeData));
  const { minHeight, minWidth, isMedia, isAudio } = layout;

  const shouldLockAspectRatio = useMemo(() => {
    return (
      isMedia &&
      (data?.media?.type === MediaType.MEDIA_IMAGE ||
        data?.media?.type === MediaType.MEDIA_VIDEO)
    );
  }, [isMedia, data?.media?.type]);

  const containerStyle = useMemo((): React.CSSProperties => {
    return {
      position: "relative",
      width: "100%",
      height: "100%",
      display: "flex",
      flexDirection: "column",
      borderRadius: "8px",
      backgroundColor: "var(--node-bg)",
      color: "var(--text-color)",
      border: "1px solid",
      borderColor: selected ? "var(--primary-color)" : "var(--node-border)",
      boxShadow: selected
        ? "0 0 0 1px var(--primary-color), 0 0 15px rgba(100, 108, 255, 0.4), 0 10px 25px rgba(0,0,0,0.4)"
        : "0 4px 12px rgba(0,0,0,0.2)",
      boxSizing: "border-box",
      transition: "border-color 0.2s ease, box-shadow 0.2s ease",
    };
  }, [selected]);

  const onChange = useCallback(
    (id: string, newData: Partial<DynamicNodeData>) => {
      updateNodeData(id, newData);
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
      dispatchNodeEvent("gallery-item-context", {
        nodeId,
        url,
        mediaType,
        x,
        y,
      });
    },
    [dispatchNodeEvent],
  );

  return {
    minHeight,
    minWidth,
    isMedia,
    isAudio,
    shouldLockAspectRatio,
    containerStyle,
    onChange,
    onWidgetClick,
    onGalleryItemContext,
  };
}
