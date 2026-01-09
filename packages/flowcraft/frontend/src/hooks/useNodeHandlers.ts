import { useCallback, useMemo } from "react";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, FlowEvent } from "@/types";

import { useNodeLayout } from "./useNodeLayout";

export interface NodeHandlersResult {
  containerStyle: React.CSSProperties;
  isAudio: boolean;
  isMedia: boolean;
  minHeight: number;
  minWidth: number;
  onChange: (id: string, data: Partial<DynamicNodeData>) => void;
  onGalleryItemContext: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
  onWidgetClick: (nodeId: string, widgetId: string) => void;
  shouldLockAspectRatio: boolean;
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
  const { isAudio, isMedia, minHeight, minWidth } = layout;

  const shouldLockAspectRatio = useMemo(() => {
    return (
      isMedia &&
      (data?.media?.type === MediaType.MEDIA_IMAGE ||
        data?.media?.type === MediaType.MEDIA_VIDEO)
    );
  }, [isMedia, data?.media?.type]);

  const containerStyle = useMemo((): React.CSSProperties => {
    return {
      backgroundColor: "var(--node-bg)",
      border: "1px solid",
      borderColor: selected ? "var(--primary-color)" : "var(--node-border)",
      borderRadius: "var(--radius-lg)",
      boxShadow: selected
        ? "0 0 0 1px var(--primary-color), 0 0 15px rgba(100, 108, 255, 0.4), 0 10px 25px rgba(0,0,0,0.4)"
        : "0 4px 12px rgba(0,0,0,0.2)",
      boxSizing: "border-box",
      color: "var(--text-color)",
      display: "flex",
      flexDirection: "column",
      height: "100%",
      position: "relative",
      transition: "border-color 0.2s ease, box-shadow 0.2s ease",
      width: "100%",
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
      dispatchNodeEvent(FlowEvent.WIDGET_CLICK, { nodeId, widgetId });
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
      dispatchNodeEvent(FlowEvent.GALLERY_ITEM_CONTEXT, {
        mediaType,
        nodeId,
        url,
        x,
        y,
      });
    },
    [dispatchNodeEvent],
  );

  return {
    containerStyle,
    isAudio,
    isMedia,
    minHeight,
    minWidth,
    onChange,
    onGalleryItemContext,
    onWidgetClick,
    shouldLockAspectRatio,
  };
}
