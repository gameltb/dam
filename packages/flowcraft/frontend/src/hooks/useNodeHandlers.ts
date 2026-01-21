import { useCallback, useMemo } from "react";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
import { useFlowStore } from "@/store/flowStore";
import { type DynamicNodeData, FlowEvent } from "@/types";

import { useNodeLayout } from "./useNodeLayout";

export interface NodeHandlersResult {
  containerStyle: React.CSSProperties;
  isAudio: boolean;
  isMedia: boolean;
  minHeight: number;
  minWidth: number;
  onChange: (id: string, data: Record<string, any>) => void;
  onGalleryItemContext: (nodeId: string, url: string, mediaType: MediaType, x: number, y: number) => void;
  onWidgetClick: (nodeId: string, widgetId: string) => void;
  shouldLockAspectRatio: boolean;
}

import { useShallow } from "zustand/react/shallow";

import { type RFState } from "@/store/types";
import { type AppNode } from "@/types";

/**
 * Hook that provides layout, styling, and common event handlers for nodes.
 */
export const useNodeHandlers = (data: DynamicNodeData, selected?: boolean, _idOrX?: number | string, _y?: number) => {
  const { allNodes, dispatchNodeEvent, nodeDraft } = useFlowStore(
    useShallow((s: RFState) => ({
      allNodes: s.allNodes,
      dispatchNodeEvent: s.dispatchNodeEvent,
      nodeDraft: s.nodeDraft,
    })),
  );

  const layout = useNodeLayout(data ?? ({} as DynamicNodeData));

  const { minHeight, minWidth } = layout;

  const isMedia = data?.activeMode === RenderMode.MODE_MEDIA;
  const getMediaType = () => {
    if (data?.media?.type !== undefined) return data.media.type;
    if (
      data?.extension?.case === "visual" ||
      data?.extension?.case === "document" ||
      data?.extension?.case === "acoustic"
    ) {
      return (data.extension.value as any).type;
    }
    return undefined;
  };
  const mediaType = getMediaType();
  const isAudio = isMedia && mediaType === MediaType.MEDIA_AUDIO;

  const shouldLockAspectRatio = useMemo(() => {
    return isMedia && (mediaType === MediaType.MEDIA_IMAGE || mediaType === MediaType.MEDIA_VIDEO);
  }, [isMedia, mediaType]);

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

  /**
   * ORM 风格更新代理
   */
  const onChange = useCallback(
    (nodeId: string, newData: Record<string, any>) => {
      const node = allNodes.find((n: AppNode) => n.id === nodeId);
      if (!node) return;

      const res = nodeDraft(node);
      if (res.ok) {
        Object.entries(newData).forEach(([key, value]) => {
          (res.value.data as any)[key] = value;
        });
      }
    },
    [allNodes, nodeDraft],
  );

  const onWidgetClick = useCallback(
    (nodeId: string, widgetId: string) => {
      dispatchNodeEvent(FlowEvent.WIDGET_CLICK, { nodeId, widgetId });
    },
    [dispatchNodeEvent],
  );

  const onGalleryItemContext = useCallback(
    (nodeId: string, url: string, mediaType: MediaType, x: number, y: number) => {
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
};
