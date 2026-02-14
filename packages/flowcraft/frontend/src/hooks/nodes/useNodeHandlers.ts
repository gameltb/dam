import { useNodeEvents } from "@/hooks/nodes/useNodeEvents";
import { useNodeLayout } from "@/hooks/nodes/useNodeLayout";
import { useNodeMedia } from "@/hooks/nodes/useNodeMedia";
import { useNodeStyle } from "@/hooks/nodes/useNodeStyle";
import { type DynamicNodeData } from "@/types";

export const useNodeHandlers = (data: DynamicNodeData, selected?: boolean) => {
  const layout = useNodeLayout(data);
  const { minHeight, minWidth } = layout;

  const media = useNodeMedia(data);
  const { isAudio, isMedia, shouldLockAspectRatio } = media;

  const containerStyle = useNodeStyle(selected);

  const events = useNodeEvents();
  const { onChange, onGalleryItemContext, onWidgetClick } = events;

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
