import { useMemo } from "react";
import { RenderMode, MediaType } from "../generated/core/node_pb";
import type { DynamicNodeData } from "../types";
import {
  AUDIO_RENDERER_CONFIG,
  IMAGE_RENDERER_CONFIG,
  VIDEO_RENDERER_CONFIG,
} from "../components/media/mediaConfigs";

const HEADER_HEIGHT = 46;
const PORT_HEIGHT_PER_ROW = 24;
const WIDGET_HEIGHT = 55;
const NODE_PADDING = 20;
const DEFAULT_NODE_WIDTH = 180;

/**
 * Calculates layout metrics for a node based on its data and active mode.
 */
export function useNodeLayout(data: DynamicNodeData) {
  return useMemo(() => {
    const isMedia = data.activeMode === RenderMode.MODE_MEDIA;
    const isAudio = isMedia && data.media?.type === MediaType.MEDIA_AUDIO;

    const portRows = Math.max(
      data.inputPorts?.length ?? 0,
      data.outputPorts?.length ?? 0,
    );
    const widgetsCount = data.widgets?.length ?? 0;

    let minHeight = 50;
    let minWidth = DEFAULT_NODE_WIDTH;

    if (isMedia) {
      switch (data.media?.type) {
        case MediaType.MEDIA_AUDIO:
          minHeight = AUDIO_RENDERER_CONFIG.minHeight;
          minWidth = AUDIO_RENDERER_CONFIG.minWidth;
          break;
        case MediaType.MEDIA_IMAGE:
          minHeight = IMAGE_RENDERER_CONFIG.minHeight;
          minWidth = IMAGE_RENDERER_CONFIG.minWidth;
          break;
        case MediaType.MEDIA_VIDEO:
          minHeight = VIDEO_RENDERER_CONFIG.minHeight;
          minWidth = VIDEO_RENDERER_CONFIG.minWidth;
          break;
        default:
          minHeight = 50;
          minWidth = DEFAULT_NODE_WIDTH;
      }
    } else {
      const portsHeight = portRows * PORT_HEIGHT_PER_ROW;
      const widgetsHeight = widgetsCount * WIDGET_HEIGHT;
      minHeight = HEADER_HEIGHT + portsHeight + widgetsHeight + NODE_PADDING;
    }

    return {
      minHeight,
      minWidth,
      isMedia,
      isAudio,
    };
  }, [data]);
}
