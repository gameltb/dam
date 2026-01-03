import { useMemo } from "react";
import { RenderMode, MediaType } from "../generated/flowcraft/v1/node_pb";
import type { DynamicNodeData } from "../types";
import { MEDIA_CONFIGS } from "../components/media/mediaConfigs";

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

    if (isMedia && data.media?.type !== undefined) {
      const config = MEDIA_CONFIGS[data.media.type];
      if (config) {
        minHeight = config.minHeight;
        minWidth = config.minWidth;
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
