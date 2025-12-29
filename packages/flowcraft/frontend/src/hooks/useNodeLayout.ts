import { useMemo } from "react";
import { RenderMode, MediaType } from "../generated/core/node_pb";
import type { DynamicNodeData } from "../types";

const HEADER_HEIGHT = 46;
const PORT_HEIGHT_PER_ROW = 24;
const WIDGET_HEIGHT = 55;
const NODE_PADDING = 20;
const MEDIA_AUDIO_MIN_HEIGHT = 110;
const MEDIA_DEFAULT_MIN_HEIGHT = 50;

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

    let minHeight = MEDIA_DEFAULT_MIN_HEIGHT;

    if (isMedia) {
      minHeight = isAudio ? MEDIA_AUDIO_MIN_HEIGHT : MEDIA_DEFAULT_MIN_HEIGHT;
    } else {
      const portsHeight = portRows * PORT_HEIGHT_PER_ROW;
      const widgetsHeight = widgetsCount * WIDGET_HEIGHT;
      minHeight = HEADER_HEIGHT + portsHeight + widgetsHeight + NODE_PADDING;
    }

    return {
      minHeight,
      isMedia,
      isAudio,
    };
  }, [data]);
}
