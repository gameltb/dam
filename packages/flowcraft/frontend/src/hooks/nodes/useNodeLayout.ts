import { useMemo } from "react";

import type { DynamicNodeData } from "@/types";

const HEADER_HEIGHT = 46;
const PORT_HEIGHT_PER_ROW = 24;
const WIDGET_HEIGHT = 55;
const NODE_PADDING = 20;
const DEFAULT_NODE_WIDTH = 180;

/**
 * Calculates base structural layout metrics for a node.
 * Business-specific constraints are handled by the renderers.
 */
export function useNodeLayout(data: DynamicNodeData) {
  return useMemo(() => {
    const portRows = Math.max(data.inputPorts?.length ?? 0, data.outputPorts?.length ?? 0);
    const widgetsCount = data.widgets?.length ?? 0;

    const portsHeight = portRows * PORT_HEIGHT_PER_ROW;
    const widgetsHeight = widgetsCount * WIDGET_HEIGHT;
    const baseHeight = HEADER_HEIGHT + portsHeight + widgetsHeight + NODE_PADDING;

    return {
      minHeight: baseHeight,
      minWidth: DEFAULT_NODE_WIDTH,
    };
  }, [data]);
}
