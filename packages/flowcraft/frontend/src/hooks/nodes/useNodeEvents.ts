import { useCallback } from "react";
import { useShallow } from "zustand/react/shallow";

import { MediaType } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { editNode } from "@/store/orchestrator";
import { type RFState } from "@/store/types";
import { type DynamicNodeData, FlowEvent } from "@/types";

export function useNodeEvents() {
  const dispatchNodeEvent = useFlowStore(useShallow((s: RFState) => s.dispatchNodeEvent));

  const onChange = useCallback((nodeId: string, newData: Partial<DynamicNodeData>) => {
    editNode(nodeId, (draft) => {
      Object.assign(draft.data, newData);
    });
  }, []);

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
    onChange,
    onGalleryItemContext,
    onWidgetClick,
  };
}
