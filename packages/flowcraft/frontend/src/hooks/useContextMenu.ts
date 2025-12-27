import { useCallback, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { Node, Edge } from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";
import type { MediaType } from "../types";

export const useContextMenu = () => {
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId?: string;
    edgeId?: string;
    galleryItemUrl?: string;
    galleryItemType?: MediaType;
  } | null>(null);

  const onPaneContextMenu = useCallback(
    (event: ReactMouseEvent | MouseEvent) => {
      event.preventDefault();
      const clientX =
        "clientX" in event ? event.clientX : (event as MouseEvent).clientX;
      const clientY =
        "clientY" in event ? event.clientY : (event as MouseEvent).clientY;
      setContextMenu({
        x: clientX,
        y: clientY,
      });
    },
    [],
  );

  const onNodeContextMenu = useCallback(
    (event: ReactMouseEvent, node: Node) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
    },
    [],
  );

  const onEdgeContextMenu = useCallback(
    (event: ReactMouseEvent, edge: Edge) => {
      event.preventDefault();
      setContextMenu({ x: event.clientX, y: event.clientY, edgeId: edge.id });
    },
    [],
  );

  const onSelectionContextMenu = useCallback((event: ReactMouseEvent) => {
    event.preventDefault();
    setContextMenu({
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const onPaneClick = useCallback(() => {
    setContextMenu(null);
    dispatchNodeEvent("pane-click", {});
  }, [dispatchNodeEvent]);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  return {
    contextMenu,
    onPaneContextMenu,
    onNodeContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu,
    onPaneClick,
    closeContextMenu,
    setContextMenu,
  };
};
