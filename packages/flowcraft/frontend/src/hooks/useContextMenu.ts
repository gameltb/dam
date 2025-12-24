import { useCallback, useState } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { Node, Edge } from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";

export const useContextMenu = () => {
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId?: string;
    edgeId?: string;
    galleryItemUrl?: string;
    galleryItemType?: string;
  } | null>(null);

  const onPaneContextMenu = useCallback(
    (event: ReactMouseEvent | MouseEvent) => {
      event.preventDefault();
      setContextMenu({
        x: (event as any).clientX, // eslint-disable-line @typescript-eslint/no-explicit-any
        y: (event as any).clientY, // eslint-disable-line @typescript-eslint/no-explicit-any
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

  const closeContextMenu = useCallback(() => setContextMenu(null), []);

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
