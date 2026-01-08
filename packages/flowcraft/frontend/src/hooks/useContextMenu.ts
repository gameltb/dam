import type { Edge, Node } from "@xyflow/react";
import type { MouseEvent as ReactMouseEvent } from "react";

import { useCallback, useEffect, useState } from "react";

import { useFlowStore } from "@/store/flowStore";
import { FlowEvent, type MediaType } from "@/types";

export const useContextMenu = (): {
  closeContextMenu: () => void;
  closeContextMenuAndClear: () => void;
  contextMenu: null | {
    edgeId?: string;
    galleryItemType?: MediaType;
    galleryItemUrl?: string;
    nodeId?: string;
    x: number;
    y: number;
  };
  onEdgeContextMenu: (event: ReactMouseEvent, edge: Edge) => void;
  onNodeContextMenu: (event: ReactMouseEvent, node: Node) => void;
  onNodeDragStop: () => void;
  onPaneClick: () => void;
  onPaneContextMenu: (event: MouseEvent | ReactMouseEvent) => void;
  onSelectionContextMenu: (event: ReactMouseEvent) => void;
  setContextMenu: React.Dispatch<
    React.SetStateAction<null | {
      edgeId?: string;
      galleryItemType?: MediaType;
      galleryItemUrl?: string;
      nodeId?: string;
      x: number;
      y: number;
    }>
  >;
} => {
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);
  const [contextMenu, setContextMenu] = useState<null | {
    edgeId?: string;
    galleryItemType?: MediaType;
    galleryItemUrl?: string;
    nodeId?: string;
    x: number;
    y: number;
  }>(null);

  useEffect(() => {
    if (!contextMenu) return;

    const handleOutsideClick = (event: MouseEvent) => {
      // We check if the click target is inside a context menu or its submenus
      // Since context menus are often portals or fixed elements, we can check for a common class or attribute
      const target = event.target as HTMLElement;
      if (target.closest(".context-menu-container")) return;

      setContextMenu(null);
    };

    // Use capture to ensure we catch clicks before they might be stopped by other handlers
    window.addEventListener("click", handleOutsideClick, true);
    return () => {
      window.removeEventListener("click", handleOutsideClick, true);
    };
  }, [contextMenu]);

  const onPaneContextMenu = useCallback(
    (event: MouseEvent | ReactMouseEvent) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
        event.stopPropagation();
        return;
      }

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
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
        event.stopPropagation();
        return;
      }

      event.preventDefault();
      setContextMenu({ nodeId: node.id, x: event.clientX, y: event.clientY });
    },
    [],
  );

  const onEdgeContextMenu = useCallback(
    (event: ReactMouseEvent, edge: Edge) => {
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
        event.stopPropagation();
        return;
      }

      event.preventDefault();
      setContextMenu({ edgeId: edge.id, x: event.clientX, y: event.clientY });
    },
    [],
  );

  const onSelectionContextMenu = useCallback((event: ReactMouseEvent) => {
    const target = event.target as HTMLElement;
    if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") {
      event.stopPropagation();
      return;
    }

    event.preventDefault();
    setContextMenu({
      x: event.clientX,
      y: event.clientY,
    });
  }, []);

  const onPaneClick = useCallback(() => {
    setContextMenu(null);
    dispatchNodeEvent(FlowEvent.PANE_CLICK, {});
  }, [dispatchNodeEvent]);

  const closeContextMenu = useCallback(() => {
    setContextMenu(null);
  }, []);

  const closeContextMenuAndClear = useCallback(() => {
    setContextMenu(null);
  }, []);

  const onNodeDragStop = useCallback(() => {
    setContextMenu(null);
  }, []);

  return {
    closeContextMenu,
    closeContextMenuAndClear,
    contextMenu,
    onEdgeContextMenu,
    onNodeContextMenu,
    onNodeDragStop,
    onPaneClick,
    onPaneContextMenu,
    onSelectionContextMenu,
    setContextMenu,
  };
};
