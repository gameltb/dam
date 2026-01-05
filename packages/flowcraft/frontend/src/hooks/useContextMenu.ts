import { useCallback, useState, useEffect } from "react";
import type { MouseEvent as ReactMouseEvent } from "react";
import type { Node, Edge } from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";
import { type MediaType, FlowEvent } from "../types";

export const useContextMenu = (): {
  contextMenu: {
    x: number;
    y: number;
    nodeId?: string;
    edgeId?: string;
    galleryItemUrl?: string;
    galleryItemType?: MediaType;
  } | null;
  onPaneContextMenu: (event: ReactMouseEvent | MouseEvent) => void;
  onNodeContextMenu: (event: ReactMouseEvent, node: Node) => void;
  onEdgeContextMenu: (event: ReactMouseEvent, edge: Edge) => void;
  onSelectionContextMenu: (event: ReactMouseEvent) => void;
  onPaneClick: () => void;
  closeContextMenu: () => void;
  closeContextMenuAndClear: () => void;
  onNodeDragStop: () => void;
  setContextMenu: React.Dispatch<
    React.SetStateAction<{
      x: number;
      y: number;
      nodeId?: string;
      edgeId?: string;
      galleryItemUrl?: string;
      galleryItemType?: MediaType;
    } | null>
  >;
} => {
  const dispatchNodeEvent = useFlowStore((state) => state.dispatchNodeEvent);
  const [contextMenu, setContextMenu] = useState<{
    x: number;
    y: number;
    nodeId?: string;
    edgeId?: string;
    galleryItemUrl?: string;
    galleryItemType?: MediaType;
  } | null>(null);

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
    (event: ReactMouseEvent | MouseEvent) => {
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
      setContextMenu({ x: event.clientX, y: event.clientY, nodeId: node.id });
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
      setContextMenu({ x: event.clientX, y: event.clientY, edgeId: edge.id });
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
    contextMenu,
    onPaneContextMenu,
    onNodeContextMenu,
    onEdgeContextMenu,
    onSelectionContextMenu,
    onPaneClick,
    closeContextMenu,
    closeContextMenuAndClear,
    onNodeDragStop,
    setContextMenu,
  };
};
