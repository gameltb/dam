import { type AppNode, isDynamicNode } from "../types";

/**
 * Hydrates a node with client-side handlers that cannot be sent over the wire (JSON).
 */
export function hydrateNode(
  node: AppNode,
  handlers: {
    onChange: (id: string, data: Record<string, unknown>) => void;
    onWidgetClick?: (nodeId: string, widgetId: string) => void;
    onGalleryItemContext?: (
      nodeId: string,
      url: string,
      x: number,
      y: number,
    ) => void;
  },
): AppNode {
  if (isDynamicNode(node)) {
    return {
      ...node,
      data: {
        ...node.data,
        onChange: handlers.onChange,
        onWidgetClick: handlers.onWidgetClick,
        onGalleryItemContext: handlers.onGalleryItemContext,
      },
    };
  }
  return node;
}

/**
 * Bulk hydrates an array of nodes.
 */
export function hydrateNodes(
  nodes: AppNode[],
  handlers: {
    onChange: (id: string, data: Record<string, unknown>) => void;
    onWidgetClick?: (nodeId: string, widgetId: string) => void;
    onGalleryItemContext?: (
      nodeId: string,
      url: string,
      x: number,
      y: number,
    ) => void;
  },
): AppNode[] {
  return nodes.map((node) => hydrateNode(node, handlers));
}
