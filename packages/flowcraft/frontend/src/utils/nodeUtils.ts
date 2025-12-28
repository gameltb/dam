import { type AppNode, isDynamicNode, type MediaType } from "../types";

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
      mediaType: MediaType,
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
      mediaType: MediaType,
      x: number,
      y: number,
    ) => void;
  },
): AppNode[] {
  return nodes.map((node) => hydrateNode(node, handlers));
}

/**
 * Dehydrates a node by removing client-side handlers so it can be safely serialized (e.g. to Yjs or JSON).
 */
export function dehydrateNode(node: AppNode): AppNode {
  if (isDynamicNode(node) && node.data) {
    // Destructure handlers out to exclude them from the returned object
    const {
      onChange: _onChange,
      onWidgetClick: _onWidgetClick,
      onGalleryItemContext: _onGalleryItemContext,
      ...rest
    } = node.data;
    return {
      ...node,
      data: rest,
    };
  }
  return node;
}
