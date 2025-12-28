import { type AppNode } from "../types";

/**
 * Dehydrates a node by ensuring it only contains serializable data.
 * In the new architecture, handlers are provided via context/hooks and are no longer part of the node state.
 */
export function dehydrateNode(node: AppNode): AppNode {
  // We want to keep all standard React Flow properties (id, type, position, data, parentId, extent, measured, etc.)
  // but ensure no non-serializable content (like functions) is present.
  const { ...serializableNode } = node;
  return serializableNode as AppNode;
}

/**
 * Legacy hydration function (kept for backward compatibility during refactor if needed, but empty).
 * In the new architecture, we don't modify node objects with functions.
 */
export function hydrateNodes(nodes: AppNode[]): AppNode[] {
  return nodes;
}
