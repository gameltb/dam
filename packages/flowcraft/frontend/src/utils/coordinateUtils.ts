import { type XYPosition } from "@xyflow/react";

import { type AppNode } from "@/types";

/**
 * Converts global coordinates to local coordinates relative to a specified parent.
 */
export function globalToLocal(
  globalPos: XYPosition,
  newParentId: null | string,
  nodesById: Record<string, AppNode>,
  visited = new Set<string>(),
): XYPosition {
  if (!newParentId) return globalPos;
  if (visited.has(newParentId)) return globalPos;
  visited.add(newParentId);

  const parent = nodesById[newParentId];
  if (!parent) {
    return globalPos;
  }

  const parentGlobalPos = localToGlobal(parent.position, parent.parentId || null, nodesById, visited);

  return {
    x: globalPos.x - parentGlobalPos.x,
    y: globalPos.y - parentGlobalPos.y,
  };
}

/**
 * Converts local coordinates (relative to parent) to global coordinates.
 */
export function localToGlobal(
  localPos: XYPosition,
  parentId: null | string,
  nodesById: Record<string, AppNode>,
  visited = new Set<string>(),
): XYPosition {
  if (!parentId) return localPos;
  if (visited.has(parentId)) return localPos;
  visited.add(parentId);

  const parent = nodesById[parentId];
  if (!parent) {
    return localPos;
  }

  // Recursive calculation
  const parentGlobalPos = localToGlobal(parent.position, parent.parentId || null, nodesById, visited);

  return {
    x: parentGlobalPos.x + localPos.x,
    y: parentGlobalPos.y + localPos.y,
  };
}
