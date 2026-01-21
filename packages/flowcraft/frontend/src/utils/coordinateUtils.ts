import { type XYPosition } from "@xyflow/react";

import { type AppNode } from "@/types";

/**
 * 将全局坐标转换为相对于指定父级的局部坐标
 */
export function globalToLocal(globalPos: XYPosition, newParentId: null | string, allNodes: AppNode[]): XYPosition {
  if (!newParentId) return globalPos;

  const parent = allNodes.find((n) => n.id === newParentId);
  if (!parent) return globalPos;

  const parentGlobalPos = localToGlobal(parent.position, parent.parentId || null, allNodes);

  return {
    x: globalPos.x - parentGlobalPos.x,
    y: globalPos.y - parentGlobalPos.y,
  };
}

/**
 * 将局部坐标（相对于父级）转换为全局坐标
 */
export function localToGlobal(localPos: XYPosition, parentId: null | string, allNodes: AppNode[]): XYPosition {
  if (!parentId) return localPos;

  const parent = allNodes.find((n) => n.id === parentId);
  if (!parent) return localPos;

  // 递归计算
  const parentGlobalPos = localToGlobal(parent.position, parent.parentId || null, allNodes);

  return {
    x: parentGlobalPos.x + localPos.x,
    y: parentGlobalPos.y + localPos.y,
  };
}
