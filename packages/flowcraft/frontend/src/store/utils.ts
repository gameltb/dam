import { type Edge } from "@xyflow/react";

import { type RFState } from "@/store/types";
import { type AppNode } from "@/types";

import { yEdges, yNodes } from "./yjsInstance";

export function syncFromYjs(state: RFState): Partial<RFState> {
  const rawNodes: AppNode[] = [];
  yNodes.forEach((v) => rawNodes.push(v as AppNode));

  const edges: Edge[] = [];
  yEdges.forEach((v) => edges.push(v as Edge));

  // If layout is not dirty, we can just update nodes without sorting
  if (!state.isLayoutDirty) {
    return { edges, nodes: rawNodes, version: state.version + 1 };
  }

  const nodes: AppNode[] = [];
  const visited = new Set<string>();

  const visit = (node: AppNode) => {
    if (visited.has(node.id)) return;

    if (node.parentId && !visited.has(node.parentId)) {
      const parent = rawNodes.find((n) => n.id === node.parentId);
      if (parent) visit(parent);
    }

    visited.add(node.id);
    nodes.push(node);
  };

  rawNodes.forEach((n) => {
    visit(n);
  });

  return {
    edges,
    isLayoutDirty: false,
    nodes,
    version: state.version + 1,
  };
}
