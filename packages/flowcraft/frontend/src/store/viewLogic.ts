import { type Edge as RFEdge } from "@xyflow/react";

import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode, Scope } from "@/types";
import { calculateNodeRelations } from "@/utils/nodeUtils";

/**
 * Computes which nodes and edges should be displayed in the current scope.
 */
export function computeView(
  nodesById: Record<string, AppNode>,
  edgesById: Record<string, RFEdge>,
  existingRelations?: Record<string, any>,
) {
  const allNodes = Object.values(nodesById);
  const allEdges = Object.values(edgesById);
  const { activeScopeId } = useNavigationStore.getState();
  const currentScope = activeScopeId || Scope.ROOT;

  // 1. Filter nodes by logical scopeId and ensure they are valid for React Flow
  const currentLevelNodes = allNodes
    .filter((n) => n.scopeId === currentScope)
    .map((n) => {
      if (!n.position) {
        console.warn(`[computeView] Node ${n.id} is missing position! Fixing to {0,0}.`, n);
        return { ...n, position: { x: 0, y: 0 } };
      }
      return n;
    });

  const nextNodes = [...currentLevelNodes];

  // 2. Virtual root node injection: If in a sub-scope, we might need to show a container reference (for UI assistance)
  if (activeScopeId && nodesById[activeScopeId]) {
    const parentNode = nodesById[activeScopeId];
    nextNodes.unshift({
      ...parentNode,
      draggable: false,
      position: { x: 0, y: 0 },
      selectable: false,
      style: { ...parentNode.style, opacity: 0, pointerEvents: "none" },
      zIndex: -1,
    });
  }

  // 3. Filter edges
  const nextEdges = allEdges.filter((e) => {
    const sourceNode = nodesById[e.source];
    const targetNode = nodesById[e.target];
    if (sourceNode && targetNode) {
      return sourceNode.scopeId === currentScope;
    }
    return true;
  });

  const nodeRelations = existingRelations || calculateNodeRelations(allNodes, allEdges);

  return {
    edges: nextEdges,
    nodeRelations,
    nodes: nextNodes,
  };
}
