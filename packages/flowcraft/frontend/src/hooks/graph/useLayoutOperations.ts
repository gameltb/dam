import { useCallback } from "react";
import dagre from "dagre";
import { Scope } from "@/types";
import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { commit } from "@/store/orchestrator";

export const useLayoutOperations = () => {
  const { nodes, edges } = useFlowStore();

  const autoLayout = useCallback(
    () => {
      const activeScopeId = useNavigationStore.getState().activeScopeId || Scope.ROOT;
      const g = new dagre.graphlib.Graph();
      g.setGraph({ rankdir: "LR", nodesep: 50, ranksep: 100 });
      g.setDefaultEdgeLabel(() => ({}));

      const currentNodes = nodes.filter(n => n.scopeId === activeScopeId);
      const currentEdges = edges.filter(e => {
        const source = nodes.find(n => n.id === e.source);
        return source?.scopeId === activeScopeId;
      });

      currentNodes.forEach((node) => {
        g.setNode(node.id, { width: node.width || 200, height: node.height || 100 });
      });

      currentEdges.forEach((edge) => {
        g.setEdge(edge.source, edge.target);
      });

      dagre.layout(g);

      commit((draft) => {
        currentNodes.forEach((node) => {
          const dagreNode = g.node(node.id);
          const dn = draft.nodesById[node.id];
          if (dn) {
            dn.position = {
              x: dagreNode.x - (node.width || 200) / 2,
              y: dagreNode.y - (node.height || 100) / 2,
            };
          }
        });
      }, { description: "Auto layout" });
    },
    [nodes, edges]
  );

  return { autoLayout };
};