import { useCallback } from "react";
import { useFlowStore } from "../store/flowStore";
import { useUiStore } from "../store/uiStore";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { XYPosition, Edge } from "@xyflow/react";
import dagre from "dagre";
import { flowcraft_proto } from "../generated/flowcraft_proto";

interface GraphOpsProps {
  clientVersion: number;
}

export const useGraphOperations = ({ clientVersion }: GraphOpsProps) => {
  const store = useFlowStore();

  const addNode = useCallback(
    (
      type: string,
      data: Partial<AppNode["data"]>,
      position: XYPosition,
      typeId?: string,
    ) => {
      const dynamicData = data as DynamicNodeData | undefined;
      const newNode: AppNode = {
        id: uuidv4(),
        type,
        position,
        data: {
          ...data,
          typeId: typeId ?? dynamicData?.typeId,
        },
      } as AppNode;
      store.applyMutations([
        { addNode: { node: newNode as unknown as flowcraft_proto.v1.INode } },
      ]);
    },
    [store],
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      store.applyMutations([{ removeNode: { id: nodeId } }]);
    },
    [store],
  );

  const deleteEdge = useCallback(
    (edgeId: string) => {
      store.applyMutations([{ removeEdge: { id: edgeId } }]);
    },
    [store],
  );

  // --- Copy / Paste Logic ---

  const copySelected = useCallback(() => {
    const selectedNodes = store.nodes.filter((n) => n.selected);
    const selectedEdges = store.edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    if (selectedNodes.length > 0) {
      useUiStore.getState().setClipboard({
        nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
        edges: JSON.parse(JSON.stringify(selectedEdges)) as Edge[],
      });
    }
  }, [store]);

  const paste = useCallback(
    (targetPosition?: XYPosition) => {
      const { applyMutations } = store;
      const clipboard = useUiStore.getState().clipboard;
      if (!clipboard) return;

      const idMap: Record<string, string> = {};
      const firstNodePos = clipboard.nodes[0]?.position ?? { x: 0, y: 0 };
      const offset = targetPosition
        ? {
            x: targetPosition.x - firstNodePos.x,
            y: targetPosition.y - firstNodePos.y,
          }
        : { x: 40, y: 40 };

      const newNodes = clipboard.nodes.map((node) => {
        const newId = uuidv4();
        idMap[node.id] = newId;
        return {
          ...node,
          id: newId,
          position: {
            x: node.position.x + offset.x,
            y: node.position.y + offset.y,
          },
          selected: true,
        };
      });

      const newEdges = clipboard.edges.map((edge) => ({
        ...edge,
        id: uuidv4(),
        source: idMap[edge.source] ?? edge.source,
        target: idMap[edge.target] ?? edge.target,
        selected: true,
      }));

      applyMutations(
        [
          {
            addSubgraph: {
              nodes: newNodes as unknown as flowcraft_proto.v1.INode[],
              edges: newEdges as unknown as flowcraft_proto.v1.IEdge[],
            },
          },
        ],
        { taskId: uuidv4(), description: "Paste Subgraph" },
      );
    },
    [store],
  );

  const duplicateSelected = useCallback(() => {
    const { nodes, edges, applyMutations } = store;
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedEdges = edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    const tempClipboard: { nodes: AppNode[]; edges: Edge[] } = {
      nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
      edges: JSON.parse(JSON.stringify(selectedEdges)) as Edge[],
    };

    const idMap: Record<string, string> = {};
    const newNodes = tempClipboard.nodes.map((n: AppNode) => {
      const newId = uuidv4();
      idMap[n.id] = newId;
      return {
        ...n,
        id: newId,
        position: { x: n.position.x + 40, y: n.position.y + 40 },
        selected: true,
      };
    });
    const newEdges = tempClipboard.edges.map((e: Edge) => ({
      ...e,
      id: uuidv4(),
      source: idMap[e.source] ?? e.source,
      target: idMap[e.target] ?? e.target,
      selected: true,
    }));

    applyMutations(
      [
        {
          addSubgraph: {
            nodes: newNodes as unknown as flowcraft_proto.v1.INode[],
            edges: newEdges as unknown as flowcraft_proto.v1.IEdge[],
          },
        },
      ],
      { taskId: uuidv4(), description: "Duplicate Selected" },
    );
  }, [store]);

  // --- Auto Layout (Dagre) ---

  const autoLayout = useCallback(() => {
    const { nodes, edges, applyMutations } = store;
    const g = new dagre.graphlib.Graph();
    g.setGraph({ rankdir: "LR", nodesep: 50, ranksep: 100 });
    g.setDefaultEdgeLabel(() => ({}));

    nodes.forEach((node) => {
      g.setNode(node.id, {
        width: node.measured?.width ?? 300,
        height: node.measured?.height ?? 200,
      });
    });

    edges.forEach((edge) => {
      g.setEdge(edge.source, edge.target);
    });

    dagre.layout(g);

    const mutations: flowcraft_proto.v1.IGraphMutation[] = nodes.map((node) => {
      const nodeWithPos = g.node(node.id);
      const width = node.measured?.width ?? 300;
      const height = node.measured?.height ?? 200;
      return {
        updateNode: {
          id: node.id,
          position: {
            x: nodeWithPos.x - width / 2,
            y: nodeWithPos.y - height / 2,
          },
          width,
          height,
          data: node.data as flowcraft_proto.v1.INodeData,
        },
      };
    });

    applyMutations(mutations);
  }, [store]);

  const groupSelected = useCallback(() => {
    const { nodes, applyMutations } = store;
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    if (selectedNodes.length < 2) return;

    // 1. Calculate bounding box
    let minX = Infinity,
      minY = Infinity,
      maxX = -Infinity,
      maxY = -Infinity;

    selectedNodes.forEach((node) => {
      const { x, y } = node.position;
      const w = node.measured?.width ?? 200;
      const h = node.measured?.height ?? 150;
      minX = Math.min(minX, x);
      minY = Math.min(minY, y);
      maxX = Math.max(maxX, x + w);
      maxY = Math.max(maxY, y + h);
    });

    const padding = 40;
    const groupX = minX - padding;
    const groupY = minY - padding;
    const groupW = maxX - minX + padding * 2;
    const groupH = maxY - minY + padding * 2;

    const groupId = uuidv4();
    const groupNode: flowcraft_proto.v1.INode = {
      id: groupId,
      type: "groupNode",
      position: { x: groupX, y: groupY },
      width: groupW,
      height: groupH,
      data: { label: "New Group" } as flowcraft_proto.v1.INodeData,
    };

    // 2. Prepare mutations for grouping
    // We add the group node FIRST so it's ready to receive children
    const mutations: flowcraft_proto.v1.IGraphMutation[] = [
      { addNode: { node: groupNode } },
    ];

    selectedNodes.forEach((node) => {
      mutations.push({
        updateNode: {
          id: node.id,
          parentId: groupId,
          // Convert absolute to relative
          position: {
            x: node.position.x - groupX,
            y: node.position.y - groupY,
          },
        },
      });
    });

    applyMutations(mutations, {
      taskId: uuidv4(),
      description: `Group ${String(selectedNodes.length)} nodes`,
    });
  }, [store]);

  return {
    addNode,
    deleteNode,
    deleteEdge,
    copySelected,
    paste,
    duplicateSelected,
    autoLayout,
    groupSelected,
    clientVersion,
  };
};
