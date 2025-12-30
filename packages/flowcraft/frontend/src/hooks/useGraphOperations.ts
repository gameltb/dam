import { useCallback } from "react";
import { useFlowStore } from "../store/flowStore";
import { useUiStore } from "../store/uiStore";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { XYPosition, Edge as RFEdge } from "@xyflow/react";
import dagre from "dagre";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/core/service_pb";
import {
  type Node as ProtoNode,
  NodeSchema,
  EdgeSchema,
} from "../generated/core/node_pb";
import { create } from "@bufbuild/protobuf";
import { useShallow } from "zustand/react/shallow";

interface GraphOpsProps {
  clientVersion: number;
}

export const useGraphOperations = ({ clientVersion }: GraphOpsProps) => {
  const { nodes, edges, applyMutations } = useFlowStore(
    useShallow((state) => ({
      nodes: state.nodes,
      edges: state.edges,
      applyMutations: state.applyMutations,
    })),
  );

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
          label: "New Node",
          modes: [],
          ...data,
          typeId: typeId ?? dynamicData?.typeId,
        },
      } as AppNode;
      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "addNode",
            value: { node: newNode as unknown as ProtoNode },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  const deleteNode = useCallback(
    (nodeId: string) => {
      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "removeNode",
            value: { id: nodeId },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  const deleteEdge = useCallback(
    (edgeId: string) => {
      applyMutations([
        create(GraphMutationSchema, {
          operation: {
            case: "removeEdge",
            value: { id: edgeId },
          },
        }),
      ]);
    },
    [applyMutations],
  );

  // --- Copy / Paste Logic ---

  const copySelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    const selectedEdges = edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    if (selectedNodes.length > 0) {
      useUiStore.getState().setClipboard({
        nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
        edges: JSON.parse(JSON.stringify(selectedEdges)) as RFEdge[],
      });
    }
  }, [nodes, edges]);

  const paste = useCallback(
    (targetPosition?: XYPosition) => {
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
          create(GraphMutationSchema, {
            operation: {
              case: "addSubgraph",
              value: {
                nodes: newNodes as unknown as ProtoNode[],
                edges: newEdges.map((e) =>
                  create(EdgeSchema, {
                    id: e.id,
                    source: e.source,
                    target: e.target,
                    sourceHandle: e.sourceHandle ?? "",
                    targetHandle: e.targetHandle ?? "",
                  }),
                ),
              },
            },
          }),
        ],
        { taskId: uuidv4(), description: "Paste Subgraph" },
      );
    },
    [applyMutations],
  );

  const duplicateSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected);
    if (selectedNodes.length === 0) return;

    const selectedEdges = edges.filter((e) => {
      const isSourceSelected = selectedNodes.some((n) => n.id === e.source);
      const isTargetSelected = selectedNodes.some((n) => n.id === e.target);
      return isSourceSelected && isTargetSelected;
    });

    const tempClipboard: { nodes: AppNode[]; edges: RFEdge[] } = {
      nodes: JSON.parse(JSON.stringify(selectedNodes)) as AppNode[],
      edges: JSON.parse(JSON.stringify(selectedEdges)) as RFEdge[],
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
    const newEdges = tempClipboard.edges.map((e: RFEdge) => ({
      ...e,
      id: uuidv4(),
      source: idMap[e.source] ?? e.source,
      target: idMap[e.target] ?? e.target,
      selected: true,
    }));

    applyMutations(
      [
        create(GraphMutationSchema, {
          operation: {
            case: "addSubgraph",
            value: {
              nodes: newNodes as unknown as ProtoNode[],
              edges: newEdges.map((e) =>
                create(EdgeSchema, {
                  id: e.id,
                  source: e.source,
                  target: e.target,
                  sourceHandle: e.sourceHandle ?? "",
                  targetHandle: e.targetHandle ?? "",
                }),
              ),
            },
          },
        }),
      ],
      { taskId: uuidv4(), description: "Duplicate Selected" },
    );
  }, [nodes, edges, applyMutations]);

  // --- Auto Layout (Dagre) ---

  const autoLayout = useCallback(() => {
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

    const mutations: GraphMutation[] = nodes.map((node) => {
      const nodeWithPos = g.node(node.id);
      const width = node.measured?.width ?? 300;
      const height = node.measured?.height ?? 200;
      return create(GraphMutationSchema, {
        operation: {
          case: "updateNode",
          value: {
            id: node.id,
            position: {
              x: nodeWithPos.x - width / 2,
              y: nodeWithPos.y - height / 2,
            },
            width,
            height,
            data: node.data as any,
          },
        },
      });
    });

    applyMutations(mutations);
  }, [nodes, edges, applyMutations]);

  const groupSelected = useCallback(() => {
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
    const groupNode = create(NodeSchema, {
      id: groupId,
      type: "groupNode",
      position: { x: groupX, y: groupY },
      width: groupW,
      height: groupH,
      data: { label: "New Group", modes: [] } as any,
    });

    // 2. Prepare mutations for grouping
    const mutations: GraphMutation[] = [
      create(GraphMutationSchema, {
        operation: {
          case: "addNode",
          value: { node: groupNode },
        },
      }),
    ];

    selectedNodes.forEach((node) => {
      mutations.push(
        create(GraphMutationSchema, {
          operation: {
            case: "updateNode",
            value: {
              id: node.id,
              parentId: groupId,
              position: {
                x: node.position.x - groupX,
                y: node.position.y - groupY,
              },
            },
          },
        }),
      );
    });

    applyMutations(mutations, {
      taskId: uuidv4(),
      description: `Group ${String(selectedNodes.length)} nodes`,
    });
  }, [nodes, applyMutations]);

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

