import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { Edge as RFEdge } from "@xyflow/react";
import dagre from "dagre";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/core/service_pb";
import { NodeSchema } from "../generated/core/node_pb";
import { create } from "@bufbuild/protobuf";
import { toProtoNodeData } from "../utils/protoAdapter";
import { type MutationContext } from "../store/flowStore";

export const useLayoutOperations = (
  nodes: AppNode[],
  edges: RFEdge[],
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void,
) => {
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
            data: toProtoNodeData(node.data as DynamicNodeData),
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
      data: toProtoNodeData({ label: "New Group", modes: [] }),
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

  return { autoLayout, groupSelected };
};
