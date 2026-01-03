import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, DynamicNodeData } from "../types";
import type { Edge as RFEdge } from "@xyflow/react";
import dagre from "dagre";
import {
  GraphMutationSchema,
  type GraphMutation,
} from "../generated/flowcraft/v1/service_pb";
import { NodeSchema } from "../generated/flowcraft/v1/node_pb";
import {
  PresentationSchema,
  NodeKind,
} from "../generated/flowcraft/v1/base_pb";
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
    const layoutSubgraph = (parentNodes: AppNode[], parentEdges: RFEdge[]) => {
      const g = new dagre.graphlib.Graph();
      g.setGraph({ rankdir: "LR", nodesep: 50, ranksep: 100 });
      g.setDefaultEdgeLabel(() => ({}));

      parentNodes.forEach((node) => {
        g.setNode(node.id, {
          width: node.measured?.width ?? 300,
          height: node.measured?.height ?? 200,
        });
      });

      parentEdges.forEach((edge) => {
        g.setEdge(edge.source, edge.target);
      });

      dagre.layout(g);

      const mutations: GraphMutation[] = [];

      parentNodes.forEach((node) => {
        const nodeWithPos = g.node(node.id);
        const width = node.measured?.width ?? 300;
        const height = node.measured?.height ?? 200;

        // Calculate position relative to parent (dagre gives center)
        const x = nodeWithPos.x - width / 2;
        const y = nodeWithPos.y - height / 2;

        const presentation = create(PresentationSchema, {
          position: { x, y },
          width,
          height,
          isInitialized: true,
          parentId: node.parentId ?? "",
        });

        mutations.push(
          create(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                id: node.id,
                presentation,
                data: toProtoNodeData(node.data as DynamicNodeData),
              },
            },
          }),
        );

        // If this is a group, layout its children
        if (node.type === "groupNode") {
          const children = nodes.filter((n) => n.parentId === node.id);
          const childEdges = edges.filter(
            (e) =>
              children.some((c) => c.id === e.source) &&
              children.some((c) => c.id === e.target),
          );

          if (children.length > 0) {
            const { mutations: childMutations, boundingBox } = layoutSubgraph(
              children,
              childEdges,
            );
            mutations.push(...childMutations);

            // Update group size to fit children with padding
            const padding = 60;
            const newWidth = boundingBox.width + padding * 2;
            const newHeight = boundingBox.height + padding * 2;

            // Shift children so they are centered/padded correctly
            // layoutSubgraph gives them relative positions starting from 0,0 typically
            // but we want to adjust based on bounding box if it's not at 0,0
            const offsetX = padding - boundingBox.x;
            const offsetY = padding - boundingBox.y;

            childMutations.forEach((m) => {
              if (
                m.operation.case === "updateNode" &&
                m.operation.value.presentation?.position
              ) {
                m.operation.value.presentation.position.x += offsetX;
                m.operation.value.presentation.position.y += offsetY;
              }
            });

            // Update the group's own size in the mutation we already created
            const groupMutation = mutations.find(
              (m) =>
                m.operation.case === "updateNode" &&
                m.operation.value.id === node.id,
            );
            if (
              groupMutation?.operation.case === "updateNode" &&
              groupMutation.operation.value.presentation
            ) {
              groupMutation.operation.value.presentation.width = newWidth;
              groupMutation.operation.value.presentation.height = newHeight;
            }
          }
        }
      });

      // Calculate bounding box of this subgraph
      let minX = Infinity,
        minY = Infinity,
        maxX = -Infinity,
        maxY = -Infinity;
      parentNodes.forEach((n) => {
        const nodePos = g.node(n.id);
        const w = n.measured?.width ?? 300;
        const h = n.measured?.height ?? 200;
        minX = Math.min(minX, nodePos.x - w / 2);
        minY = Math.min(minY, nodePos.y - h / 2);
        maxX = Math.max(maxX, nodePos.x + w / 2);
        maxY = Math.max(maxY, nodePos.y + h / 2);
      });

      return {
        mutations,
        boundingBox: {
          x: minX,
          y: minY,
          width: maxX - minX,
          height: maxY - minY,
        },
      };
    };

    const rootNodes = nodes.filter((n) => !n.parentId);
    const rootEdges = edges.filter(
      (e) =>
        rootNodes.some((n) => n.id === e.source) &&
        rootNodes.some((n) => n.id === e.target),
    );

    const { mutations } = layoutSubgraph(rootNodes, rootEdges);
    applyMutations(mutations);
  }, [nodes, edges, applyMutations]);

  const groupSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    if (selectedNodes.length < 1) return;

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
      nodeId: groupId,
      nodeKind: NodeKind.GROUP,
      templateId: "group",
      presentation: create(PresentationSchema, {
        position: { x: groupX, y: groupY },
        width: groupW,
        height: groupH,
        isInitialized: true,
      }),
      state: toProtoNodeData({ label: "New Group", modes: [] }),
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
              presentation: create(PresentationSchema, {
                parentId: groupId,
                position: {
                  x: node.position.x - groupX,
                  y: node.position.y - groupY,
                },
                isInitialized: true,
              }),
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
