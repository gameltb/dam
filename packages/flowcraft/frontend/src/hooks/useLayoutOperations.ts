import type { Edge as RFEdge } from "@xyflow/react";

import { create } from "@bufbuild/protobuf";
import dagre from "dagre";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation, GraphMutationSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { type MutationContext } from "@/store/types";
import { type AppNode, AppNodeType, type DynamicNodeData } from "@/types";
import { appNodeDataToProto } from "@/utils/nodeProtoUtils";

export const useLayoutOperations = (
  nodes: AppNode[],
  edges: RFEdge[],
  applyMutations: (mutations: GraphMutation[], context?: MutationContext) => void,
) => {
  const autoLayout = useCallback(() => {
    const layoutSubgraph = (parentNodes: AppNode[], parentEdges: RFEdge[]) => {
      const g = new dagre.graphlib.Graph();
      g.setGraph({ nodesep: 50, rankdir: "LR", ranksep: 100 });
      g.setDefaultEdgeLabel(() => ({}));

      parentNodes.forEach((node) => {
        g.setNode(node.id, {
          height: node.measured?.height ?? 200,
          width: node.measured?.width ?? 300,
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
          height,
          isInitialized: true,
          parentId: node.parentId ?? "",
          position: { x, y },
          width,
        });

        mutations.push(
          create(GraphMutationSchema, {
            operation: {
              case: "updateNode",
              value: {
                data: appNodeDataToProto(node.data as DynamicNodeData),
                id: node.id,
                presentation,
              },
            },
          }),
        );

        // If this is a group, layout its children
        if (node.type === AppNodeType.GROUP) {
          const children = nodes.filter((n) => n.parentId === node.id);
          const childEdges = edges.filter(
            (e) => children.some((c) => c.id === e.source) && children.some((c) => c.id === e.target),
          );

          if (children.length > 0) {
            const { boundingBox, mutations: childMutations } = layoutSubgraph(children, childEdges);
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
              if (m.operation.case === "updateNode" && m.operation.value.presentation?.position) {
                m.operation.value.presentation.position.x += offsetX;
                m.operation.value.presentation.position.y += offsetY;
              }
            });

            // Update the group's own size in the mutation we already created
            const groupMutation = mutations.find(
              (m) => m.operation.case === "updateNode" && m.operation.value.id === node.id,
            );
            if (groupMutation?.operation.case === "updateNode" && groupMutation.operation.value.presentation) {
              groupMutation.operation.value.presentation.width = newWidth;
              groupMutation.operation.value.presentation.height = newHeight;
            }
          }
        }
      });

      // Calculate bounding box of this subgraph
      let maxX = -Infinity,
        maxY = -Infinity,
        minX = Infinity,
        minY = Infinity;
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
        boundingBox: {
          height: maxY - minY,
          width: maxX - minX,
          x: minX,
          y: minY,
        },
        mutations,
      };
    };

    const rootNodes = nodes.filter((n) => !n.parentId);
    const rootEdges = edges.filter(
      (e) => rootNodes.some((n) => n.id === e.source) && rootNodes.some((n) => n.id === e.target),
    );

    const { mutations } = layoutSubgraph(rootNodes, rootEdges);
    applyMutations(mutations);
  }, [nodes, edges, applyMutations]);

  const groupSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    if (selectedNodes.length < 1) return;

    // 1. Calculate bounding box
    let maxX = -Infinity,
      maxY = -Infinity,
      minX = Infinity,
      minY = Infinity;

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
      presentation: create(PresentationSchema, {
        height: groupH,
        isInitialized: true,
        position: { x: groupX, y: groupY },
        width: groupW,
      }),
      state: appNodeDataToProto({
        availableModes: [],
        displayName: "New Group",
      } as any),
      templateId: "group",
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
                isInitialized: true,
                parentId: groupId,
                position: {
                  x: node.position.x - groupX,
                  y: node.position.y - groupY,
                },
              }),
            },
          },
        }),
      );
    });

    applyMutations(mutations, {
      description: `Group ${String(selectedNodes.length)} nodes`,
      taskId: uuidv4(),
    });
  }, [nodes, applyMutations]);

  return { autoLayout, groupSelected };
};
