import type { Edge as RFEdge } from "@xyflow/react";

import { create as createProto } from "@bufbuild/protobuf";
import dagre from "dagre";
import { useCallback } from "react";
import { v4 as uuidv4 } from "uuid";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { AddNodeRequestSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { useFlowStore } from "@/store/flowStore";
import { type MutationContext, type MutationInput } from "@/store/types";
import { type AppNode, AppNodeType } from "@/types";
import { appNodeDataToProto } from "@/utils/nodeProtoUtils";

export const useLayoutOperations = (
  nodes: AppNode[],
  edges: RFEdge[],
  applyMutations: (mutations: MutationInput[], context?: MutationContext) => void,
) => {
  const nodeDraft = useFlowStore((s) => s.nodeDraft);

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

      parentNodes.forEach((node) => {
        const nodeWithPos = g.node(node.id);
        const width = node.measured?.width ?? 300;
        const height = node.measured?.height ?? 200;

        const x = nodeWithPos.x - width / 2;
        const y = nodeWithPos.y - height / 2;

        // 使用 nodeDraft 进行精确路径更新，替代 UpdateNodeRequest
        const res = nodeDraft(node);
        if (res.ok) {
          const draft = res.value;
          draft.position = { x, y };

          if (node.type === AppNodeType.GROUP) {
            const children = nodes.filter((n) => n.parentId === node.id);
            const childEdges = edges.filter(
              (e) => children.some((c) => c.id === e.source) && children.some((c) => c.id === e.target),
            );

            if (children.length > 0) {
              const { boundingBox } = layoutSubgraph(children, childEdges);

              const padding = 60;
              const newWidth = boundingBox.width + padding * 2;
              const newHeight = boundingBox.height + padding * 2;

              const offsetX = padding - boundingBox.x;
              const offsetY = padding - boundingBox.y;

              children.forEach((child) => {
                const childRes = nodeDraft(child);
                if (childRes.ok) {
                  childRes.value.position.x += offsetX;
                  childRes.value.position.y += offsetY;
                }
              });

              draft.width = newWidth;
              draft.height = newHeight;
            }
          }
        }
      });

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
      };
    };

    const rootNodes = nodes.filter((n) => !n.parentId);
    const rootEdges = edges.filter(
      (e) => rootNodes.some((n) => n.id === e.source) && rootNodes.some((n) => n.id === e.target),
    );

    layoutSubgraph(rootNodes, rootEdges);
  }, [nodes, edges, nodeDraft]);

  const groupSelected = useCallback(() => {
    const selectedNodes = nodes.filter((n) => n.selected && !n.parentId);
    if (selectedNodes.length < 1) return;

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
    const groupNode = createProto(NodeSchema, {
      nodeId: groupId,
      nodeKind: NodeKind.GROUP,
      presentation: createProto(PresentationSchema, {
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

    applyMutations([createProto(AddNodeRequestSchema, { node: groupNode })]);

    selectedNodes.forEach((node) => {
      // 使用 nodeDraft 替代重亲化指令中的局部更新部分
      const res = nodeDraft(node);
      if (res.ok) {
        const draft = res.value;
        draft.parentId = groupId;
        draft.position = {
          x: node.position.x - groupX,
          y: node.position.y - groupY,
        };
      }
    });
  }, [nodes, applyMutations, nodeDraft]);

  return { autoLayout, groupSelected };
};
