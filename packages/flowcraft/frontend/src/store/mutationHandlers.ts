import { create as createProto, type MessageShape, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { type Edge } from "@xyflow/react";

import { NodeKind } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  GraphMutationSchema,
  PathUpdate_UpdateType,
} from "@/generated/flowcraft/v1/core/service_pb";
import { assertNever } from "@/lib/utils";
import { type NodeId, type AppNode, AppNodeType, type DynamicNodeData } from "@/types";

const KIND_TO_NODE_TYPE: Record<NodeKind, AppNodeType> = {
  [NodeKind.DYNAMIC]: AppNodeType.DYNAMIC,
  [NodeKind.GROUP]: AppNodeType.GROUP,
  [NodeKind.PROCESS]: AppNodeType.PROCESSING,
  [NodeKind.NOTE]: AppNodeType.DYNAMIC,
  [NodeKind.UNSPECIFIED]: AppNodeType.DYNAMIC,
};

function setByPath(obj: Record<string, unknown>, path: string, value: unknown, merge = false) {
  const parts = path.split(".");
  let current = obj;
  for (let i = 0; i < parts.length - 1; i++) {
    const part = parts[i];
    if (!part) continue;
    if (!(part in current) || typeof current[part] !== "object") {
      current[part] = {};
    }
    current = current[part] as Record<string, unknown>;
  }
  const lastPart = parts[parts.length - 1];
  if (!lastPart) return;
  if (merge && typeof value === "object" && value !== null) {
    current[lastPart] = { ...(current[lastPart] as object), ...value };
  } else {
    current[lastPart] = value;
  }
}

export const handleGraphMutation = (
  mutInput: GraphMutation,
  currentNodes: AppNode[],
  currentEdges: Edge[],
): { edges: Edge[]; nodes: AppNode[] } => {
  const mut = createProto(GraphMutationSchema, mutInput);
  const op = mut.operation;
  let nodes = [...currentNodes];
  let edges = [...currentEdges];

  if (!op.case) return { edges, nodes };

  switch (op.case) {
    case "addEdge":
      if (op.value.edge) {
        const e = op.value.edge;
        edges = [
          ...edges.filter((ex) => ex.id !== e.edgeId),
          {
            data: e.metadata,
            id: e.edgeId,
            source: e.sourceNodeId,
            sourceHandle: e.sourceHandle || undefined,
            target: e.targetNodeId,
            targetHandle: e.targetHandle || undefined,
          },
        ];
      }
      break;

    case "addNode":
      if (op.value.node) {
        const n = op.value.node;
        const appData = (n.state ?? createProto(NodeDataSchema, { displayName: "New Node" })) as DynamicNodeData;
        appData.templateId = n.templateId as any;

        const newNode: AppNode = {
          data: appData,
          id: n.nodeId as NodeId,
          measured: n.presentation?.width ? { height: n.presentation.height, width: n.presentation.width } : undefined,
          parentId: n.presentation?.parentId || undefined,
          position: {
            x: n.presentation?.position?.x ?? 0,
            y: n.presentation?.position?.y ?? 0,
          },
          selected: n.isSelected,
          type: KIND_TO_NODE_TYPE[n.nodeKind] ?? AppNodeType.DYNAMIC,
        };
        nodes = [...nodes.filter((ex) => ex.id !== newNode.id), newNode];
      }
      break;

    case "clearGraph":
      nodes = [];
      edges = [];
      break;

    case "pathUpdate": {
      const { path, targetId, type, value } = op.value;
      const idx = nodes.findIndex((n) => n.id === targetId);
      if (idx !== -1 && value) {
        const val = toJson(ValueSchema, value);
        const existingNode = nodes[idx];
        if (!existingNode) break;
        const updatedData = createProto(
          NodeDataSchema,
          existingNode.data as unknown as MessageShape<typeof NodeDataSchema>,
        );
        // We still need 'as any' for recursive dynamic property access,
        // but we've narrowed the scope.
        setByPath(updatedData as unknown as Record<string, unknown>, path, val, type === PathUpdate_UpdateType.MERGE);
        nodes[idx] = {
          ...existingNode,
          data: updatedData as DynamicNodeData,
        };
      }
      break;
    }

    case "removeEdge":
      edges = edges.filter((e) => e.id !== op.value.id);
      break;

    case "removeNode":
      nodes = nodes.filter((n) => n.id !== op.value.id);
      edges = edges.filter((e) => e.source !== op.value.id && e.target !== op.value.id);
      break;

    case "updateNode": {
      const val = op.value;
      const idx = nodes.findIndex((n) => n.id === val.id);
      if (idx !== -1) {
        const existing = nodes[idx];
        if (!existing) break;
        const updated = { ...existing };
        if (val.presentation) {
          if (val.presentation.position) updated.position = val.presentation.position;
          if (val.presentation.width)
            updated.measured = {
              height: val.presentation.height,
              width: val.presentation.width,
            };
          updated.parentId = val.presentation.parentId || undefined;
        }
        if (val.data) {
          updated.data = createProto(NodeDataSchema, {
            ...(existing.data as unknown as MessageShape<typeof NodeDataSchema>),
            ...(val.data as unknown as MessageShape<typeof NodeDataSchema>),
          }) as DynamicNodeData;
        }
        nodes[idx] = updated;
      }
      break;
    }

    default:
      assertNever(op);
  }
  return { edges, nodes };
};
