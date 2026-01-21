import { create as createProto, toJson } from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { type Edge } from "@xyflow/react";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { NodeDataSchema } from "@/generated/flowcraft/v1/core/node_pb";
import {
  AddEdgeRequestSchema,
  AddNodeRequestSchema,
  ClearGraphRequestSchema,
  GraphMutationSchema,
  PathUpdateRequest_UpdateType,
  PathUpdateRequestSchema,
  RemoveEdgeRequestSchema,
  RemoveNodeRequestSchema,
  ReparentNodeRequestSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type AppNode, AppNodeType, type DynamicNodeData } from "@/types";
import { protoToAppEdge } from "@/utils/nodeProtoUtils";

import { type MutationInput } from "./types";

const KIND_TO_NODE_TYPE: Record<number, AppNodeType> = {
  [NodeKind.DYNAMIC]: AppNodeType.DYNAMIC,
  [NodeKind.GROUP]: AppNodeType.GROUP,
  [NodeKind.NOTE]: AppNodeType.DYNAMIC,
  [NodeKind.PROCESS]: AppNodeType.PROCESSING,
  [NodeKind.UNSPECIFIED]: AppNodeType.DYNAMIC,
};

export const handleMutation = (
  input: MutationInput,
  currentNodes: AppNode[],
  currentEdges: Edge[],
): { edges: Edge[]; nodes: AppNode[] } => {
  let nodes = [...currentNodes];
  let edges = [...currentEdges];

  switch (input.$typeName) {
    case AddEdgeRequestSchema.typeName:
      if (input.edge) {
        const e = protoToAppEdge(input.edge);
        edges = [...edges.filter((ex) => ex.id !== e.id), e];
      }
      break;

    case AddNodeRequestSchema.typeName:
      if (input.node) {
        const n = input.node;
        const newNode: AppNode = {
          data: createProto(NodeDataSchema, n.state) as unknown as DynamicNodeData,
          height: n.presentation?.height || 400,
          id: n.nodeId,
          parentId: n.presentation?.parentId || undefined,
          position: { x: n.presentation?.position?.x ?? 0, y: n.presentation?.position?.y ?? 0 },
          presentation: n.presentation ?? createProto(PresentationSchema, {}),
          selected: n.presentation?.isSelected,
          type: KIND_TO_NODE_TYPE[n.nodeKind] ?? AppNodeType.DYNAMIC,
          width: n.presentation?.width || 500,
        };
        nodes = [...nodes.filter((ex) => ex.id !== newNode.id), newNode];
      }
      break;

    case ClearGraphRequestSchema.typeName:
      return { edges: [], nodes: [] };

    case GraphMutationSchema.typeName:
      if (input.operation.case) return handleMutation(input.operation.value as any, nodes, edges);
      break;

    case PathUpdateRequestSchema.typeName: {
      const idx = nodes.findIndex((n) => n.id === input.targetId);
      if (idx !== -1) {
        const jsValue = input.value ? (toJson(ValueSchema, input.value) as any) : null;
        const pathParts = input.path.split(".");

        const applyPath = (obj: any, parts: string[], value: any): any => {
          if (parts.length === 0) return value;
          const [first, ...rest] = parts;
          if (!first) return value;

          if (input.type === PathUpdateRequest_UpdateType.DELETE) {
            const { [first]: _, ...newObj } = obj || {};
            return newObj;
          }
          return {
            ...(obj || {}),
            [first]: applyPath((obj || {})[first] || {}, rest, value),
          };
        };

        const node = nodes[idx]!;
        const updated = { ...node };

        if (pathParts[0] === "state" || pathParts[0] === "data") {
          updated.data = applyPath(updated.data, pathParts.slice(1), jsValue);
        } else if (pathParts[0] === "presentation") {
          updated.presentation = applyPath(updated.presentation, pathParts.slice(1), jsValue);

          const field = pathParts[1];
          if (field === "width" || field === "height") {
            const num = Number(jsValue);
            if (isNaN(num)) {
              throw new Error(`[Mutation] CRITICAL: Attempted to set ${field} to NaN for node ${node.id}`);
            }
            (updated as any)[field] = num;
            updated.measured = { ...updated.measured, [field]: num };
          } else if (field === "isSelected") {
            updated.selected = !!jsValue;
          } else if (field === "parentId") {
            updated.parentId = String(jsValue) || undefined;
          } else if (field === "position") {
            const px = Number(updated.presentation.position?.x);
            const py = Number(updated.presentation.position?.y);
            if (isNaN(px) || isNaN(py)) {
              throw new Error(`[Mutation] CRITICAL: Position resulted in NaN for node ${node.id}`);
            }
            updated.position = { x: px, y: py };
          }
        }

        nodes[idx] = updated;
      }
      break;
    }

    case RemoveEdgeRequestSchema.typeName:
      edges = edges.filter((e) => e.id !== input.id);
      break;

    case RemoveNodeRequestSchema.typeName:
      nodes = nodes.filter((n) => n.id !== input.id);
      edges = edges.filter((e) => e.source !== input.id && e.target !== input.id);
      break;

    case ReparentNodeRequestSchema.typeName: {
      const idx = nodes.findIndex((n) => n.id === input.nodeId);
      if (idx !== -1) {
        const node = nodes[idx]!;
        nodes[idx] = {
          ...node,
          parentId: input.newParentId || undefined,
          position: input.newPosition || node.position,
          presentation: {
            ...node.presentation,
            parentId: input.newParentId || "",
            position: input.newPosition || node.presentation.position,
          },
        };
      }
      break;
    }
  }

  return { edges, nodes };
};
