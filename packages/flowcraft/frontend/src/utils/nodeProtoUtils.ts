import { create } from "@bufbuild/protobuf";
import { type Edge } from "@xyflow/react";

import { NodeKind, PositionSchema, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import {
  EdgeSchema,
  NodeDataSchema,
  NodeSchema,
  type Edge as ProtoEdge,
  type Node as ProtoNode,
  type NodeData as ProtoNodeData,
  RenderMode,
} from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, AppNodeType, type DynamicNodeData, isDynamicNode } from "@/types";

import { getConstraintsForTemplate } from "./nodeRegistry";

const KIND_MAPPING: Record<string, NodeKind> = {
  [AppNodeType.DYNAMIC]: NodeKind.DYNAMIC,
  [AppNodeType.GROUP]: NodeKind.GROUP,
  [AppNodeType.PROCESSING]: NodeKind.PROCESS,
};

export function appEdgeToProto(edge: Edge): ProtoEdge {
  return create(EdgeSchema, {
    edgeId: edge.id,
    metadata: (edge.data as Record<string, string>) ?? {},
    sourceHandle: edge.sourceHandle ?? "",
    sourceNodeId: edge.source,
    targetHandle: edge.targetHandle ?? "",
    targetNodeId: edge.target,
  });
}

/**
 * Ensures node data is a valid Protobuf message.
 */
export function appNodeDataToProto(data?: DynamicNodeData): ProtoNodeData {
  if (!data) return create(NodeDataSchema, {});

  // Ensure enum values are correct numbers
  const activeMode =
    typeof data.activeMode === "string"
      ? (RenderMode[data.activeMode as keyof typeof RenderMode] as unknown as RenderMode)
      : data.activeMode;

  const availableModes = (data.availableModes || []).map((m) =>
    typeof m === "string" ? (RenderMode[m as keyof typeof RenderMode] as unknown as RenderMode) : m,
  );

  return create(NodeDataSchema, {
    ...data,
    activeMode,
    availableModes,
  });
}

/**
 * Converts a client AppNode to a Protobuf Node message.
 */
export function appNodeToProto(node: AppNode): ProtoNode {
  const templateId = isDynamicNode(node) ? node.data.templateId : undefined;

  if (!templateId) {
    throw new Error(`[Serialization] Missing templateId for node ${node.id}. Fail-fast to prevent DB corruption.`);
  }

  const constraints = getConstraintsForTemplate(templateId);

  const getDimension = (dim: "height" | "width"): number => {
    let val = 0;
    if (node.measured?.[dim] !== undefined && node.measured[dim] > 0) val = node.measured[dim]!;
    else if (node[dim] !== undefined && node[dim] > 0) val = node[dim];
    else {
      const styleVal = node.style?.[dim];
      if (typeof styleVal === "number") val = styleVal;
      else if (typeof styleVal === "string") {
        const parsed = parseFloat(styleVal);
        val = isNaN(parsed) ? 0 : parsed;
      }
    }

    const minVal = dim === "width" ? constraints.minWidth : constraints.minHeight;
    return minVal ? Math.max(val, minVal) : val;
  };

  const presentation = create(PresentationSchema, {
    height: getDimension("height"),
    isInitialized: true,
    isSelected: !!node.selected,
    parentId: node.parentId ?? "",
    position: create(PositionSchema, {
      x: isNaN(node.position.x) ? 0 : node.position.x,
      y: isNaN(node.position.y) ? 0 : node.position.y,
    }),
    width: getDimension("width"),
  });

  const nodeKind = KIND_MAPPING[node.type || ""] ?? NodeKind.DYNAMIC;

  return create(NodeSchema, {
    nodeId: node.id,
    nodeKind,
    presentation,
    state: isDynamicNode(node) ? appNodeDataToProto(node.data) : undefined,
    templateId,
  });
}

export function protoToAppEdge(e: ProtoEdge): Edge {
  return {
    data: e.metadata,
    id: e.edgeId,
    source: e.sourceNodeId,
    sourceHandle: e.sourceHandle || undefined,
    target: e.targetNodeId,
    targetHandle: e.targetHandle || undefined,
  };
}
