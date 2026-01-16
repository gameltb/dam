import { create, type MessageShape } from "@bufbuild/protobuf";
import { type Edge } from "@xyflow/react";

import { NodeKind, PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import {
  EdgeSchema,
  NodeDataSchema,
  NodeSchema,
  type Edge as ProtoEdge,
  type Node as ProtoNode,
  type NodeData as ProtoNodeData,
} from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, AppNodeType, type DynamicNodeData } from "@/types";

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

  // Defense against non-primitive values in enum fields that can cause serialization errors
  const cleanedData = { ...data };
  if (Array.isArray(cleanedData.availableModes)) {
    cleanedData.availableModes = cleanedData.availableModes.map((m: any) => {
      if (typeof m === "object" && m !== null) {
        return typeof m.mode === "number" ? m.mode : (Number(m) || 0);
      }
      return typeof m === "number" ? m : (Number(m) || 0);
    });
  }
  if (typeof cleanedData.activeMode === "object" && cleanedData.activeMode !== null) {
    const m = cleanedData.activeMode as any;
    cleanedData.activeMode = typeof m.mode === "number" ? m.mode : (Number(m) || 0);
  }

  return create(NodeDataSchema, cleanedData as unknown as MessageShape<typeof NodeDataSchema>);
}

/**
 * Converts a client AppNode to a Protobuf Node message.
 */
export function appNodeToProto(node: AppNode): ProtoNode {
  const presentation = create(PresentationSchema, {
    height: node.measured?.height ?? 0,
    isInitialized: true,
    parentId: node.parentId ?? "",
    position: node.position,
    width: node.measured?.width ?? 0,
  });

  const nodeKind =
    node.type === AppNodeType.GROUP
      ? NodeKind.GROUP
      : node.type === AppNodeType.PROCESSING
        ? NodeKind.PROCESS
        : NodeKind.DYNAMIC;

  return create(NodeSchema, {
    isSelected: !!node.selected,
    nodeId: node.id,
    nodeKind,
    presentation,
    state: node.type === AppNodeType.DYNAMIC ? appNodeDataToProto(node.data as DynamicNodeData) : undefined,
    templateId: (node.data.templateId as string) ?? "unknown",
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
