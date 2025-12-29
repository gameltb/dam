import { create } from "@bufbuild/protobuf";
import {
  type Node,
  type NodeData,
  NodeSchema,
  NodeDataSchema,
} from "../generated/core/node_pb";
import { type GraphSnapshot } from "../generated/core/service_pb";
import type { AppNode, DynamicNodeData } from "../types";
import type { Edge } from "@xyflow/react";

/**
 * Adapter functions to convert between Protobuf generated types and internal React Flow types.
 */

// --- To Proto ---

export function toProtoNode(node: AppNode): Node {
  return create(NodeSchema, {
    id: node.id,
    type: node.type,
    position: node.position,
    width: node.measured?.width ?? 0,
    height: node.measured?.height ?? 0,
    selected: node.selected,
    parentId: node.parentId ?? "",
    // Passthrough data generically
    data: node.data as unknown as NodeData,
  });
}

// --- From Proto ---

export function fromProtoGraph(protoGraph: GraphSnapshot): {
  nodes: AppNode[];
  edges: Edge[];
} {
  const nodes: AppNode[] = protoGraph.nodes.map((n) => fromProtoNode(n));
  const edges: Edge[] = protoGraph.edges.map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle || undefined,
    targetHandle: e.targetHandle || undefined,
    data: (e.metadata as Record<string, unknown> | undefined) ?? {},
  }));

  return { nodes, edges };
}

export function fromProtoNode(n: Node): AppNode {
  const rawType = n.type || "dynamic";
  const isStandardSpecial = rawType === "groupNode" || rawType === "processing";
  const reactFlowType = isStandardSpecial ? rawType : "dynamic";

  const node: AppNode = {
    id: n.id,
    type: reactFlowType,
    position: { x: n.position?.x ?? 0, y: n.position?.y ?? 0 },
    selected: n.selected,
    parentId: n.parentId || undefined,
    // Generic passthrough for data
    data: n.data
      ? (n.data as unknown as DynamicNodeData)
      : (create(NodeDataSchema, {}) as unknown as DynamicNodeData),
  } as AppNode;

  if (!isStandardSpecial && node.type === "dynamic") {
    node.data.typeId = rawType;
  }

  // Restore dimensions
  if (n.width && n.height) {
    node.measured = { width: n.width, height: n.height };
    node.style = { width: n.width, height: n.height };
  }

  return node;
}

// Minimal compatibility shim for existing imports if any
export function toProtoNodeData(data: DynamicNodeData): unknown {
  return data;
}
export function fromProtoNodeData(data: unknown): DynamicNodeData {
  return data as DynamicNodeData;
}
