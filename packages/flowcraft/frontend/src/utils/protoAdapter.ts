import { create } from "@bufbuild/protobuf";
import {
  type Node,
  type NodeData,
  NodeSchema,
  NodeDataSchema,
  WidgetSchema,
  type Widget,
  EdgeSchema,
  type Edge as ProtoEdge,
} from "../generated/core/node_pb";
import { type GraphSnapshot } from "../generated/core/service_pb";
import type { AppNode, DynamicNodeData, WidgetDef } from "../types";
import type { Edge } from "@xyflow/react";

/**
 * Adapter functions to convert between Protobuf generated types and internal React Flow types.
 */

// --- To Proto ---

export function toProtoNode(node: AppNode): Node {
  const data = node.data;
  let protoData: NodeData | undefined;

  if (node.type === "dynamic" || node.type === "groupNode") {
    const dynData = data as DynamicNodeData;
    const widgets: Widget[] = (dynData.widgets ?? []).map((w) =>
      create(WidgetSchema, {
        ...w,
        valueJson: JSON.stringify(w.value),
        options: w.options?.map((o) => ({
          label: o.label,
          value: String(o.value),
        })),
      }),
    );

    protoData = create(NodeDataSchema, {
      label: dynData.label,
      availableModes: dynData.modes,
      activeMode: dynData.activeMode,
      media: dynData.media
        ? {
            ...dynData.media,
            aspectRatio: dynData.media.aspectRatio ?? 0,
          }
        : undefined,
      widgets,
      inputPorts: dynData.inputPorts ?? [],
      outputPorts: dynData.outputPorts ?? [],
      metadata: {},
    });
  }

  return create(NodeSchema, {
    id: node.id,
    type: node.type,
    position: node.position,
    width: node.measured?.width ?? 0,
    height: node.measured?.height ?? 0,
    selected: node.selected,
    parentId: node.parentId ?? "",
    data: protoData,
  });
}

export function toProtoEdge(edge: Edge): ProtoEdge {
  const metadata: Record<string, string> = {};
  if (edge.data) {
    Object.entries(edge.data).forEach(([k, v]) => {
      if (typeof v === "string") metadata[k] = v;
    });
  }

  return create(EdgeSchema, {
    id: edge.id,
    source: edge.source,
    target: edge.target,
    sourceHandle: edge.sourceHandle ?? "",
    targetHandle: edge.targetHandle ?? "",
    metadata,
  });
}

// --- From Proto ---

export function fromProtoGraph(protoGraph: GraphSnapshot): {
  nodes: AppNode[];
  edges: Edge[];
} {
  const nodes: AppNode[] = protoGraph.nodes.map((n) => fromProtoNode(n));
  const edges: Edge[] = protoGraph.edges.map(fromProtoEdge);

  return { nodes, edges };
}

export function fromProtoEdge(e: ProtoEdge): Edge {
  return {
    id: e.id,
    source: e.source,
    target: e.target,
    sourceHandle: e.sourceHandle || undefined,
    targetHandle: e.targetHandle || undefined,
    data: (e.metadata as Record<string, unknown> | undefined) ?? {},
  };
}

export function fromProtoNode(n: Node): AppNode {
  const rawType = n.type || "dynamic";
  const isStandardSpecial = rawType === "groupNode" || rawType === "processing";
  const reactFlowType = isStandardSpecial ? rawType : "dynamic";

  const protoData = n.data;
  let appData: DynamicNodeData;

  if (protoData) {
    appData = fromProtoNodeData(protoData);
  } else {
    // Fallback for missing data
    appData = {
      label: "Unknown",
      modes: [],
      widgets: [],
    };
  }

  if (!isStandardSpecial && reactFlowType === "dynamic") {
    appData.typeId = rawType;
  }

  const node: AppNode = {
    id: n.id,
    type: reactFlowType,
    position: { x: n.position?.x ?? 0, y: n.position?.y ?? 0 },
    selected: n.selected,
    parentId: n.parentId || undefined,
    data: appData,
  } as AppNode;

  // Restore dimensions
  if (n.width && n.height) {
    node.measured = { width: n.width, height: n.height };
    node.style = { width: n.width, height: n.height };
  }

  return node;
}

export function fromProtoNodeData(protoData: NodeData): DynamicNodeData {
  const widgets: WidgetDef[] = protoData.widgets.map((w) => {
    let value: unknown = null;
    try {
      value = JSON.parse(w.valueJson);
    } catch {
      value = null;
    }
    return {
      id: w.id,
      type: w.type,
      label: w.label,
      value,
      options: w.options.map((o) => ({
        label: o.label,
        value: o.value,
      })),
      config: w.config as unknown as Record<string, unknown>,
      inputPortId: w.inputPortId,
    };
  });

  return {
    label: protoData.label,
    modes: protoData.availableModes,
    activeMode: protoData.activeMode,
    media: protoData.media
      ? {
          type: protoData.media.type,
          url: protoData.media.url,
          content: protoData.media.content,
          galleryUrls: protoData.media.galleryUrls,
          aspectRatio: protoData.media.aspectRatio,
        }
      : undefined,
    widgets,
    inputPorts: protoData.inputPorts,
    outputPorts: protoData.outputPorts,
  };
}

// Minimal compatibility shim
export function toProtoNodeData(data?: DynamicNodeData): NodeData {
  if (!data) {
    return create(NodeDataSchema, {
      label: "",
      availableModes: [],
      widgets: [],
      inputPorts: [],
      outputPorts: [],
      metadata: {},
    });
  }
  const widgets: Widget[] = (data.widgets ?? []).map((w) =>
    create(WidgetSchema, {
      ...w,
      valueJson: JSON.stringify(w.value),
      options: w.options?.map((o) => ({
        label: o.label,
        value: o.value as string,
      })),
    }),
  );

  return create(NodeDataSchema, {
    label: data.label,
    availableModes: data.modes,
    activeMode: data.activeMode,
    media: data.media
      ? {
          ...data.media,
          aspectRatio: data.media.aspectRatio ?? 0,
        }
      : undefined,
    widgets,
    inputPorts: data.inputPorts ?? [],
    outputPorts: data.outputPorts ?? [],
    metadata: {},
  });
}
