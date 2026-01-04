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
  type Port,
  type PortType,
} from "../generated/flowcraft/v1/core/node_pb";
import {
  PresentationSchema,
  PortMainType,
  NodeKind,
} from "../generated/flowcraft/v1/core/base_pb";
import { type GraphSnapshot } from "../generated/flowcraft/v1/core/service_pb";
import type { AppNode, DynamicNodeData, WidgetDef } from "../types";
import type { Edge } from "@xyflow/react";

/**
 * Mappings for NodeKind and PortMainType enums.
 */
const NODE_TYPE_TO_KIND: Record<string, NodeKind> = {
  dynamic: NodeKind.DYNAMIC,
  groupNode: NodeKind.GROUP,
  processing: NodeKind.PROCESS,
};

const KIND_TO_NODE_TYPE: Record<number, string> = {
  [NodeKind.UNSPECIFIED]: "dynamic",
  [NodeKind.DYNAMIC]: "dynamic",
  [NodeKind.GROUP]: "groupNode",
  [NodeKind.PROCESS]: "processing",
  [NodeKind.NOTE]: "dynamic",
};

const PORT_TYPE_TO_PROTO: Record<string, PortMainType> = {
  any: PortMainType.ANY,
  string: PortMainType.STRING,
  number: PortMainType.NUMBER,
  boolean: PortMainType.BOOLEAN,
  image: PortMainType.IMAGE,
  video: PortMainType.VIDEO,
  audio: PortMainType.AUDIO,
  list: PortMainType.LIST,
  set: PortMainType.SET,
  system: PortMainType.SYSTEM,
};

export const PROTO_TO_PORT_TYPE: Record<number, string> = {
  [PortMainType.UNSPECIFIED]: "any",
  [PortMainType.ANY]: "any",
  [PortMainType.STRING]: "string",
  [PortMainType.NUMBER]: "number",
  [PortMainType.BOOLEAN]: "boolean",
  [PortMainType.IMAGE]: "image",
  [PortMainType.VIDEO]: "video",
  [PortMainType.AUDIO]: "audio",
  [PortMainType.LIST]: "list",
  [PortMainType.SET]: "set",
  [PortMainType.SYSTEM]: "system",
};

function mapToProtoPortMainType(val: string | number): PortMainType {
  if (typeof val === "number") return val as PortMainType;
  return PORT_TYPE_TO_PROTO[val.toLowerCase()] ?? PortMainType.ANY;
}

/**
 * Adapter functions to convert between Protobuf generated types and internal React Flow types.
 */

// --- To Proto ---

export function toProtoNode(node: AppNode): Node {
  const data = node.data;
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

  let taskId = "";
  if (node.type === "processing") {
    taskId = (data as Record<string, unknown>).taskId as string;
  }

  const protoData = create(NodeDataSchema, {
    displayName: dynData.label,
    availableModes: dynData.modes,
    activeMode: dynData.activeMode,
    media: dynData.media
      ? {
          ...dynData.media,
          aspectRatio: dynData.media.aspectRatio ?? 0,
        }
      : undefined,
    widgets,
    inputPorts: (dynData.inputPorts ?? []).map((p) => ({
      ...p,
      type: p.type
        ? {
            ...p.type,
            mainType: mapToProtoPortMainType(p.type.mainType),
          }
        : undefined,
    })),
    outputPorts: (dynData.outputPorts ?? []).map((p) => ({
      ...p,
      type: p.type
        ? {
            ...p.type,
            mainType: mapToProtoPortMainType(p.type.mainType),
          }
        : undefined,
    })),
    metadata: {},
    taskId: taskId,
    widgetsValuesJson: JSON.stringify(dynData.widgetsValues || {}),
    widgetsSchemaJson: dynData.widgetsSchemaJson || "",
  });

  const presentation = create(PresentationSchema, {
    position: node.position,
    width: node.measured?.width ?? 0,
    height: node.measured?.height ?? 0,
    parentId: node.parentId ?? "",
    isInitialized: true,
  });

  return create(NodeSchema, {
    nodeId: node.id,
    templateId: dynData.typeId ?? "unknown",
    nodeKind: NODE_TYPE_TO_KIND[node.type] ?? NodeKind.DYNAMIC,
    presentation,
    state: protoData,
    isSelected: !!node.selected,
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
    edgeId: edge.id,
    sourceNodeId: edge.source,
    targetNodeId: edge.target,
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

function fromProtoEdge(e: ProtoEdge): Edge {
  return {
    id: e.edgeId,
    source: e.sourceNodeId,
    target: e.targetNodeId,
    sourceHandle: e.sourceHandle || undefined,
    targetHandle: e.targetHandle || undefined,
    data: (e.metadata as Record<string, unknown> | undefined) ?? {},
  };
}

export function fromProtoNode(n: Node): AppNode {
  const reactFlowType = KIND_TO_NODE_TYPE[n.nodeKind] ?? "dynamic";

  const protoData = n.state;
  let appData: DynamicNodeData;

  if (protoData) {
    appData = fromProtoNodeData(protoData);
  } else {
    appData = {
      label: "Unknown",
      modes: [],
      widgets: [],
    };
  }

  appData.typeId = n.templateId;

  const pres = n.presentation;
  let parentId: string | undefined = pres?.parentId;
  if (parentId === "") parentId = undefined;

  const node: AppNode = {
    id: n.nodeId,
    type: reactFlowType,
    position: { x: pres?.position?.x ?? 0, y: pres?.position?.y ?? 0 },
    selected: n.isSelected,
    parentId,
    extent: parentId ? "parent" : undefined,
    data: appData,
  } as AppNode;

  if (pres && pres.width && pres.height) {
    node.measured = { width: pres.width, height: pres.height };
    node.style = { width: pres.width, height: pres.height };
  }

  if (n.visualHint && !pres?.isInitialized) {
    (node.data as Record<string, unknown>)._visualHint = n.visualHint;
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
    label: protoData.displayName,
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
    inputPorts: protoData.inputPorts.map((p) => {
      const type = p.type;
      return {
        ...p,
        type: type
          ? ({
              ...type,
              mainType: PROTO_TO_PORT_TYPE[type.mainType] ?? "any",
            } as unknown as PortType)
          : undefined,
      } as Port;
    }),
    outputPorts: protoData.outputPorts.map((p) => {
      const type = p.type;
      return {
        ...p,
        type: type
          ? ({
              ...type,
              mainType: PROTO_TO_PORT_TYPE[type.mainType] ?? "any",
            } as unknown as PortType)
          : undefined,
      } as Port;
    }),
    taskId: protoData.taskId || undefined,
    widgetsSchemaJson: protoData.widgetsSchemaJson || undefined,
    widgetsValues: protoData.widgetsValuesJson
      ? JSON.parse(protoData.widgetsValuesJson)
      : undefined,
  };
}

// Minimal compatibility shim
export function toProtoNodeData(data?: DynamicNodeData): NodeData {
  if (!data) {
    return create(NodeDataSchema, {
      displayName: "",
      availableModes: [],
      widgets: [],
      inputPorts: [],
      outputPorts: [],
      metadata: {},
      taskId: "",
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

  const taskId = (data as Record<string, unknown>).taskId as string | undefined;

  return create(NodeDataSchema, {
    displayName: data.label,
    availableModes: data.modes,
    activeMode: data.activeMode,
    media: data.media
      ? {
          ...data.media,
          aspectRatio: data.media.aspectRatio ?? 0,
        }
      : undefined,
    widgets,
    inputPorts: (data.inputPorts ?? []).map((p) => ({
      ...p,
      type: p.type
        ? {
            ...p.type,
            mainType: mapToProtoPortMainType(p.type.mainType),
          }
        : undefined,
    })),
    outputPorts: (data.outputPorts ?? []).map((p) => ({
      ...p,
      type: p.type
        ? {
            ...p.type,
            mainType: mapToProtoPortMainType(p.type.mainType),
          }
        : undefined,
    })),
    metadata: {},
    taskId: taskId ?? "",
    widgetsValuesJson: JSON.stringify(data.widgetsValues || {}),
    widgetsSchemaJson: data.widgetsSchemaJson || "",
  });
}
