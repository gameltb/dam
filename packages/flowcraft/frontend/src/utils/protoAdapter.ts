import type { Edge } from "@xyflow/react";

import {
  create,
  fromJson,
  type JsonObject,
  type JsonValue,
} from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import {
  MediaType,
  NodeKind,
  PortMainType,
  PresentationSchema,
} from "@/generated/flowcraft/v1/core/base_pb";
import {
  EdgeSchema,
  type Node,
  type NodeData,
  NodeDataSchema,
  NodeSchema,
  PortSchema,
  PortTypeSchema,
  type Edge as ProtoEdge,
  type Port as ProtoPort,
  RenderMode,
  type Widget,
  WidgetSchema,
} from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphSnapshot } from "@/generated/flowcraft/v1/core/service_pb";
import {
  type AppNode,
  AppNodeType,
  type ClientPort,
  type DynamicNodeData,
} from "@/types";

/**
 * Mappings for NodeKind and PortMainType enums.
 */
const NODE_TYPE_TO_KIND: Record<string, NodeKind> = {
  [AppNodeType.DYNAMIC]: NodeKind.DYNAMIC,
  [AppNodeType.GROUP]: NodeKind.GROUP,
  [AppNodeType.PROCESSING]: NodeKind.PROCESS,
};

const KIND_TO_NODE_TYPE: Record<number, AppNodeType> = {
  [NodeKind.DYNAMIC]: AppNodeType.DYNAMIC,
  [NodeKind.GROUP]: AppNodeType.GROUP,
  [NodeKind.NOTE]: AppNodeType.DYNAMIC,
  [NodeKind.PROCESS]: AppNodeType.PROCESSING,
  [NodeKind.UNSPECIFIED]: AppNodeType.DYNAMIC,
};

export const PORT_MAIN_TYPE_TO_PROTO: Record<string, PortMainType> = {
  any: PortMainType.ANY,
  audio: PortMainType.AUDIO,
  boolean: PortMainType.BOOLEAN,
  image: PortMainType.IMAGE,
  list: PortMainType.LIST,
  number: PortMainType.NUMBER,
  set: PortMainType.SET,
  string: PortMainType.STRING,
  system: PortMainType.SYSTEM,
  video: PortMainType.VIDEO,
};

export const PORT_MAIN_TYPE_FROM_PROTO: Record<number, string> = {
  [PortMainType.ANY]: "any",
  [PortMainType.AUDIO]: "audio",
  [PortMainType.BOOLEAN]: "boolean",
  [PortMainType.IMAGE]: "image",
  [PortMainType.LIST]: "list",
  [PortMainType.NUMBER]: "number",
  [PortMainType.SET]: "set",
  [PortMainType.STRING]: "string",
  [PortMainType.SYSTEM]: "system",
  [PortMainType.UNSPECIFIED]: "any",
  [PortMainType.VIDEO]: "video",
};

export function fromProtoGraph(protoGraph: GraphSnapshot): {
  edges: Edge[];
  nodes: AppNode[];
} {
  const nodes: AppNode[] = protoGraph.nodes.map((n) => fromProtoNode(n));
  const edges: Edge[] = protoGraph.edges.map(fromProtoEdge);

  return { edges, nodes };
}

// --- Utils ---

function mapToMediaType(val?: number | string): MediaType {
  if (val === undefined) return MediaType.MEDIA_UNSPECIFIED;
  if (typeof val === "number") return val as MediaType;
  return (MediaType as any)[val] ?? MediaType.MEDIA_UNSPECIFIED;
}

function mapToRenderMode(val?: number | string): RenderMode {
  if (val === undefined) return RenderMode.MODE_UNSPECIFIED;
  if (typeof val === "number") return val as RenderMode;
  return (RenderMode as any)[val] ?? RenderMode.MODE_UNSPECIFIED;
}

function mapToProtoPortMainType(val?: number | string): PortMainType {
  if (val === undefined) return PortMainType.ANY;
  if (typeof val === "number") return val as PortMainType;
  return PORT_MAIN_TYPE_TO_PROTO[val.toLowerCase()] ?? PortMainType.ANY;
}

export function toProtoEdge(edge: Edge): ProtoEdge {
  const metadata: Record<string, string> = {};
  if (edge.data && typeof edge.data === "object") {
    Object.entries(edge.data).forEach(([k, v]) => {
      if (typeof v === "string") metadata[k] = v;
    });
  }

  return create(EdgeSchema, {
    edgeId: edge.id,
    metadata,
    sourceHandle: edge.sourceHandle ?? "",
    sourceNodeId: edge.source,
    targetHandle: edge.targetHandle ?? "",
    targetNodeId: edge.target,
  });
}

// --- From Proto ---

export function fromProtoNode(n: Node): AppNode {
  const reactFlowType = KIND_TO_NODE_TYPE[n.nodeKind] ?? AppNodeType.DYNAMIC;
  const protoData = n.state;
  const appData = protoData ? fromProtoNodeData(protoData) : {};

  appData.label ??= "Unknown";
  appData.modes ??= [];
  appData.widgets ??= [];
  appData.typeId = n.templateId;

  const pres = n.presentation;
  let parentId: string | undefined = pres?.parentId;
  if (parentId === "") parentId = undefined;

  const node: AppNode = {
    data: appData as DynamicNodeData,
    extent: parentId ? "parent" : undefined,
    id: n.nodeId,
    parentId,
    position: { x: pres?.position?.x ?? 0, y: pres?.position?.y ?? 0 },
    selected: n.isSelected,
    type: reactFlowType,
  } as AppNode;

  if (pres && pres.width && pres.height) {
    node.measured = { height: pres.height, width: pres.width };
    node.style = { height: pres.height, width: pres.width };
  }

  return node;
}

export function fromProtoNodeData(
  protoData: NodeData,
): Partial<DynamicNodeData> {
  const appData: Partial<DynamicNodeData> = {};

  if (protoData.displayName !== "") appData.label = protoData.displayName;
  if (protoData.availableModes.length > 0)
    appData.modes = protoData.availableModes;
  if (protoData.activeMode !== RenderMode.MODE_UNSPECIFIED)
    appData.activeMode = protoData.activeMode;

  if (protoData.media) {
    appData.media = {
      aspectRatio: protoData.media.aspectRatio,
      content: protoData.media.content,
      galleryUrls: protoData.media.galleryUrls,
      type: protoData.media.type,
      url: protoData.media.url,
    };
  }

  if (protoData.widgets.length > 0) {
    appData.widgets = protoData.widgets.map((w) => ({
      config: w.config as unknown as Record<string, unknown>,
      id: w.id,
      inputPortId: w.inputPortId,
      label: w.label,
      options: w.options.map((o) => ({ label: o.label, value: o.value })),
      type: w.type,
      value: w.value,
    }));
  }

  if (protoData.inputPorts.length > 0)
    appData.inputPorts = protoData.inputPorts.map(protoPortToClient);
  if (protoData.outputPorts.length > 0)
    appData.outputPorts = protoData.outputPorts.map(protoPortToClient);
  if (protoData.taskId !== "") appData.taskId = protoData.taskId;

  if (protoData.widgetsSchema)
    appData.widgetsSchema = protoData.widgetsSchema as Record<string, unknown>;
  if (protoData.widgetsValues)
    appData.widgetsValues = protoData.widgetsValues as Record<string, unknown>;

  if (protoData.extension.case !== undefined)
    appData.extension = protoData.extension;

  return appData;
}

// --- To Proto ---

export function toProtoNode(node: AppNode): Node {
  const dynData = node.data as DynamicNodeData;

  const widgets: Widget[] = (dynData.widgets ?? []).map((w) =>
    create(WidgetSchema, {
      config: w.config as unknown as JsonObject,
      id: w.id,
      inputPortId: w.inputPortId,
      label: w.label,
      options: w.options?.map((o) => ({
        label: o.label,
        value: String(o.value),
      })),
      type: w.type,
      value: fromJson(ValueSchema, w.value as JsonValue),
    }),
  );

  const protoData = create(NodeDataSchema, {
    activeMode: mapToRenderMode(dynData.activeMode),
    availableModes: (dynData.modes ?? []).map(mapToRenderMode),
    displayName: dynData.label,
    extension: dynData.extension,
    inputPorts: (dynData.inputPorts ?? []).map(clientPortToProto),
    media: dynData.media
      ? {
          ...dynData.media,
          aspectRatio: dynData.media.aspectRatio ?? 0,
          type: mapToMediaType(dynData.media.type),
        }
      : undefined,
    metadata: {},
    outputPorts: (dynData.outputPorts ?? []).map(clientPortToProto),
    taskId: (node.type === AppNodeType.PROCESSING ? (node.data as any).taskId : "") || "",
    widgets,
    widgetsSchema: dynData.widgetsSchema as unknown as JsonObject,
    widgetsValues: dynData.widgetsValues as unknown as JsonObject,
  });

  const presentation = create(PresentationSchema, {
    height: node.measured?.height ?? 0,
    isInitialized: true,
    parentId: node.parentId ?? "",
    position: node.position,
    width: node.measured?.width ?? 0,
  });

  return create(NodeSchema, {
    isSelected: !!node.selected,
    nodeId: node.id,
    nodeKind: NODE_TYPE_TO_KIND[node.type] ?? NodeKind.DYNAMIC,
    presentation,
    state: protoData,
    templateId: dynData.typeId ?? "unknown",
  });
}

export function toProtoNodeData(data?: DynamicNodeData): NodeData {
  if (!data) return create(NodeDataSchema, {});

  const widgets: Widget[] = (data.widgets ?? []).map((w) =>
    create(WidgetSchema, {
      config: (w.config ?? {}) as unknown as JsonObject,
      id: w.id,
      inputPortId: w.inputPortId,
      label: w.label,
      options: w.options?.map((o) => ({
        label: o.label,
        value: o.value as string,
      })),
      type: w.type,
      value: fromJson(ValueSchema, (w.value ?? null) as JsonValue),
    }),
  );

  return create(NodeDataSchema, {
    activeMode: mapToRenderMode(data.activeMode),
    availableModes: (data.modes ?? []).map(mapToRenderMode),
    displayName: data.label,
    extension: data.extension,
    inputPorts: (data.inputPorts ?? []).map(clientPortToProto),
    media: data.media
      ? {
          ...data.media,
          aspectRatio: data.media.aspectRatio ?? 0,
          type: mapToMediaType(data.media.type),
        }
      : undefined,
    metadata: {},
    outputPorts: (data.outputPorts ?? []).map(clientPortToProto),
    taskId: (data as any).taskId || "",
    widgets,
    widgetsSchema: (data.widgetsSchema ?? {}) as unknown as JsonObject,
    widgetsValues: (data.widgetsValues ?? {}) as unknown as JsonObject,
  });
}

function clientPortToProto(p: ClientPort): ProtoPort {
  return create(PortSchema, {
    description: p.description ?? "",
    id: p.id,
    label: p.label,
    style: p.style,
    type: p.type
      ? create(PortTypeSchema, {
          isGeneric: p.type.isGeneric,
          itemType: p.type.itemType,
          mainType: mapToProtoPortMainType(p.type.mainType),
        })
      : undefined,
  });
}

function fromProtoEdge(e: ProtoEdge): Edge {
  return {
    data: (e.metadata as Record<string, unknown> | undefined) ?? {},
    id: e.edgeId,
    source: e.sourceNodeId,
    sourceHandle: e.sourceHandle || undefined,
    target: e.targetNodeId,
    targetHandle: e.targetHandle || undefined,
  };
}

function protoPortToClient(p: ProtoPort): ClientPort {
  const type = p.type;
  return {
    description: p.description,
    id: p.id,
    label: p.label,
    style: p.style,
    type: type
      ? {
          isGeneric: type.isGeneric,
          itemType: type.itemType,
          mainType: type.mainType,
        }
      : undefined,
  };
}