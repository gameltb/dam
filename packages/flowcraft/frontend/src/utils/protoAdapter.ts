import type { Edge } from "@xyflow/react";

import {
  create,
  fromJson,
  type JsonObject,
  type JsonValue,
} from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";

import {
  NodeKind,
  PortMainType,
  PresentationSchema,
} from "../generated/flowcraft/v1/core/base_pb";
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
  type Widget,
  WidgetSchema,
} from "../generated/flowcraft/v1/core/node_pb";
import { type GraphSnapshot } from "../generated/flowcraft/v1/core/service_pb";
import {
  type AppNode,
  AppNodeType,
  type ClientPort,
  type DynamicNodeData,
  type WidgetDef,
} from "../types";

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

/**
 * Adapter functions to convert between Protobuf generated types and internal React Flow types.
 */

// --- To Proto ---

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
    data: appData,
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

  if (n.visualHint && !pres?.isInitialized) {
    (node.data as Record<string, unknown>)._visualHint = n.visualHint;
  }

  return node;
}

export function fromProtoNodeData(protoData: NodeData): DynamicNodeData {
  const widgets: WidgetDef[] = protoData.widgets.map((w) => {
    const value = w.value as unknown;
    return {
      config: w.config as unknown as Record<string, unknown>,
      id: w.id,
      inputPortId: w.inputPortId,
      label: w.label,
      options: w.options.map((o) => ({
        label: o.label,
        value: o.value,
      })),
      type: w.type,
      value,
    };
  });

  const appData = {
    activeMode: protoData.activeMode,
    extension: protoData.extension,
    inputPorts: protoData.inputPorts.map(protoPortToClient),
    label: protoData.displayName,
    media: protoData.media
      ? {
          aspectRatio: protoData.media.aspectRatio,
          content: protoData.media.content,
          galleryUrls: protoData.media.galleryUrls,
          type: protoData.media.type,
          url: protoData.media.url,
        }
      : undefined,
    modes: protoData.availableModes,
    outputPorts: protoData.outputPorts.map(protoPortToClient),
    taskId: protoData.taskId || undefined,
    widgets,
    widgetsSchema: protoData.widgetsSchema as Record<string, unknown>,
    widgetsValues: protoData.widgetsValues as Record<string, unknown>,
  };

  return appData as DynamicNodeData;
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

export function toProtoNode(node: AppNode): Node {
  const data = node.data;
  const dynData = data as DynamicNodeData;

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

  let taskId = "";
  if (node.type === AppNodeType.PROCESSING) {
    const dataObj = data as Record<string, unknown>;
    taskId = typeof dataObj.taskId === "string" ? dataObj.taskId : "";
  }

  const protoData = create(NodeDataSchema, {
    activeMode: dynData.activeMode,
    availableModes: dynData.modes,
    displayName: dynData.label,
    inputPorts: (dynData.inputPorts ?? []).map(clientPortToProto),
    media: dynData.media
      ? {
          ...dynData.media,
          aspectRatio: dynData.media.aspectRatio ?? 0,
        }
      : undefined,
    metadata: {},
    outputPorts: (dynData.outputPorts ?? []).map(clientPortToProto),
    taskId: taskId,
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

// Minimal compatibility shim
export function toProtoNodeData(data?: DynamicNodeData): NodeData {
  if (!data) {
    return create(NodeDataSchema, {
      availableModes: [],
      displayName: "",
      inputPorts: [],
      metadata: {},
      outputPorts: [],
      taskId: "",
      widgets: [],
    });
  }
  const widgets: Widget[] = (data.widgets ?? []).map((w) =>
    create(WidgetSchema, {
      config: w.config as unknown as JsonObject,
      id: w.id,
      inputPortId: w.inputPortId,
      label: w.label,
      options: w.options?.map((o) => ({
        label: o.label,
        value: o.value as string,
      })),
      type: w.type,
      value: fromJson(ValueSchema, w.value as JsonValue),
    }),
  );

  const taskId = (data as Record<string, unknown>).taskId as string | undefined;

  return create(NodeDataSchema, {
    activeMode: data.activeMode,
    availableModes: data.modes,
    displayName: data.label,
    extension: data.extension,
    inputPorts: (data.inputPorts ?? []).map(clientPortToProto),
    media: data.media
      ? {
          ...data.media,
          aspectRatio: data.media.aspectRatio ?? 0,
        }
      : undefined,
    metadata: {},
    outputPorts: (data.outputPorts ?? []).map(clientPortToProto),
    taskId: taskId ?? "",
    widgets,
    widgetsSchema: data.widgetsSchema as unknown as JsonObject,
    widgetsValues: data.widgetsValues as unknown as JsonObject,
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

function mapToProtoPortMainType(val?: number | string): PortMainType {
  if (val === undefined) return PortMainType.ANY;
  if (typeof val === "number") return val as PortMainType;
  return PORT_MAIN_TYPE_TO_PROTO[val.toLowerCase()] ?? PortMainType.ANY;
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
          mainType: PORT_MAIN_TYPE_FROM_PROTO[type.mainType] ?? "any",
        }
      : undefined,
  };
}
