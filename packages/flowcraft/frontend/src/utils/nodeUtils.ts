import { create } from "@bufbuild/protobuf";
import { type Edge } from "@xyflow/react";

import { MediaType, NodeKind, PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle, PortTypeSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { type AppNode, AppNodeType, type ClientPort, isDynamicNode } from "@/types";

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

export const KIND_TO_NODE_TYPE: Record<number, AppNodeType> = {
  [NodeKind.DYNAMIC]: AppNodeType.DYNAMIC,
  [NodeKind.GROUP]: AppNodeType.GROUP,
  [NodeKind.NOTE]: AppNodeType.DYNAMIC,
  [NodeKind.PROCESS]: AppNodeType.PROCESSING,
  [NodeKind.UNSPECIFIED]: AppNodeType.DYNAMIC,
};

export function calculateNodeRelations(
  nodes: AppNode[],
  edges: Edge[],
): Record<
  string,
  {
    firstChildId?: string;

    left?: string;

    nextSiblingId?: string;

    parentId?: string;

    prevSiblingId?: string;

    right?: string;
  }
> {
  const relations: Record<
    string,
    {
      firstChildId?: string;

      left?: string;

      nextSiblingId?: string;

      parentId?: string;

      prevSiblingId?: string;

      right?: string;
    }
  > = {};

  // 1. Initialize containers for all nodes

  nodes.forEach((n) => {
    relations[n.id] = {
      parentId: n.parentId || undefined,
    };
  });

  // 2. Establish logical left/right relations (based on edges)

  edges.forEach((e) => {
    if (relations[e.source]) relations[e.source]!.right = e.target;

    if (relations[e.target]) relations[e.target]!.left = e.source;
  });

  // 3. Establish hierarchical relations (drill-down/up)

  // Find the first child for each parent

  nodes.forEach((n) => {
    if (n.parentId && relations[n.parentId]) {
      if (!relations[n.parentId]!.firstChildId) {
        relations[n.parentId]!.firstChildId = n.id;
      }
    }
  });

  // 4. Establish sibling relations at the same level

  const nodesByParent: Record<string, string[]> = {};

  nodes.forEach((n) => {
    const pId = n.parentId || "root";

    if (!nodesByParent[pId]) nodesByParent[pId] = [];

    nodesByParent[pId].push(n.id);
  });

  Object.values(nodesByParent).forEach((childrenIds) => {
    for (let i = 0; i < childrenIds.length; i++) {
      const current = childrenIds[i]!;

      if (i > 0) relations[current]!.prevSiblingId = childrenIds[i - 1];

      if (i < childrenIds.length - 1) relations[current]!.nextSiblingId = childrenIds[i + 1];
    }
  });

  return relations;
}

/**
 * Dehydrates a node by ensuring it only contains serializable data.
 */
export function dehydrateNode<T>(obj: T): T {
  if (obj === null || typeof obj !== "object") {
    return obj;
  }

  if (Array.isArray(obj)) {
    return obj.map(dehydrateNode) as unknown as T;
  }

  const result: Record<string, unknown> = {};
  for (const [key, value] of Object.entries(obj)) {
    // Skip functions and undefined values
    if (typeof value === "function" || value === undefined) {
      continue;
    }
    result[key] = dehydrateNode(value);
  }
  return result as T;
}

/**
 * Finds a port by its ID within a node.
 */
export function findPort(node: AppNode, portId: string): ClientPort | undefined {
  if (!isDynamicNode(node)) return undefined;

  const data = node.data;
  const explicitPort = data.outputPorts?.find((p) => p.id === portId) ?? data.inputPorts?.find((p) => p.id === portId);

  if (explicitPort) return explicitPort;

  const widget = data.widgets?.find((w) => w.inputPortId === portId);
  if (widget?.inputPortId) {
    // Implicit widget ports are treated as STRING type by convention
    return {
      description: "",
      id: widget.inputPortId,
      label: widget.label,
      style: PortStyle.CIRCLE,
      type: create(PortTypeSchema, {
        isGeneric: false,
        itemType: "",
        mainType: PortMainType.STRING,
      }),
    } as ClientPort;
  }

  return undefined;
}

/**
 * Maps a MIME type string to the appropriate MediaType enum.
 */
export function getMediaTypeFromMime(mimeType?: string): MediaType {
  if (!mimeType) return MediaType.MEDIA_UNSPECIFIED;

  if (mimeType.startsWith("image/")) return MediaType.MEDIA_IMAGE;
  if (mimeType.startsWith("video/")) return MediaType.MEDIA_VIDEO;
  if (mimeType.startsWith("audio/")) return MediaType.MEDIA_AUDIO;
  if (mimeType === "text/markdown") return MediaType.MEDIA_MARKDOWN;

  return MediaType.MEDIA_UNSPECIFIED;
}

/**
 * Calculates direct hierarchical relations between nodes (Jump relations)
 */
export function mapToMediaType(val?: number | string): MediaType {
  if (val === undefined) return MediaType.MEDIA_UNSPECIFIED;
  if (typeof val === "number") return val as MediaType;
  const key = val as keyof typeof MediaType;
  return MediaType[key];
}
