import { create } from "@bufbuild/protobuf";

import { MediaType, NodeKind, PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { PortStyle, PortTypeSchema, RenderMode } from "@/generated/flowcraft/v1/core/node_pb";
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
 * Legacy hydration function (kept for backward compatibility during refactor if needed, but empty).
 * In the new architecture, we don't modify node objects with functions.
 */
export function hydrateNodes(nodes: AppNode[]): AppNode[] {
  return nodes;
}

export function mapToMediaType(val?: number | string): MediaType {
  if (val === undefined) return MediaType.MEDIA_UNSPECIFIED;
  if (typeof val === "number") return val as MediaType;
  const key = val as keyof typeof MediaType;
  return MediaType[key];
}

export function mapToRenderMode(val?: number | string): RenderMode {
  if (val === undefined) return RenderMode.MODE_UNSPECIFIED;
  if (typeof val === "number") return val as RenderMode;
  const key = val as keyof typeof RenderMode;
  return RenderMode[key];
}
