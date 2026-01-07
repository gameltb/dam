import { MediaType, PortStyle } from "../generated/flowcraft/v1/core/node_pb";
import { type AppNode, type ClientPort, isDynamicNode } from "../types";

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
export function findPort(
  node: AppNode,
  portId: string,
): ClientPort | undefined {
  if (!isDynamicNode(node)) return undefined;

  const data = node.data;
  const explicitPort =
    data.outputPorts?.find((p) => p.id === portId) ??
    data.inputPorts?.find((p) => p.id === portId);

  if (explicitPort) return explicitPort;

  const widget = data.widgets?.find((w) => w.inputPortId === portId);
  if (widget?.inputPortId) {
    // Implicit widget ports are treated as STRING type by convention
    return {
      id: widget.inputPortId,
      label: widget.label,
      style: PortStyle.CIRCLE,
      type: {
        isGeneric: false,
        itemType: "",
        mainType: "string",
      },
    };
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
