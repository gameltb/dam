import { type AppNode, isDynamicNode, type Port, PortMainType } from "../types";
import { type PortType } from "../generated/flowcraft/v1/core/node_pb";

/**
 * Finds a port by its ID within a node.
 */
export function findPort(node: AppNode, portId: string): Port | undefined {
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
      type: {
        mainType: PortMainType.STRING,
        itemType: "",
        isGeneric: false,
      } as unknown as PortType,
    } as Port;
  }

  return undefined;
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
 * Legacy hydration function (kept for backward compatibility during refactor if needed, but empty).
 * In the new architecture, we don't modify node objects with functions.
 */
export function hydrateNodes(nodes: AppNode[]): AppNode[] {
  return nodes;
}
