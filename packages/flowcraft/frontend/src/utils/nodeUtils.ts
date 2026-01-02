import { type AppNode } from "../types";

/**
 * Dehydrates a node by ensuring it only contains serializable data.
 * It recursively removes any functions or non-serializable properties.
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
