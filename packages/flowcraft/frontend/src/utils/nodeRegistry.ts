import { RenderMode } from "@/generated/flowcraft/v1/core/node_pb";

export interface NodeConstraint {
  defaultHeight?: number;
  defaultWidth?: number;
  minHeight?: number;
  minWidth?: number;
  modeConstraints?: Partial<Record<RenderMode, { minHeight: number; minWidth: number }>>;
}

export const NODE_CONSTRAINTS: Record<string, NodeConstraint> = {
  chat: {
    minHeight: 600,
    minWidth: 450,
    modeConstraints: {
      [RenderMode.MODE_CHAT]: { minHeight: 600, minWidth: 450 },
      [RenderMode.MODE_WIDGETS]: { minHeight: 400, minWidth: 300 },
    },
  },
  default: {
    minHeight: 150,
    minWidth: 200,
  },
  group: {
    minHeight: 100,
    minWidth: 150,
  },
  processing: {
    minHeight: 150,
    minWidth: 280,
  },
};

/**
 * Resolves constraints for a given template ID or node type.
 */
export function getConstraintsForTemplate(templateId: string): NodeConstraint {
  const key = Object.keys(NODE_CONSTRAINTS).find((k) => templateId.toLowerCase().includes(k.toLowerCase()));
  return NODE_CONSTRAINTS[key || "default"]!;
}
