import type { Edge, Viewport, Node } from "@xyflow/react";
import type { GroupNodeType } from "./components/GroupNode";

export type WidgetType = "text" | "select" | "checkbox" | "slider" | "button";

export interface WidgetDef {
  id: string;
  type: WidgetType;
  label: string;
  value: unknown;
  options?: { label: string; value: unknown }[]; // For select
  config?: Record<string, unknown>; // min/max for slider, etc.
}

export interface MediaDef {
  type: "image" | "video" | "audio" | "markdown";
  url?: string;
  content?: string; // For markdown
  gallery?: string[]; // Array of additional media URLs (same type as main)
}

export type RenderMode = "media" | "widgets" | "markdown";

export interface DynamicNodeData extends Record<string, unknown> {
  label: string;
  modes: RenderMode[];
  activeMode?: RenderMode;
  media?: MediaDef & { aspectRatio?: number };
  widgets?: WidgetDef[];
  inputType: string;
  outputType: string;
  onChange: (id: string, data: Partial<DynamicNodeData>) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
  onGalleryItemContext?: (
    nodeId: string,
    imageUrl: string,
    x: number,
    y: number,
  ) => void;
}

export type DynamicNodeType = Node<DynamicNodeData, "dynamic">;

export function isDynamicNode(node: AppNode): node is DynamicNodeType {
  return node.type === "dynamic";
}

export interface NodeTemplate {
  id: string;
  label: string;
  path: string[]; // Menu hierarchy, e.g. ["Basic", "Input"]
  defaultData: Omit<DynamicNodeData, "onChange" | "onWidgetClick">;
}

export type NodeData = GroupNodeType["data"] | DynamicNodeData;

export interface TypedNodeData {
  inputType?: string;
  outputType?: string;
}

/* eslint-disable @typescript-eslint/no-explicit-any */
export function isTypedNodeData(data: any): data is TypedNodeData {
  return "inputType" in data || "outputType" in data;
}
/* eslint-enable @typescript-eslint/no-explicit-any */

export type AppNode = GroupNodeType | DynamicNodeType;

export type GraphState = {
  graph: {
    nodes: AppNode[];
    edges: Edge[];
    viewport?: Viewport;
  };
  version: number;
};
