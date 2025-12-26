import type { Edge, Viewport, Node } from "@xyflow/react";
import type { GroupNodeType } from "./components/GroupNode";
import { flowcraft as _flowcraft } from "./generated/flowcraft";

// The data structure for a processing node
export interface ProcessingNodeData extends Record<string, unknown> {
  taskId: string;
  label: string;
  onCancel?: (taskId: string) => void;
}

// Export the generated flowcraft namespace types
export type { _flowcraft as flowcraft };

// Re-export core types for other modules
export type { Edge, Viewport, Node };

export type ProcessingNodeType = Node<ProcessingNodeData, "processing">;

// Re-export Enums from Protobuf
export const WidgetType = _flowcraft.v1.WidgetType;
export type WidgetType = _flowcraft.v1.WidgetType;

export const RenderMode = _flowcraft.v1.RenderMode;
export type RenderMode = _flowcraft.v1.RenderMode;

export const MediaType = _flowcraft.v1.MediaType;
export type MediaType = _flowcraft.v1.MediaType;

export const ActionExecutionStrategy = _flowcraft.v1.ActionExecutionStrategy;
export type ActionExecutionStrategy = _flowcraft.v1.ActionExecutionStrategy;

export const PortStyle = _flowcraft.v1.PortStyle;
export type PortStyle = _flowcraft.v1.PortStyle;

// Manually define TaskStatus due to generation issues
export const TaskStatus = {
  TASK_PENDING: 0,
  TASK_PROCESSING: 1,
  TASK_COMPLETED: 2,
  TASK_FAILED: 3,
  TASK_CANCELLED: 4,
} as const;
export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

export interface WidgetDef {
  id: string;
  type: WidgetType;
  label: string;
  value: unknown;
  options?: { label: string; value: unknown }[]; // For select
  config?: Record<string, unknown>; // min/max for slider, etc.
  inputPortId?: string; // New field
}

export interface MediaDef {
  type: MediaType;
  url?: string;
  content?: string; // For markdown
  galleryUrls?: string[]; // Array of additional media URLs (same type as main)
}

export interface DynamicNodeData extends Record<string, unknown> {
  typeId?: string; // Original backend type ID
  label: string;
  modes: RenderMode[];
  activeMode?: RenderMode;
  media?: MediaDef & { aspectRatio?: number };
  widgets?: WidgetDef[];

  // New Port definitions
  inputPorts?: _flowcraft.v1.IPort[];
  outputPorts?: _flowcraft.v1.IPort[];

  onChange: (id: string, data: Partial<DynamicNodeData>) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
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

// --- Protocol Definitions ---

export interface WidgetUpdatePayload {
  nodeId: string;
  widgetId: string;
  value: unknown;
}

export interface WidgetOptionsRequest {
  nodeId: string;
  widgetId: string;
  context?: Record<string, unknown>;
}

export interface StreamChunk {
  nodeId: string;
  widgetId: string;
  chunk: string;
  isDone: boolean;
}

// --- Task / Job System Definitions ---

export enum MutationSource {
  USER = "USER",
  REMOTE_TASK = "REMOTE_TASK",
  SYSTEM = "SYSTEM",
  SYNC = "SYNC",
}

export interface MutationLogEntry {
  id: string;
  taskId: string;
  source: MutationSource;
  timestamp: number;
  description: string;
  mutations: flowcraft.v1.IGraphMutation[];
}

export interface TaskDefinition {
  taskId: string;
  type: string;
  label: string;
  source: MutationSource;
  status: TaskStatus;
  progress: number;
  message: string;
  createdAt: number;
  updatedAt: number;
  mutationIds: string[]; // Reference to log entries
}

export interface TaskCancelRequest {
  taskId: string;
  reason?: string;
}

export type AppNode = GroupNodeType | DynamicNodeType | ProcessingNodeType;

export type GraphState = {
  graph: {
    nodes: AppNode[];
    edges: Edge[];
    viewport?: Viewport;
  };
  version: number;
};
