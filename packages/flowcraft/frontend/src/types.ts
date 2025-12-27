import type { Edge, Viewport, Node } from "@xyflow/react";
import type { GroupNodeType } from "./components/GroupNode";
import { flowcraft_proto } from "./generated/flowcraft_proto";

/**
 * SECTION 1: PROTOCOL & CRDT
 * Definitions related to the communication protocol and state synchronization.
 */

export { flowcraft_proto };
export type { flowcraft_proto as flowcraft_proto_type } from "./generated/flowcraft_proto";

// Re-export Enums from Protobuf as both values and types
export const WidgetType = flowcraft_proto.v1.WidgetType;
export type WidgetType = flowcraft_proto.v1.WidgetType;

export const RenderMode = flowcraft_proto.v1.RenderMode;
export type RenderMode = flowcraft_proto.v1.RenderMode;

export const MediaType = flowcraft_proto.v1.MediaType;
export type MediaType = flowcraft_proto.v1.MediaType;

export const ActionExecutionStrategy =
  flowcraft_proto.v1.ActionExecutionStrategy;
export type ActionExecutionStrategy =
  flowcraft_proto.v1.ActionExecutionStrategy;

export const PortStyle = flowcraft_proto.v1.PortStyle;
export type PortStyle = flowcraft_proto.v1.PortStyle;

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
  mutations: flowcraft_proto.v1.IGraphMutation[];
}

/**
 * SECTION 2: TASK & JOB SYSTEM
 * Tracking long-running operations and their lifecycle.
 */

// Manually define TaskStatus due to generation issues in some environments
export const TaskStatus = {
  TASK_PENDING: 0,
  TASK_PROCESSING: 1,
  TASK_COMPLETED: 2,
  TASK_FAILED: 3,
  TASK_CANCELLED: 4,
} as const;
export type TaskStatus = (typeof TaskStatus)[keyof typeof TaskStatus];

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

/**
 * SECTION 3: CORE NODE & GRAPH TYPES
 * The main building blocks of the Flowcraft editor.
 */

export type { Edge, Viewport, Node };

export interface WidgetDef {
  id: string;
  type: WidgetType;
  label: string;
  value: unknown;
  options?: { label: string; value: unknown }[];
  config?: Record<string, unknown>;
  inputPortId?: string;
}

export interface MediaDef {
  type: MediaType;
  url?: string;
  content?: string;
  galleryUrls?: string[];
}

export interface DynamicNodeData extends Record<string, unknown> {
  typeId?: string; // Original backend template ID
  label: string;
  modes: RenderMode[];
  activeMode?: RenderMode;
  media?: MediaDef & { aspectRatio?: number };
  widgets?: WidgetDef[];

  // Port definitions
  inputPorts?: flowcraft_proto.v1.IPort[];
  outputPorts?: flowcraft_proto.v1.IPort[];

  // Handlers (attached during hydration)
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

export interface ProcessingNodeData extends Record<string, unknown> {
  taskId: string;
  label: string;
  onCancel?: (taskId: string) => void;
}

export type DynamicNodeType = Node<DynamicNodeData, "dynamic">;
export type ProcessingNodeType = Node<ProcessingNodeData, "processing">;
export type AppNode = GroupNodeType | DynamicNodeType | ProcessingNodeType;

export type NodeData = AppNode["data"];

export interface GraphState {
  graph: {
    nodes: AppNode[];
    edges: Edge[];
    viewport?: Viewport;
  };
  version: number;
}

/**
 * SECTION 4: TEMPLATES & PROTOCOL PAYLOADS
 */

export interface NodeTemplate {
  id: string;
  label: string;
  path: string[]; // Menu hierarchy, e.g. ["Basic", "Input"]
  defaultData: Omit<DynamicNodeData, "onChange" | "onWidgetClick">;
}

export interface WidgetUpdatePayload {
  nodeId: string;
  widgetId: string;
  value: unknown;
}

/**
 * SECTION 5: TYPE GUARDS & UTILITIES
 */

export function isDynamicNode(node: AppNode): node is DynamicNodeType {
  return node.type === "dynamic";
}

export interface TypedNodeData {
  inputType?: string;
  outputType?: string;
}

export function isTypedNodeData(data: unknown): data is TypedNodeData {
  return (
    typeof data === "object" &&
    data !== null &&
    ("inputType" in data || "outputType" in data)
  );
}
