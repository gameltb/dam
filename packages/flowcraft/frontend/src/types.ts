import type { Edge, Viewport, Node } from "@xyflow/react";
import type { GroupNodeType } from "./components/GroupNode";
import {
  WidgetType,
  RenderMode,
  MediaType,
  PortStyle,
  TaskStatus,
  type Port,
  type NodeTemplate as ProtoNodeTemplate,
} from "./generated/flowcraft/v1/core/node_pb";
import {
  MutationSource,
  PortMainType,
} from "./generated/flowcraft/v1/core/base_pb";
import { type GraphMutation } from "./generated/flowcraft/v1/core/service_pb";
import { ActionExecutionStrategy } from "./generated/flowcraft/v1/core/action_pb";

/**
 * SECTION 1: PROTOCOL & CRDT
 * Definitions related to the communication protocol and state synchronization.
 */

// Re-export Enums
export {
  WidgetType,
  RenderMode,
  MediaType,
  ActionExecutionStrategy,
  PortStyle,
  TaskStatus,
  MutationSource,
  PortMainType,
};

export type { Port };

export interface MutationLogEntry {
  id: string;
  taskId: string;
  source: MutationSource;
  timestamp: number;
  description: string;
  mutations: GraphMutation[];
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
  widgetsSchemaJson?: string;
  widgetsValues?: Record<string, unknown>;

  // Port definitions
  inputPorts?: Port[];
  outputPorts?: Port[];
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

export type NodeTemplate = ProtoNodeTemplate;

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
