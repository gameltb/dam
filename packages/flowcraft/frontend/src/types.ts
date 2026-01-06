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

/**
 * SECTION 2: UI & INTERACTION
 * Enums and types for frontend-specific logic.
 */

export enum AppNodeType {
  DYNAMIC = "dynamic",
  PROCESSING = "processing",
  GROUP = "groupNode",
}

export enum FlowEvent {
  WIDGET_CLICK = "widget-click",
  GALLERY_ITEM_CONTEXT = "gallery-item-context",
  PANE_CLICK = "pane-click",
  OPEN_PREVIEW = "open-preview",
  OPEN_EDITOR = "open-editor",
}

/**
 * Client-side representation of a Port, using string types for easier UI handling
 * while keeping the structure similar to the Protobuf definition.
 */
export interface ClientPort {
  id: string;
  label: string;
  description?: string;
  type?: {
    mainType: string;
    itemType: string;
    isGeneric: boolean;
  };
  style: PortStyle;
}

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

export interface DynamicNodeData {
  typeId?: string; // Original backend template ID
  label: string;
  modes: RenderMode[];
  activeMode?: RenderMode;
  media?: MediaDef & { aspectRatio?: number };
  widgets?: WidgetDef[];
  widgetsSchemaJson?: string;
  widgetsValues?: Record<string, unknown>;

  // Port definitions
  inputPorts?: ClientPort[];
  outputPorts?: ClientPort[];

  // Custom metadata or transient state
  [key: string]: unknown;
}

export interface ProcessingNodeData {
  taskId: string;
  label: string;
  onCancel?: (taskId: string) => void;
  [key: string]: unknown;
}

export type DynamicNodeType = Node<DynamicNodeData, AppNodeType.DYNAMIC>;
export type ProcessingNodeType = Node<
  ProcessingNodeData,
  AppNodeType.PROCESSING
>;
export type AppNode = GroupNodeType | DynamicNodeType | ProcessingNodeType;

/**
 * SECTION 4: TEMPLATES & PROTOCOL PAYLOADS
 */

export type NodeTemplate = ProtoNodeTemplate;

/**
 * SECTION 5: TYPE GUARDS & UTILITIES
 */

export function isDynamicNode(node: AppNode): node is DynamicNodeType {
  return node.type === AppNodeType.DYNAMIC;
}
