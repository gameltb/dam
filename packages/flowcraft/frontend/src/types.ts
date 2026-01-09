import type { Edge, Node, Viewport } from "@xyflow/react";

import { ActionExecutionStrategy } from "@/generated/flowcraft/v1/core/action_pb";
import {
  MediaType,
  MutationSource,
  PortMainType,
} from "@/generated/flowcraft/v1/core/base_pb";
import {
  type Port,
  PortStyle,
  type NodeTemplate as ProtoNodeTemplate,
  RenderMode,
  TaskStatus,
  WidgetType,
} from "@/generated/flowcraft/v1/core/node_pb";
import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { type AiGenNodeState } from "@/generated/flowcraft/v1/nodes/ai_gen_node_pb";
import { type ChatNodeState } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import {
  type AcousticNodeState,
  type DocumentNodeState,
  type VisualNodeState,
} from "@/generated/flowcraft/v1/nodes/media_node_pb";

import type { GroupNodeType } from "./components/GroupNode";

/**
 * SECTION 1: PROTOCOL & CRDT
 * Definitions related to the communication protocol and state synchronization.
 */

// Re-export Enums
export {
  ActionExecutionStrategy,
  MediaType,
  MutationSource,
  PortMainType,
  PortStyle,
  RenderMode,
  TaskStatus,
  WidgetType,
};

export type { Port };

/**
 * SECTION 2: UI & INTERACTION
 * Enums and types for frontend-specific logic.
 */

export enum AppNodeType {
  DYNAMIC = "dynamic",
  GROUP = "groupNode",
  PROCESSING = "processing",
}

export enum ChatViewMode {
  FULLSCREEN = "fullscreen",
  INLINE = "inline",
  SIDEBAR = "sidebar",
}

export enum DragMode {
  PAN = "pan",
  SELECT = "select",
}

export enum FlowEvent {
  GALLERY_ITEM_CONTEXT = "gallery-item-context",
  OPEN_EDITOR = "open-editor",
  OPEN_PREVIEW = "open-preview",
  PANE_CLICK = "pane-click",
  WIDGET_CLICK = "widget-click",
}

export enum OverflowMode {
  HIDDEN = "hidden",
  VISIBLE = "visible",
}

export enum TaskType {
  ANONYMOUS = "anonymous",
  MANUAL = "manual",
  REMOTE = "remote-task",
  UNKNOWN = "unknown",
}

export enum Theme {
  DARK = "dark",
  LIGHT = "light",
}

export enum VideoMode {
  FIT = "fit",
  ORIGINAL = "original",
}

export interface AcousticNodeData extends BaseDynamicNodeData {
  extension: {
    case: "acoustic";
    value: Omit<AcousticNodeState, "$typeName">;
  };
}

export interface AiGenNodeData extends BaseDynamicNodeData {
  extension: {
    case: "aiGen";
    value: Omit<AiGenNodeState, "$typeName">;
  };
}

export type AppNode = DynamicNodeType | GroupNodeType | ProcessingNodeType;

export interface BaseDynamicNodeData {
  // Custom metadata or transient state
  [key: string]: unknown;
  activeMode?: RenderMode;
  // Port definitions
  inputPorts?: ClientPort[];
  label: string;
  media?: MediaDef & { aspectRatio?: number };
  modes: RenderMode[];
  outputPorts?: ClientPort[];

  typeId?: string; // Original backend template ID

  widgets?: WidgetDef[];
  widgetsSchema?: Record<string, unknown>;

  widgetsValues?: Record<string, unknown>;
}

export interface ChatNodeData extends BaseDynamicNodeData {
  extension: {
    case: "chat";
    value: Omit<ChatNodeState, "$typeName">;
  };
}

/**
 * Client-side representation of a Port, using string types for easier UI handling
 * while keeping the structure similar to the Protobuf definition.
 */
export interface ClientPort {
  description?: string;
  id: string;
  label: string;
  style: PortStyle;
  type?: {
    isGeneric: boolean;
    itemType: string;
    mainType: PortMainType;
  };
}

export interface DocumentNodeData extends BaseDynamicNodeData {
  extension: {
    case: "document";
    value: Omit<DocumentNodeState, "$typeName">;
  };
}

export type DynamicNodeData =
  | AcousticNodeData
  | AiGenNodeData
  | ChatNodeData
  | DocumentNodeData
  | GenericDynamicNodeData
  | VisualNodeData;

export type DynamicNodeType = Node<DynamicNodeData, AppNodeType.DYNAMIC>;

export interface GenericDynamicNodeData extends BaseDynamicNodeData {
  extension?: {
    case: undefined;
    value?: undefined;
  };
}

/**
 * SECTION 3: CORE NODE & GRAPH TYPES
 * The main building blocks of the Flowcraft editor.
 */

export type { Edge, Node, Viewport };

export interface LocalLLMClientConfig {
  apiKey: string;
  baseUrl: string;
  id: string;
  model: string;
  name: string;
}

export interface MediaDef {
  content?: string;
  galleryUrls?: string[];
  type: MediaType;
  url?: string;
}

export interface MutationLogEntry {
  description: string;
  id: string;
  mutations: GraphMutation[];
  source: MutationSource;
  taskId: string;
  timestamp: number;
}

export type NodeTemplate = ProtoNodeTemplate;

export interface ProcessingNodeData {
  [key: string]: unknown;
  label: string;
  onCancel?: (taskId: string) => void;
  taskId: string;
}
export type ProcessingNodeType = Node<
  ProcessingNodeData,
  AppNodeType.PROCESSING
>;

export interface TaskDefinition {
  createdAt: number;
  label: string;
  message: string;
  mutationIds: string[]; // Reference to log entries
  nodeId?: string; // Associated node if any
  progress: number;
  source: MutationSource;
  status: TaskStatus;
  taskId: string;
  type: TaskType;
  updatedAt: number;
}
export interface VisualNodeData extends BaseDynamicNodeData {
  extension: {
    case: "visual";
    value: Omit<VisualNodeState, "$typeName">;
  };
}

/**
 * SECTION 4: TEMPLATES & PROTOCOL PAYLOADS
 */

export interface WidgetDef {
  config?: Record<string, unknown>;
  id: string;
  inputPortId?: string;
  label: string;
  options?: { label: string; value: unknown }[];
  type: WidgetType;
  value: unknown;
}

/**
 * SECTION 5: TYPE GUARDS & UTILITIES
 */

export function isAiGenNode(
  node: AppNode,
): node is Node<AiGenNodeData, AppNodeType.DYNAMIC> {
  return isDynamicNode(node) && node.data.extension?.case === "aiGen";
}

export function isChatNode(
  node: AppNode,
): node is Node<ChatNodeData, AppNodeType.DYNAMIC> {
  return isDynamicNode(node) && node.data.extension?.case === "chat";
}

export function isDynamicNode(node: AppNode): node is DynamicNodeType {
  return node.type === AppNodeType.DYNAMIC;
}
