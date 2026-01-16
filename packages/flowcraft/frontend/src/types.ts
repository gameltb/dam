import type { Edge, Node, Viewport } from "@xyflow/react";
import {
  type NodeData,
  type Port,
  PortStyle,
  type NodeTemplate as ProtoNodeTemplate,
  RenderMode,
  TaskStatus,
} from "@/generated/flowcraft/v1/core/node_pb";

import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import { MediaType, MutationSource, PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { type AiGenNodeState } from "@/generated/flowcraft/v1/nodes/ai_gen_node_pb";
import { type ChatNodeState } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import {
  type AcousticNodeState,
  type DocumentNodeState,
  type VisualNodeState,
} from "@/generated/flowcraft/v1/nodes/media_node_pb";

export { MediaType, MutationSource, PortMainType, PortStyle, RenderMode, TaskStatus };

/**
 * SECTION 0: CORE IDENTIFIER TYPES
 * Simple string types for identifiers.
 */
export type NodeId = string;
export type EdgeId = string;
export type TaskId = string;
export type TemplateId = string;

/**
 * SECTION 1: CORE DATA TYPES
 * Unified with Protobuf definitions.
 */

export type DynamicNodeData = NodeData & {
  templateId?: TemplateId; // Kept for convenient access in React Flow, but derived from Node message
  [key: string]: unknown;
};

export type AppNode = Node<DynamicNodeData | ProcessingNodeData | GroupNodeData, string> & {
  id: NodeId;
};
export type DynamicNodeType = Node<DynamicNodeData, "dynamic">;

export type { Edge, Node, Viewport };
export type NodeTemplate = ProtoNodeTemplate;

export enum AppNodeType {
  DYNAMIC = "dynamic",
  GROUP = "groupNode",
  PROCESSING = "processing",
}

export enum FlowEvent {
  GALLERY_ITEM_CONTEXT = "gallery-item-context",
  OPEN_EDITOR = "open-editor",
  OPEN_PREVIEW = "open-preview",
  PANE_CLICK = "pane-click",
  WIDGET_CLICK = "widget-click",
}

export enum ChatViewMode {
  FULLSCREEN = "fullscreen",
  INLINE = "inline",
  SIDEBAR = "sidebar",
}

export enum Theme {
  DARK = "dark",
  LIGHT = "light",
}

export enum OverflowMode {
  HIDDEN = "hidden",
  VISIBLE = "visible",
}

export enum VideoMode {
  FIT = "fit",
  ORIGINAL = "original",
}

export enum ChatStatus {
  ERROR = "error",
  READY = "ready",
  STREAMING = "streaming",
  SUBMITTED = "submitted",
}

export enum TaskType {
  ANONYMOUS = "anonymous",
  MANUAL = "manual",
  REMOTE = "remote-task",
  UNKNOWN = "unknown",
}

export enum DragMode {
  PAN = "pan",
  SELECT = "select",
}

export interface ProcessingNodeData extends NodeData {
  displayName: string;
  taskId: TaskId;
  [key: string]: unknown;
}

export type ProcessingNodeType = Node<ProcessingNodeData, AppNodeType.PROCESSING>;

export interface GroupNodeData extends NodeData {
  displayName: string;
  [key: string]: unknown;
}

export type GroupNodeType = Node<GroupNodeData, AppNodeType.GROUP>;

export interface MutationLogEntry {
  description: string;
  id: string;
  mutations: GraphMutation[];
  source: MutationSource;
  taskId: TaskId;
  timestamp: number;
}

export interface TaskDefinition {
  createdAt: number;
  label: string;
  message: string;
  mutationIds: string[];
  nodeId?: NodeId;
  progress: number;
  source: MutationSource;
  status: TaskStatus;
  taskId: TaskId;
  updatedAt: number;
  type?: TaskType;
}

export interface LocalLLMClientConfig {
  apiKey: string;
  baseUrl: string;
  id: string;
  model: string;
  name: string;
}

/**
 * SECTION 2: SPECIALIZED NODE DATA
 * Strictly typed extensions.
 */

export interface AcousticNodeData extends DynamicNodeData {
  extension: {
    case: "acoustic";
    value: AcousticNodeState;
  };
}

export interface AiGenNodeData extends DynamicNodeData {
  extension: {
    case: "aiGen";
    value: AiGenNodeState;
  };
}

export interface ChatNodeData extends DynamicNodeData {
  extension: {
    case: "chat";
    value: ChatNodeState;
  };
}

export interface DocumentNodeData extends DynamicNodeData {
  extension: {
    case: "document";
    value: DocumentNodeState;
  };
}

export interface VisualNodeData extends DynamicNodeData {
  extension: {
    case: "visual";
    value: VisualNodeState;
  };
}

export enum NodeSignalCase {
  CHAT_EDIT = "chatEdit",
  CHAT_GENERATE = "chatGenerate",
  CHAT_SWITCH = "chatSwitch",
  CHAT_SYNC = "chatSync",
  RESTART_INSTANCE = "restartInstance",
}

/**
 * SECTION 5: TYPE GUARDS & UTILITIES
 */

export function isDynamicNode(node: AppNode): node is DynamicNodeType {
  return node.type === AppNodeType.DYNAMIC;
}

export function isChatNode(node: AppNode): node is Node<ChatNodeData, string> {
  return isDynamicNode(node) && node.data.extension?.case === "chat";
}

export function isAiGenNode(node: AppNode): node is Node<AiGenNodeData, string> {
  return isDynamicNode(node) && node.data.extension?.case === "aiGen";
}

export type ClientPort = Port;
