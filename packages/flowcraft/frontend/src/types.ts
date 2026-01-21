import type { Edge, Node, Viewport } from "@xyflow/react";

import { MediaType, MutationSource, PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { type Presentation } from "@/generated/flowcraft/v1/core/base_pb";
import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import {
  type NodeData,
  type Port,
  PortStyle,
  type NodeTemplate as ProtoNodeTemplate,
  RenderMode,
} from "@/generated/flowcraft/v1/core/node_pb";
import { type AiGenNodeState } from "@/generated/flowcraft/v1/nodes/ai_gen_node_pb";
import { type ChatNodeState } from "@/generated/flowcraft/v1/nodes/chat_node_pb";
import {
  type AcousticNodeState,
  type DocumentNodeState,
  type VisualNodeState,
} from "@/generated/flowcraft/v1/nodes/media_node_pb";

export { MediaType, MutationSource, PortMainType, PortStyle, RenderMode, TaskStatus };

export enum AppNodeType {
  CHAT_MESSAGE = "chatMessage",
  DYNAMIC = "dynamic",
  GROUP = "groupNode",
  PROCESSING = "processing",
}

export enum ChatStatus {
  ERROR = "error",
  READY = "ready",
  STREAMING = "streaming",
  SUBMITTED = "submitted",
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

export enum NodeSignalCase {
  CHAT_EDIT = "chatEdit",
  CHAT_GENERATE = "chatGenerate",
  CHAT_SWITCH = "chatSwitch",
  CHAT_SYNC = "chatSync",
  RESTART_INSTANCE = "restartInstance",
}

export enum OverflowMode {
  HIDDEN = "hidden",
  VISIBLE = "visible",
}

export type { Edge, Node, Viewport };

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

/**
 * AppNode 现在是 React Flow Node 与 Protobuf Node 的合体
 */
export type AppNode = {
  id: NodeId;
  // 直接包含 PB Presentation，用于 ORM 自动生成路径
  presentation: Presentation;
} & Node<DynamicNodeData | GroupNodeData | ProcessingNodeData, string>;

export interface ChatNodeData extends DynamicNodeData {
  extension: {
    case: "chat";
    value: ChatNodeState;
  };
}

export type ClientPort = Port;

export interface DocumentNodeData extends DynamicNodeData {
  extension: {
    case: "document";
    value: DocumentNodeState;
  };
}

export type DynamicNodeData = NodeData & {
  [key: string]: unknown;
  managedScopeId?: string;
  templateId?: TemplateId;
};

export type DynamicNodeType = Node<DynamicNodeData, "dynamic">;

export type EdgeId = string;

export interface GroupNodeData extends NodeData {
  [key: string]: unknown;
  displayName: string;
  managedScopeId?: string;
  templateId?: TemplateId;
}

export type GroupNodeType = Node<GroupNodeData, AppNodeType.GROUP>;

export interface LocalLLMClientConfig {
  apiKey: string;
  baseUrl: string;
  id: string;
  model: string;
  name: string;
}

export interface MutationLogEntry {
  description: string;
  id: string;
  mutationsJson: string;
  source: MutationSource;
  taskId: TaskId;
  timestamp: number;
}

export type NodeId = string;

export type NodeTemplate = ProtoNodeTemplate;

export interface ProcessingNodeData extends NodeData {
  [key: string]: unknown;
  displayName: string;
  message?: string;
  progress?: number;
  taskId: TaskId;
  templateId?: TemplateId;
}

export type ProcessingNodeType = Node<ProcessingNodeData, AppNodeType.PROCESSING>;

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
  type?: TaskType;
  updatedAt: number;
}

export type TaskId = string;

export type TemplateId = string;

export interface VisualNodeData extends DynamicNodeData {
  extension: {
    case: "visual";
    value: VisualNodeState;
  };
}

export function isAiGenNode(node: AppNode): node is Node<AiGenNodeData, string> & { presentation: Presentation } {
  return isDynamicNode(node) && node.data.extension?.case === "aiGen";
}

export function isChatNode(node: AppNode): node is Node<ChatNodeData, string> & { presentation: Presentation } {
  return isDynamicNode(node) && node.data.extension?.case === "chat";
}

export function isDynamicNode(node: AppNode): node is DynamicNodeType & { presentation: Presentation } {
  return node.type === AppNodeType.DYNAMIC;
}
