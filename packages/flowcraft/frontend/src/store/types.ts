import { type Edge, type OnConnect, type OnEdgesChange, type OnNodesChange } from "@xyflow/react";

import {
  type AddEdgeRequest,
  type AddNodeRequest,
  type AddSubGraphRequest,
  type ClearGraphRequest,
  type GraphMutation,
  type PathUpdateRequest,
  type RemoveEdgeRequest,
  type RemoveNodeRequest,
  type ReparentNodeRequest,
} from "@/generated/flowcraft/v1/core/service_pb";
import { type NodeSignal, type WidgetSignal } from "@/generated/flowcraft/v1/core/signals_pb";
import { type AppNode, type FlowEvent, MediaType, type MutationSource } from "@/types";
import { type Draftable, type Result } from "@/utils/draft";
import { type PbConnection } from "@/utils/pb-client";

/**
 * 历史记录条目
 */
export interface HistoryEntry {
  description: string;
  forward: MutationInput[];
  id: string;
  inverse: MutationInput[];
  scopeId: null | string;
  timestamp: number;
}

export interface MutationContext {
  description?: string;
  isHistoryOp?: boolean; // 标识是否为撤销重做触发的操作
  source?: MutationSource;
  taskId?: string;
}

export type MutationInput =
  | AddEdgeRequest
  | AddNodeRequest
  | AddSubGraphRequest
  | ClearGraphRequest
  | GraphMutation
  | PathUpdateRequest
  | RemoveEdgeRequest
  | RemoveNodeRequest
  | ReparentNodeRequest;

export interface NodeHandlers {
  onChange: (id: string, data: Record<string, unknown>) => void;
  onGalleryItemContext?: (nodeId: string, url: string, mediaType: MediaType, x: number, y: number) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
}

export interface RFState {
  // 基础指令
  addNode: (node: AppNode) => void;
  allEdges: Edge[];
  allNodes: AppNode[];
  applyMutations: (mutations: MutationInput[], context?: MutationContext) => void;

  dispatchNodeEvent: (type: FlowEvent, payload: Record<string, unknown>) => void;
  edges: Edge[];
  handleIncomingWidgetSignal: (signal: WidgetSignal) => void;
  lastLocalUpdate: Record<string, number>;
  lastNodeEvent: null | {
    payload: Record<string, unknown>;
    timestamp: number;
    type: FlowEvent;
  };
  // ORM 接口 (Rust-style Result)
  nodeDraft: (nodeIdOrNode: AppNode | string) => Result<Draftable<AppNode>>;

  nodes: AppNode[];

  onConnect: OnConnect;
  onEdgesChange: OnEdgesChange;
  onNodesChange: OnNodesChange<AppNode>;
  redo: () => void;

  redoStack: HistoryEntry[];
  refreshView: () => void;

  reparentNode: (nodeId: string, newParentId: null | string) => void;
  resetStore: () => void;
  sendNodeSignal: (signal: NodeSignal) => void;

  sendWidgetSignal: (signal: WidgetSignal) => void;
  setEdges: (edges: Edge[]) => void;
  setGraph: (graph: { edges: Edge[]; nodes: AppNode[] }) => void;
  setNodes: (nodes: AppNode[]) => void;

  spacetimeConn?: PbConnection;

  // 历史指令
  undo: () => void;
  undoStack: HistoryEntry[];
}
