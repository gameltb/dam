import { type Edge, type OnConnect, type OnEdgesChange, type OnNodesChange } from "@xyflow/react";

import { type AppNode, FlowEvent, InboundChangeCategory } from "@/types";
import { type PbConnection } from "@/utils/pb-client";

export interface AppAction {
  actionId: string;
  nodeId: string;
}

export type Result<T, E = string> = { error: E; ok: false } | { ok: true; value: T };

/**
 * RFState (V14.1 - Orchestrated & Fixed)
 */
export interface RFState {
  activeScopeId: null | string;
  addNode: (node: AppNode) => void;
  // Clipboard state MUST remain in store
  clipboard: null | { edges: Edge[]; nodes: AppNode[] };

  dispatchNodeEvent: (type: FlowEvent, payload: unknown) => void;
  edges: Edge[];

  edgesById: Record<string, Edge>;
  handleIncomingWidgetSignal: (payload: unknown) => void;

  // New: Track incoming change signals from DB
  lastInboundChange: null | {
    category: InboundChangeCategory;
    id: string; // node.id or 'viewport' etc.
    timestamp: number;
  };
  lastLocalUpdate: Record<string, number>;

  lastNodeEvent: null | { payload: unknown; timestamp: number; type: FlowEvent };

  moveNodeToScope: (nodeId: string, newScopeId: null | string) => void;
  nodeRelations: Record<
    string,
    {
      firstChildId?: string;
      left?: string;
      nextSiblingId?: string;
      parentId?: string;
      prevSiblingId?: string;
      right?: string;
    }
  >;

  nodes: AppNode[];
  nodesById: Record<string, AppNode>;
  onConnect: OnConnect;
  onEdgesChange: OnEdgesChange;
  onNodesChange: OnNodesChange<AppNode>;

  redo: () => void;

  redoStack: { edgesById: Record<string, Edge>; nodesById: Record<string, AppNode> }[];

  refreshView: () => void;

  reparentNode: (nodeId: string, newParentId: null | string) => void;

  resetStore: () => void;
  sendNodeSignal: (signal: unknown) => void;
  sendWidgetSignal: (signal: unknown) => void;
  setClipboard: (content: null | { edges: Edge[]; nodes: AppNode[] }) => void;
  setEdges: (edges: Edge[]) => void;
  setGraph: (g: { edges: Edge[]; nodes: AppNode[] }) => void;
  setNodes: (nodes: AppNode[]) => void;

  setSpacetimeConn: (conn: null | PbConnection) => void;
  spacetimeConn: null | PbConnection;
  takeSnapshot: () => void;
  undo: () => void;
  undoStack: { edgesById: Record<string, Edge>; nodesById: Record<string, AppNode> }[];

  viewport: { x: number; y: number; zoom: number };
}
