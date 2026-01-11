import {
  type Connection,
  type Edge,
  type EdgeChange,
  type NodeChange,
} from "@xyflow/react";

import { type GraphMutation } from "@/generated/flowcraft/v1/core/service_pb";
import {
  type NodeSignal,
  type WidgetSignal,
} from "@/generated/flowcraft/v1/core/signals_pb";
import {
  type AppNode,
  type FlowEvent,
  MediaType,
  type MutationSource,
} from "@/types";
import { type DynamicNodeData } from "@/types";

export interface MutationContext {
  description?: string;
  source?: MutationSource;
  taskId?: string;
}

export interface NodeHandlers {
  onChange: (id: string, data: Record<string, unknown>) => void;
  onGalleryItemContext?: (
    nodeId: string,
    url: string,
    mediaType: MediaType,
    x: number,
    y: number,
  ) => void;
  onWidgetClick?: (nodeId: string, widgetId: string) => void;
}

export interface RFState {
  addNode: (node: AppNode) => void;
  applyMutations: (
    mutations: GraphMutation[],
    context?: MutationContext,
  ) => void;
  dispatchNodeEvent: (
    type: FlowEvent,
    payload: Record<string, unknown>,
  ) => void;

  edges: Edge[];
  handleIncomingWidgetSignal: (signal: WidgetSignal) => void;
  isLayoutDirty: boolean; // Flag to indicate if topological sort is needed

  lastNodeEvent: null | {
    payload: Record<string, unknown>;
    timestamp: number;
    type: FlowEvent;
  };
  nodes: AppNode[];

  onConnect: (connection: Connection) => void;
  onEdgesChange: (changes: EdgeChange[]) => void;
  onNodesChange: (changes: NodeChange[]) => void;

  resetStore: () => void;
  // --- Node & Widget Interaction Signals ---
  sendNodeSignal: (signal: NodeSignal) => void;
  sendWidgetSignal: (signal: WidgetSignal) => void;
  setGraph: (
    graph: { edges: Edge[]; nodes: AppNode[] },
    version: number,
  ) => void;

  spacetimeConn?: any;
  updateNodeData: (id: string, data: Partial<DynamicNodeData>) => void;
}
