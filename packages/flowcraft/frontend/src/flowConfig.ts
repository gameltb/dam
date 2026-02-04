import { type EdgeTypes, type NodeTypes } from "@xyflow/react";

import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";
import { SystemEdge } from "./components/edges/SystemEdge";
import { ChatMessageNode } from "./components/nodes/ChatMessageNode";
import { NodeAssembler } from "./components/nodes/core/NodeAssembler";
import { GroupNode } from "./components/nodes/GroupNode";
import { initNodeRegistry } from "./components/nodes/implementations";
import { PortalNode } from "./components/nodes/PortalNode";
import { ProcessingNode } from "./components/nodes/ProcessingNode";

// Initialize the registry
initNodeRegistry();

export const nodeTypes: NodeTypes = {
  chatMessage: ChatMessageNode as any,
  dynamic: NodeAssembler as any,
  groupNode: GroupNode as any,
  portal: PortalNode as any,
  processing: ProcessingNode as any,
};

export const edgeTypes: EdgeTypes = {
  default: BaseFlowEdge,
  system: SystemEdge,
};

export const defaultEdgeOptions = {
  animated: true,
  style: { strokeWidth: 2 },
  type: "default",
};

export const snapGrid: [number, number] = [15, 15];
