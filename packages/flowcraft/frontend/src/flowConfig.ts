import { type EdgeTypes, type NodeTypes } from "@xyflow/react";

import { DynamicNode } from "./components/DynamicNode";
import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";
import { SystemEdge } from "./components/edges/SystemEdge";
import { GroupNode } from "./components/GroupNode";
import { ChatMessageNode } from "./components/nodes/ChatMessageNode";
import { PortalNode } from "./components/nodes/PortalNode";
import { ProcessingNode } from "./components/ProcessingNode";

export const nodeTypes: NodeTypes = {
  chatMessage: ChatMessageNode as any,
  dynamic: DynamicNode as any,
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
