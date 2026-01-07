import { type EdgeTypes, type NodeTypes } from "@xyflow/react";

import { DynamicNode } from "./components/DynamicNode";
import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";
import SystemEdge from "./components/edges/SystemEdge";
import GroupNode from "./components/GroupNode";
import ProcessingNode from "./components/ProcessingNode";

export const nodeTypes: NodeTypes = {
  dynamic: DynamicNode,
  groupNode: GroupNode,
  processing: ProcessingNode,
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
