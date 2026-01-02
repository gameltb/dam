import { type NodeTypes, type EdgeTypes } from "@xyflow/react";
import GroupNode from "./components/GroupNode";
import { DynamicNode } from "./components/DynamicNode";
import ProcessingNode from "./components/ProcessingNode";
import SystemEdge from "./components/edges/SystemEdge";
import { BaseFlowEdge } from "./components/edges/BaseFlowEdge";

export const nodeTypes: NodeTypes = {
  groupNode: GroupNode,
  dynamic: DynamicNode,
  processing: ProcessingNode,
};

export const edgeTypes: EdgeTypes = {
  system: SystemEdge,
  default: BaseFlowEdge,
};

export const defaultEdgeOptions = {
  type: "default",
  animated: true,
  style: { strokeWidth: 2 },
};

export const snapGrid: [number, number] = [15, 15];
