import { flowcraft_proto } from '../flowcraft_proto';

// Re-export core types from the central bundle
export const core = flowcraft_proto.v1;
export type Node = flowcraft_proto.v1.INode;
export type NodeData = flowcraft_proto.v1.INodeData;
export type FlowMessage = flowcraft_proto.v1.IFlowMessage;
export type GraphSnapshot = flowcraft_proto.v1.IGraphSnapshot;
export type GraphMutation = flowcraft_proto.v1.IGraphMutation;
export type TaskUpdate = flowcraft_proto.v1.ITaskUpdate;
export const TaskStatus = flowcraft_proto.v1.TaskStatus;
export const WidgetType = flowcraft_proto.v1.WidgetType;
export const RenderMode = flowcraft_proto.v1.RenderMode;
export const MediaType = flowcraft_proto.v1.MediaType;
