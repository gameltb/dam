import {
  Background,
  BackgroundVariant,
  type ColorMode,
  Controls,
  MiniMap,
  type OnConnect,
  type OnConnectEnd,
  type OnConnectStart,
  type OnEdgesChange,
  type OnMoveEnd,
  type OnNodesChange,
  ReactFlow,
  type ReactFlowInstance,
  type Edge as RFEdge,
  SelectionMode,
} from "@xyflow/react";
import React from "react";

import { defaultEdgeOptions, edgeTypes, nodeTypes, snapGrid } from "@/flowConfig";
import { type HelperLines } from "@/hooks/useHelperLines";
import { type AppNode, DragMode, Theme } from "@/types";

import { HelperLinesRenderer } from "../HelperLinesRenderer";
import { Notifications } from "../Notifications";

interface FlowCanvasProps {
  dragMode: DragMode;
  edges: RFEdge[];
  helperLines: HelperLines;
  nodes: AppNode[];
  onConnect: OnConnect;
  onConnectEnd: OnConnectEnd;
  onConnectStart: OnConnectStart;
  onEdgeContextMenu: (e: React.MouseEvent, edge: RFEdge) => void;
  onEdgesChange: OnEdgesChange;
  onInit: (instance: ReactFlowInstance<AppNode>) => void;
  onMoveEnd: OnMoveEnd;
  onNodeContextMenu: (e: React.MouseEvent, node: AppNode) => void;
  onNodeDragStart: (e: React.MouseEvent, node: AppNode) => void;
  onNodeDragStop: (e: React.MouseEvent, node: AppNode) => void;
  onNodesChange: OnNodesChange<AppNode>;
  onPaneContextMenu: (e: MouseEvent | React.MouseEvent) => void;
  onSelectionContextMenu: (e: React.MouseEvent, nodes: AppNode[]) => void;
  theme: Theme;
}

export const FlowCanvas: React.FC<FlowCanvasProps> = (props) => {
  return (
    <ReactFlow<AppNode>
      colorMode={props.theme as ColorMode}
      defaultEdgeOptions={defaultEdgeOptions}
      edges={props.edges}
      edgeTypes={edgeTypes}
      maxZoom={2.5}
      minZoom={0.1}
      nodes={props.nodes}
      nodeTypes={nodeTypes}
      onConnect={props.onConnect}
      onConnectEnd={props.onConnectEnd}
      onConnectStart={props.onConnectStart}
      onEdgeContextMenu={props.onEdgeContextMenu}
      onEdgesChange={props.onEdgesChange}
      onInit={props.onInit}
      onMoveEnd={props.onMoveEnd}
      onNodeContextMenu={props.onNodeContextMenu}
      onNodeDragStart={props.onNodeDragStart}
      onNodeDragStop={props.onNodeDragStop}
      onNodesChange={props.onNodesChange}
      onPaneContextMenu={props.onPaneContextMenu}
      onSelectionContextMenu={props.onSelectionContextMenu}
      panOnDrag={props.dragMode === DragMode.PAN ? [0, 1] : [1]}
      selectionMode={SelectionMode.Partial}
      selectionOnDrag={props.dragMode === DragMode.SELECT}
      selectNodesOnDrag={false}
      snapGrid={snapGrid}
      snapToGrid={false}
      zoomOnPinch={true}
      zoomOnScroll={true}
    >
      <Background gap={15} size={1} variant={BackgroundVariant.Dots} />
      <Controls />
      <MiniMap
        maskColor="var(--xy-minimap-mask-background-color)"
        style={{ borderRadius: "8px", overflow: "hidden" }}
      />
      <Notifications />
      <HelperLinesRenderer lines={props.helperLines} />
    </ReactFlow>
  );
};
