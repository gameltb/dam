import React from "react";
import {
  ReactFlow,
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  SelectionMode,
} from "@xyflow/react";
import {
  nodeTypes,
  edgeTypes,
  defaultEdgeOptions,
  snapGrid,
} from "../../flowConfig";
import { HelperLinesRenderer } from "../HelperLinesRenderer";
import { Notifications } from "../Notifications";
import { type AppNode } from "../../types";
import type {
  Edge as RFEdge,
  OnNodesChange,
  OnEdgesChange,
  OnConnect,
  OnMoveEnd,
  ColorMode,
} from "@xyflow/react";
import { type HelperLines } from "../../hooks/useHelperLines";

interface FlowCanvasProps {
  nodes: AppNode[];
  edges: RFEdge[];
  onNodesChange: OnNodesChange<AppNode>;
  onEdgesChange: OnEdgesChange;
  onConnect: OnConnect;
  onInit: (instance: any) => void;
  onNodeDragStart: (e: any, node: AppNode) => void;
  onNodeDragStop: (e: any, node: AppNode) => void;
  onConnectStart: (e: any, params: any) => void;
  onConnectEnd: (e: any) => void;
  onNodeContextMenu: (e: any, node: AppNode) => void;
  onEdgeContextMenu: (e: any, edge: RFEdge) => void;
  onSelectionContextMenu: (e: any, nodes: AppNode[]) => void;
  onPaneContextMenu: (e: any) => void;
  onMoveEnd: OnMoveEnd;
  theme: string;
  dragMode: "pan" | "select";
  helperLines: HelperLines;
}

export const FlowCanvas: React.FC<FlowCanvasProps> = (props) => {
  return (
    <ReactFlow
      nodes={props.nodes}
      edges={props.edges}
      onNodesChange={props.onNodesChange}
      onEdgesChange={props.onEdgesChange}
      onConnect={props.onConnect}
      onInit={props.onInit}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      onNodeDragStart={props.onNodeDragStart}
      onNodeDragStop={props.onNodeDragStop}
      onConnectStart={props.onConnectStart}
      onConnectEnd={props.onConnectEnd}
      onNodeContextMenu={props.onNodeContextMenu}
      onEdgeContextMenu={props.onEdgeContextMenu}
      onSelectionContextMenu={props.onSelectionContextMenu}
      onPaneContextMenu={props.onPaneContextMenu}
      onMoveEnd={props.onMoveEnd}
      fitView
      colorMode={props.theme as ColorMode}
      selectionMode={SelectionMode.Partial}
      panOnDrag={props.dragMode === "pan" ? [0, 1] : [1]}
      selectionOnDrag={props.dragMode === "select"}
      selectNodesOnDrag={false}
      snapToGrid={false}
      snapGrid={snapGrid}
      defaultEdgeOptions={defaultEdgeOptions}
    >
      <Background variant={BackgroundVariant.Dots} gap={15} size={1} />
      <Controls />
      <MiniMap
        style={{ borderRadius: "8px", overflow: "hidden" }}
        maskColor="var(--xy-minimap-mask-background-color)"
      />
      <Notifications />
      <HelperLinesRenderer lines={props.helperLines} />
    </ReactFlow>
  );
};
