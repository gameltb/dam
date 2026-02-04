import {
  Background,
  BackgroundVariant,
  Controls,
  MiniMap,
  type OnConnect,
  type OnConnectEnd,
  type OnConnectStart,
  type OnEdgesChange,
  type OnMove,
  type OnMoveEnd,
  type OnNodesChange,
  ReactFlow,
  type ReactFlowInstance,
  type Edge as RFEdge,
  SelectionMode,
} from "@xyflow/react";
import React, { useCallback } from "react";

import { defaultEdgeOptions, edgeTypes, nodeTypes } from "@/flowConfig";
import { useFileDrop } from "@/hooks/ux/useFileDrop";
import { useGraphMutation } from "@/hooks/graph/useGraphMutation";
import { type HelperLines } from "@/hooks/graph/useHelperLines";
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
  onMove?: OnMove;
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
  const { handleDragOver, handleDrop } = useFileDrop();
  const { updateViewport } = useGraphMutation();

  const handleMoveEnd: OnMoveEnd = useCallback(
    (_e, viewport) => {
      updateViewport(viewport.x, viewport.y, viewport.zoom);
      props.onMoveEnd?.(_e, viewport);
    },
    [updateViewport, props.onMoveEnd],
  );

  return (
    <div
      className="w-full h-full"
      onDragOver={handleDragOver}
      onDrop={handleDrop}
      style={{ touchAction: "manipulation" }}
    >
      <ReactFlow<AppNode>
        colorMode={props.theme as any}
        defaultEdgeOptions={defaultEdgeOptions}
        edges={props.edges}
        edgeTypes={edgeTypes}
        elevateNodesOnSelect={true}
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
        onMove={props.onMove}
        onMoveEnd={handleMoveEnd}
        onNodeContextMenu={props.onNodeContextMenu}
        onNodeDragStart={props.onNodeDragStart}
        onNodeDragStop={props.onNodeDragStop}
        onNodesChange={props.onNodesChange}
        onPaneContextMenu={props.onPaneContextMenu}
        onSelectionContextMenu={props.onSelectionContextMenu}
        panOnDrag={props.dragMode === DragMode.PAN ? [0, 1, 2] : [1, 2]}
        panOnScroll={false}
        selectionKeyCode={props.dragMode === DragMode.SELECT ? null : "Shift"}
        selectionMode={SelectionMode.Partial}
        selectionOnDrag={props.dragMode === DragMode.SELECT}
        zoomOnPinch={true}
        zoomOnScroll={true}
      >
        <Background gap={15} key="background" size={1} variant={BackgroundVariant.Dots} />
        <Controls key="controls" />
        <MiniMap
          key="minimap"
          maskColor="var(--xy-minimap-mask-background-color)"
          style={{ borderRadius: "8px", overflow: "hidden" }}
        />
        <Notifications key="notifications" />
        <HelperLinesRenderer key="helper-lines" lines={props.helperLines} />
      </ReactFlow>
    </div>
  );
};
