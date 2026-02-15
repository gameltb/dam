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
  useReactFlow,
} from "@xyflow/react";
import React, { memo, useCallback, useEffect } from "react";

import { defaultEdgeOptions, edgeTypes, nodeTypes } from "@/flowConfig";
import { useGraphMutation } from "@/hooks/graph/useGraphMutation";
import { type HelperLines } from "@/hooks/graph/useHelperLines";
import { useFileDrop } from "@/hooks/ux/useFileDrop";
import { useFlowStore } from "@/store/flowStore";
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
  onNodesDelete: (nodes: AppNode[]) => void;
  onPaneContextMenu: (e: MouseEvent | React.MouseEvent) => void;
  onSelectionContextMenu: (e: React.MouseEvent, nodes: AppNode[]) => void;
  theme: Theme;
}

export const FlowCanvas: React.FC<FlowCanvasProps> = memo((props) => {
  const { handleDragOver, handleDrop } = useFileDrop();
  const { updateViewport } = useGraphMutation();
  const { setEdges, setNodes, updateNodeData } = useReactFlow();

  // 1. Reactive Sync Bridge: Push business data changes to RF internal store
  useEffect(() => {
    const lastDataJson = new Map<string, string>();

    const unsubscribe = useFlowStore.subscribe((state) => {
      // PERFORMANCE GUARD: Skip heavy serialization/sync during high-frequency interactions
      const isInteracting = state.nodes.some((n) => n.dragging || (n as any).resizing);
      if (isInteracting) return;

      const nodesById = state.nodesById;

      Object.entries(nodesById).forEach(([id, node]) => {
        const appNode = node;
        const currentJson = JSON.stringify(appNode.data);
        if (lastDataJson.get(id) !== currentJson) {
          updateNodeData(id, appNode.data);
          lastDataJson.set(id, currentJson);
        }
      });
    });
    return unsubscribe;
  }, [updateNodeData]);

  // 2. Structural Sync: Handle adds/removes or external position updates (DB pushes)
  const activeScopeId = useFlowStore((s) => s.activeScopeId);
  useEffect(() => {
    // If scope changes, we force a complete reset of RF internal nodes
    setNodes(props.nodes);
  }, [activeScopeId, setNodes, props.nodes]);

  useEffect(() => {
    const unsubscribe = useFlowStore.subscribe((state) => {
      const newNodes = state.nodes;
      setNodes((currentRfNodes) => {
        return newNodes.map((n) => {
          const existing = currentRfNodes.find((rn) => rn.id === n.id);
          // If the node exists and is being interacted with, preserve its transient UI state
          if (existing && (existing.dragging || (existing as any).resizing)) {
            return { ...n, ...existing };
          }
          return n;
        });
      });
    });
    return unsubscribe;
  }, [setNodes]);

  // Sync Edges
  useEffect(() => {
    const unsubscribe = useFlowStore.subscribe((state) => {
      setEdges(state.edges);
    });
    return unsubscribe;
  }, [setEdges]);

  const onMoveEndCallback = props.onMoveEnd;

  const handleMoveEnd: OnMoveEnd = useCallback(
    (_e, viewport) => {
      updateViewport(viewport.x, viewport.y, viewport.zoom);
      onMoveEndCallback(_e, viewport);
    },
    [updateViewport, onMoveEndCallback],
  );

  return (
    <div
      className="w-full h-full"
      onDragOver={handleDragOver}
      onDrop={(e) => {
        e.preventDefault();
        e.stopPropagation();
        void handleDrop(e);
      }}
      style={{ touchAction: "manipulation" }}
    >
      <ReactFlow<AppNode>
        colorMode={(props.theme === Theme.DARK ? "dark" : "light") as unknown as undefined}
        defaultEdgeOptions={defaultEdgeOptions}
        defaultEdges={props.edges}
        // Uncontrolled Mode: We do not pass 'nodes' or 'edges' here.
        // They are initialized and updated via the sync bridge effects.
        defaultNodes={props.nodes}
        edgeTypes={edgeTypes}
        elevateNodesOnSelect={true}
        maxZoom={2.5}
        minZoom={0.1}
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
        onNodesDelete={props.onNodesDelete}
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
});

FlowCanvas.displayName = "FlowCanvas";
