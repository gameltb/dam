import { useCallback } from "react";
import {
  type OnConnectStartParams,
  type NodeChange,
  type ReactFlowInstance,
  type XYPosition,
  type Node,
} from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";
import { useUiStore } from "../store/uiStore";
import { socketClient } from "../utils/SocketClient";
import { create } from "@bufbuild/protobuf";
import { ActionDiscoveryRequestSchema } from "../generated/flowcraft/v1/core/action_pb";
import { PortMainType } from "../generated/flowcraft/v1/core/base_pb";
import { type AppNode } from "../types";
import { type HelperLines } from "./useHelperLines";
import { findPort } from "../utils/nodeUtils";

interface FlowHandlersProps {
  nodes: AppNode[];
  onNodesChange: (changes: NodeChange[]) => void;
  updateViewport: (
    x: number,
    y: number,
    zoom: number,
    width: number,
    height: number,
  ) => void;
  calculateLines: (
    node: Node,
    nodes: Node[],
    show: boolean,
    pos: XYPosition,
  ) => { snappedPosition: XYPosition; helperLines: HelperLines };
  setHelperLines: (lines: HelperLines) => void;
  onNodeContextMenuHook: (event: React.MouseEvent, node: AppNode) => void;
  contextMenuDragStop: () => void;
}

export function useFlowHandlers({
  nodes,
  onNodesChange,
  updateViewport,
  calculateLines,
  setHelperLines,
  onNodeContextMenuHook,
  contextMenuDragStop,
}: FlowHandlersProps) {
  const setConnectionStartHandle = useUiStore(
    (s) => s.setConnectionStartHandle,
  );

  const handleMoveEnd = useCallback(
    (_: unknown, viewport: { x: number; y: number; zoom: number }) => {
      const { innerWidth, innerHeight } = window;
      const x = -viewport.x / viewport.zoom;
      const y = -viewport.y / viewport.zoom;
      const width = innerWidth / viewport.zoom;
      const height = innerHeight / viewport.zoom;
      updateViewport(x, y, viewport.zoom, width, height);
    },
    [updateViewport],
  );

  const handleNodeDragStop = useCallback(() => {
    setHelperLines({});
    contextMenuDragStop();
  }, [setHelperLines, contextMenuDragStop]);

  const onNodesChangeWithSnapping = useCallback(
    (changes: NodeChange[]) => {
      const snappedChanges = changes.map((change) => {
        if (change.type === "position" && change.position) {
          const node = nodes.find((n) => n.id === change.id);
          if (node) {
            const { snappedPosition } = calculateLines(
              node,
              nodes,
              true,
              change.position,
            );
            return { ...change, position: snappedPosition };
          }
        }
        return change;
      });
      onNodesChange(snappedChanges);
    },
    [onNodesChange, nodes, calculateLines],
  );

  const onConnectStart = useCallback(
    (_: unknown, { nodeId, handleId, handleType }: OnConnectStartParams) => {
      setTimeout(() => {
        const store = useFlowStore.getState();
        const node = store.nodes.find((n) => n.id === nodeId);
        let portInfo = {
          mainType: PortMainType.ANY as number,
          itemType: "",
        };

        if (node) {
          const port = findPort(node, handleId ?? "");
          if (port?.type) {
            portInfo = {
              mainType: port.type.mainType as number,
              itemType: port.type.itemType,
            };
          }
        }

        if (handleType) {
          setConnectionStartHandle({
            nodeId: nodeId ?? "",
            handleId: handleId ?? "",
            type: handleType,
            ...portInfo,
          });
        }
      }, 0);
    },
    [setConnectionStartHandle],
  );

  const onConnectEnd = useCallback(() => {
    setTimeout(() => {
      setConnectionStartHandle(null);
    }, 0);
  }, [setConnectionStartHandle]);

  const handleNodeContextMenu = useCallback(
    (event: React.MouseEvent, node: AppNode) => {
      event.preventDefault();
      const target = event.target as HTMLElement;
      if (target.tagName === "INPUT" || target.tagName === "TEXTAREA") return;

      onNodeContextMenuHook(event, node);
      void socketClient
        .send({
          payload: {
            case: "actionDiscovery",
            value: create(ActionDiscoveryRequestSchema, {
              nodeId: node.id,
              selectedNodeIds: nodes.filter((n) => n.selected).map((n) => n.id),
            }),
          },
        })
        .catch((e: unknown) => {
          console.error("Failed to send action discovery", e);
        });
    },
    [onNodeContextMenuHook, nodes],
  );

  const onInit = useCallback((instance: ReactFlowInstance<AppNode>) => {
    console.log("React Flow Instance Ready", instance);
  }, []);

  return {
    handleMoveEnd,
    handleNodeDragStop,
    onNodesChangeWithSnapping,
    onConnectStart,
    onConnectEnd,
    handleNodeContextMenu,
    onInit,
  };
}
