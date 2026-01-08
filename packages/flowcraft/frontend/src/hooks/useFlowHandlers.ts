import { create } from "@bufbuild/protobuf";
import {
  type Node,
  type NodeChange,
  type OnConnectStartParams,
  type ReactFlowInstance,
  type XYPosition,
} from "@xyflow/react";
import { useCallback } from "react";

import { ActionDiscoveryRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { PortMainType } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type AppNode } from "@/types";
import { findPort } from "@/utils/nodeUtils";
import { socketClient } from "@/utils/SocketClient";
import { type HelperLines } from "./useHelperLines";

interface FlowHandlersProps {
  calculateLines: (
    node: Node,
    nodes: Node[],
    show: boolean,
    pos: XYPosition,
  ) => { helperLines: HelperLines; snappedPosition: XYPosition };
  contextMenuDragStop: () => void;
  nodes: AppNode[];
  onNodeContextMenuHook: (event: React.MouseEvent, node: AppNode) => void;
  onNodesChange: (changes: NodeChange[]) => void;
  setHelperLines: (lines: HelperLines) => void;
  updateViewport: (
    x: number,
    y: number,
    zoom: number,
    width: number,
    height: number,
  ) => void;
}

export function useFlowHandlers({
  calculateLines,
  contextMenuDragStop,
  nodes,
  onNodeContextMenuHook,
  onNodesChange,
  setHelperLines,
  updateViewport,
}: FlowHandlersProps) {
  const setConnectionStartHandle = useUiStore(
    (s) => s.setConnectionStartHandle,
  );

  const handleMoveEnd = useCallback(
    (_: unknown, viewport: { x: number; y: number; zoom: number }) => {
      const { innerHeight, innerWidth } = window;
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
    (_: unknown, { handleId, handleType, nodeId }: OnConnectStartParams) => {
      setTimeout(() => {
        const store = useFlowStore.getState();
        const node = store.nodes.find((n) => n.id === nodeId);
        let portInfo = {
          itemType: "",
          mainType: PortMainType.ANY,
        };

        if (node) {
          const port = findPort(node, handleId ?? "");
          if (port?.type) {
            portInfo = {
              itemType: port.type.itemType,
              mainType: port.type.mainType,
            };
          }
        }

        if (handleType) {
          setConnectionStartHandle({
            handleId: handleId ?? "",
            nodeId: nodeId ?? "",
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
    handleNodeContextMenu,
    handleNodeDragStop,
    onConnectEnd,
    onConnectStart,
    onInit,
    onNodesChangeWithSnapping,
  };
}
