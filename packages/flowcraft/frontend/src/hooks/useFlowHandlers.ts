import { create as createProto } from "@bufbuild/protobuf";
import {
  type NodeChange,
  type OnConnectStartParams,
  type OnNodesChange,
  type ReactFlowInstance,
  type XYPosition,
} from "@xyflow/react";
import { useCallback, useRef } from "react";

import { ActionDiscoveryRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { PortMainType, PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { type AppNode, AppNodeType } from "@/types";
import { findPort } from "@/utils/nodeUtils";
import { socketClient } from "@/utils/SocketClient";

import { type HelperLines } from "./useHelperLines";

interface FlowHandlersProps {
  calculateLines: (
    node: AppNode,
    nodes: AppNode[],
    show: boolean,
    pos: XYPosition,
  ) => { helperLines: HelperLines; snappedPosition: XYPosition };
  contextMenuDragStop: () => void;
  nodes: AppNode[];
  onNodeContextMenuHook: (event: React.MouseEvent, node: AppNode) => void;
  onNodesChange: OnNodesChange<AppNode>;
  setHelperLines: (lines: HelperLines) => void;
  updateViewport: (x: number, y: number, zoom: number) => void;
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
  const setConnectionStartHandle = useUiStore((s) => s.setConnectionStartHandle);

  const handleMoveEnd = useCallback(
    (_: unknown, viewport: { x: number; y: number; zoom: number }) => {
      const x = isNaN(viewport.x) ? 0 : viewport.x;
      const y = isNaN(viewport.y) ? 0 : viewport.y;
      const zoom = isNaN(viewport.zoom) ? 1 : viewport.zoom;
      updateViewport(x, y, zoom);
    },
    [updateViewport],
  );

  const handleNodeDragStop = useCallback(
    (_: unknown, node: AppNode) => {
      setHelperLines({});
      contextMenuDragStop();

      const { allNodes, nodeDraft, reparentNode } = useFlowStore.getState();
      const { activeScopeId } = useUiStore.getState();

      // 1. Detect "Enter" logic
      const targetContainer = nodes.find(
        (n) =>
          n.id !== node.id &&
          n.type === AppNodeType.GROUP &&
          node.position.x > 0 &&
          node.position.x < (n.measured?.width || 0) &&
          node.position.y > 0 &&
          node.position.y < (n.measured?.height || 0),
      );

      if (targetContainer) {
        reparentNode(node.id, targetContainer.id);
        return;
      }

      // 2. Detect "Escape" logic
      if (activeScopeId) {
        const padding = -50;
        const parent = allNodes.find((n) => n.id === activeScopeId);
        if (
          node.position.x < padding ||
          node.position.y < padding ||
          (parent &&
            (node.position.x > (parent.measured?.width || 0) - padding ||
              node.position.y > (parent.measured?.height || 0) - padding))
        ) {
          reparentNode(node.id, parent?.parentId || null);
          return;
        }
      }

      // 3. Regular position update via ORM draft
      const storeNode = allNodes.find((n) => n.id === node.id);
      if (storeNode) {
        const res = nodeDraft(storeNode);
        if (res.ok) {
          res.value.presentation.position = createProto(PositionSchema, { x: node.position.x, y: node.position.y });
        }
      }
    },
    [setHelperLines, contextMenuDragStop, nodes],
  );

  const nodesRef = useRef(nodes);
  nodesRef.current = nodes;

  const onNodesChangeWithSnapping = useCallback(
    (changes: NodeChange[]) => {
      const currentNodes = nodesRef.current;
      const snappedChanges = changes.map((change) => {
        if (change.type === "position" && change.position) {
          const node = currentNodes.find((n) => n.id === change.id);
          if (node) {
            const { snappedPosition } = calculateLines(node, currentNodes, true, change.position);
            return { ...change, position: snappedPosition };
          }
        }
        return change;
      });
      onNodesChange(snappedChanges as any);
    },
    [onNodesChange, calculateLines],
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
            value: createProto(ActionDiscoveryRequestSchema, {
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
