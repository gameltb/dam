import { create as createProto } from "@bufbuild/protobuf";
import { type Node, type NodeChange, type NodePositionChange, type XYPosition } from "@xyflow/react";
import { useCallback } from "react";

import { PositionSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { useFlowStore } from "@/store/flowStore";
import { commit } from "@/store/orchestrator";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode } from "@/types";

interface FlowHandlersProps {
  calculateLines: (
    draggingNode: Node<any, any>,
    allNodes: Node<any, any>[],
    shouldUpdateState: boolean,
    overriddenPosition?: XYPosition,
  ) => { helperLines: HelperLines; snappedPosition: XYPosition };
  contextMenuDragStop: () => void;
  nodes: AppNode[];
  onNodeContextMenuHook: (event: React.MouseEvent, node: AppNode) => void;
  onNodesChange: (changes: NodeChange<AppNode>[]) => void;
  setHelperLines: (lines: HelperLines) => void;
  updateViewport: (scopeId: string, x: number, y: number, zoom: number) => void;
}

interface HelperLines {
  horizontal?: number;
  vertical?: number;
}

export const useFlowHandlers = ({
  calculateLines,
  contextMenuDragStop,
  nodes,
  onNodeContextMenuHook: _onNodeContextMenuHook,
  onNodesChange,
  setHelperLines,
  updateViewport: _updateViewport,
}: FlowHandlersProps) => {
  const onNodesChangeWithSnapping = useCallback(
    (changes: NodeChange<AppNode>[]) => {
      const { nodesById, reparentNode } = useFlowStore.getState();
      const activeScopeId = useNavigationStore.getState().activeScopeId;

      const positionChange = changes.find((c): c is NodePositionChange => c.type === "position");

      if (positionChange && positionChange.dragging && positionChange.position) {
        const node = nodesById[positionChange.id];
        if (node) {
          const { helperLines, snappedPosition } = calculateLines(
            node as any,
            nodes as any,
            false,
            positionChange.position,
          );
          setHelperLines(helperLines);

          if (snappedPosition.x !== undefined) positionChange.position.x = snappedPosition.x;
          if (snappedPosition.y !== undefined) positionChange.position.y = snappedPosition.y;
        }
      } else {
        setHelperLines({ horizontal: undefined, vertical: undefined });
      }

      const dragStopChange = changes.find(
        (c): c is NodePositionChange => c.type === "position" && c.dragging === false,
      );

      if (dragStopChange) {
        const node = nodesById[dragStopChange.id];
        if (node) {
          const { moveNodeToScope } = useFlowStore.getState();
          const allNodesArray = Object.values(nodesById);
          const potentialParent = allNodesArray.find(
            (n) =>
              n.id !== node.id &&
              n.type === "groupNode" &&
              node.position.x > 0 &&
              node.position.y > 0 &&
              node.position.x < (n.measured?.width || 0) &&
              node.position.y < (n.measured?.height || 0),
          );

          if (potentialParent && node.parentId !== potentialParent.id) {
            reparentNode(node.id, potentialParent.id);
          } else if (!potentialParent && node.parentId && node.parentId !== activeScopeId) {
            const padding = 20;
            const parent = nodesById[activeScopeId || ""];
            // Perform edge detection only when the parent node is loaded and has dimension information
            if (
              parent?.measured &&
              (node.position.x < -padding ||
                node.position.y < -padding ||
                node.position.x > parent.measured.width - padding ||
                node.position.y > parent.measured.height - padding)
            ) {
              // This should be a logical Scope change; may need to decide between reparent or moveScope based on interaction logic
              // Current logic: Physical dragging out implies moving back to the previous logical level
              moveNodeToScope(node.id, parent.scopeId || "root");
            }
          }
        }
      }

      onNodesChange(changes);

      if (dragStopChange) {
        const node = nodesById[dragStopChange.id];
        if (node) {
          commit(
            (draft) => {
              const dn = draft.nodesById[node.id];
              if (dn?.presentation) {
                dn.presentation.position = createProto(PositionSchema, {
                  x: node.position.x,
                  y: node.position.y,
                });
              }
            },
            { description: "Drag stop sync" },
          );
        }
      }
    },
    [onNodesChange, calculateLines, setHelperLines, nodes],
  );

  const handleNodeDragStop = useCallback(() => {
    contextMenuDragStop();
  }, [contextMenuDragStop]);

  const onInit = useCallback((_rf: any) => {
    console.log("[Flow] Canvas Initialized");
  }, []);

  return {
    handleMove: () => {},
    handleNodeDragStop,
    onConnectEnd: () => {},
    onConnectStart: () => {},
    onInit,
    onNodesChangeWithSnapping,
  };
};
