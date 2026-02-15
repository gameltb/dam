import { type NodeChange, type XYPosition } from "@xyflow/react";
import { useCallback } from "react";

import { useFlowStore } from "@/store/flowStore";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { type AppNode } from "@/types";

interface FlowHandlersProps {
  calculateLines: (
    draggingNode: AppNode,
    allNodes: AppNode[],
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
      const { nodesById } = useFlowStore.getState();

      // 1. Map changes to a new array, applying snapping without mutation
      const nextChanges = changes.map((change) => {
        if (change.type === "position" && change.dragging && change.position) {
          const node = nodesById[change.id];
          if (node) {
            const { helperLines, snappedPosition } = calculateLines(node, nodes, false, change.position);
            setHelperLines(helperLines);

            return {
              ...change,
              position: {
                x: snappedPosition.x ?? change.position.x,
                y: snappedPosition.y ?? change.position.y,
              },
            };
          }
        }
        return change;
      });

      // 2. Clear helper lines if not dragging
      const isDragging = changes.some((c) => c.type === "position" && c.dragging);
      if (!isDragging) {
        setHelperLines({ horizontal: undefined, vertical: undefined });
      }

      // 3. Pass the new, pure data to the store for transient update
      onNodesChange(nextChanges);
    },
    [onNodesChange, calculateLines, setHelperLines, nodes],
  );

  const handleNodeDragStop = useCallback(
    (_e: React.MouseEvent, node: AppNode) => {
      console.debug("[Handlers] Node drag stop - Persisting...", node.id);
      const { commitNodes, moveNodeToScope, nodesById, reparentNode } = useFlowStore.getState();
      const { activeScopeId } = useNavigationStore.getState();

      // 1. Persistent sync to DB/History
      commitNodes([node]);

      // 2. Logical Scope/Hierarchy logic
      const allNodesArray = Object.values(nodesById);
      const potentialParent = allNodesArray.find(
        (n) =>
          n.id !== node.id &&
          n.type === "groupNode" &&
          node.position &&
          node.position.x > 0 &&
          node.position.y > 0 &&
          node.position.x < (n.measured?.width ?? 0) &&
          node.position.y < (n.measured?.height ?? 0),
      );

      if (potentialParent && node.parentId !== potentialParent.id) {
        reparentNode(node.id, potentialParent.id);
      } else if (!potentialParent && node.parentId && node.parentId !== activeScopeId) {
        const padding = 20;
        const parent = nodesById[activeScopeId ?? ""];
        if (
          parent?.measured &&
          (node.position.x < -padding ||
            node.position.y < -padding ||
            node.position.x > parent.measured.width - padding ||
            node.position.y > parent.measured.height - padding)
        ) {
          moveNodeToScope(node.id, parent.scopeId || "root");
        }
      }

      contextMenuDragStop();
    },
    [contextMenuDragStop],
  );

  const onNodesDelete = useCallback((deletedNodes: AppNode[]) => {
    console.debug(
      "[Handlers] Nodes deleted",
      deletedNodes.map((n) => n.id),
    );
    const { deleteNodes } = useFlowStore.getState();
    deleteNodes(deletedNodes.map((n) => n.id));
  }, []);

  const onInit = useCallback(() => {
    console.log("[Flow] Canvas Initialized");
  }, []);

  return {
    handleMove: () => {},
    handleNodeDragStop,
    onConnectEnd: () => {},
    onConnectStart: () => {},
    onInit,
    onNodesChangeWithSnapping,
    onNodesDelete,
  };
};
