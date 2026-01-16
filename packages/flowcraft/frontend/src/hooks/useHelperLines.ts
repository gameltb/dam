import { type Node, type XYPosition } from "@xyflow/react";
import { useCallback, useState } from "react";

export interface HelperLines {
  horizontal?: number;
  vertical?: number;
}

const SNAP_DISTANCE = 10;

/**
 * Hook to calculate snapping and helper lines using absolute coordinates.
 */
export function useHelperLines(): {
  calculateLines: (
    draggingNode: Node,
    allNodes: Node[],
    shouldUpdateState: boolean,
    overriddenPosition?: XYPosition,
  ) => { helperLines: HelperLines; snappedPosition: XYPosition };
  helperLines: HelperLines;
  setHelperLines: React.Dispatch<React.SetStateAction<HelperLines>>;
} {
  const [helperLines, setHelperLines] = useState<HelperLines>({});

  // Helper to find absolute position of a node
  const getAbsolutePosition = (node: Node, allNodes: Node[], overriddenPosition?: XYPosition): XYPosition => {
    let x = overriddenPosition ? overriddenPosition.x : node.position.x;
    let y = overriddenPosition ? overriddenPosition.y : node.position.y;
    let parentId = node.parentId;

    while (parentId) {
      const parent = allNodes.find((n) => n.id === parentId);
      if (parent) {
        x += parent.position.x;
        y += parent.position.y;
        parentId = parent.parentId;
      } else {
        break;
      }
    }
    return { x, y };
  };

  const calculateLines = useCallback(
    (draggingNode: Node, allNodes: Node[], shouldUpdateState: boolean, overriddenPosition?: XYPosition) => {
      const result: HelperLines = {
        horizontal: undefined,
        vertical: undefined,
      };

      // We only want to snap against nodes that are NOT being dragged
      const targetNodesForSnapping = allNodes.filter((n) => n.id !== draggingNode.id && !n.dragging);

      // Calculate dragging node's absolute position
      const draggingAbsPos = getAbsolutePosition(draggingNode, allNodes, overriddenPosition);

      const draggingRect = {
        height: draggingNode.measured?.height ?? 0,
        width: draggingNode.measured?.width ?? 0,
        x: draggingAbsPos.x,
        y: draggingAbsPos.y,
      };

      const draggingNodes = [
        { type: "left", x: draggingRect.x },
        { type: "v-center", x: draggingRect.x + draggingRect.width / 2 },
        { type: "right", x: draggingRect.x + draggingRect.width },
        { type: "top", y: draggingRect.y },
        { type: "h-center", y: draggingRect.y + draggingRect.height / 2 },
        { type: "bottom", y: draggingRect.y + draggingRect.height },
      ];

      // Final snapped absolute position
      const snappedAbsPos: XYPosition = { ...draggingAbsPos };

      for (const node of targetNodesForSnapping) {
        const nodeAbsPos = getAbsolutePosition(node, allNodes);
        const nodeRect = {
          height: node.measured?.height ?? 0,
          width: node.measured?.width ?? 0,
          x: nodeAbsPos.x,
          y: nodeAbsPos.y,
        };

        const targetNodes = [
          { type: "left", x: nodeRect.x },
          { type: "v-center", x: nodeRect.x + nodeRect.width / 2 },
          { type: "right", x: nodeRect.x + nodeRect.width },
          { type: "top", y: nodeRect.y },
          { type: "h-center", y: nodeRect.y + nodeRect.height / 2 },
          { type: "bottom", y: nodeRect.y + nodeRect.height },
        ];

        // Snap X
        for (const drag of draggingNodes) {
          if (drag.x === undefined) continue;
          for (const target of targetNodes) {
            if (target.x === undefined) continue;
            if (Math.abs(drag.x - target.x) < SNAP_DISTANCE) {
              result.vertical = target.x;
              if (drag.type === "left") snappedAbsPos.x = target.x;
              else if (drag.type === "v-center") snappedAbsPos.x = target.x - draggingRect.width / 2;
              else if (drag.type === "right") snappedAbsPos.x = target.x - draggingRect.width;
            }
          }
        }

        // Snap Y
        for (const drag of draggingNodes) {
          if (drag.y === undefined) continue;
          for (const target of targetNodes) {
            if (target.y === undefined) continue;
            if (Math.abs(drag.y - target.y) < SNAP_DISTANCE) {
              result.horizontal = target.y;
              if (drag.type === "top") snappedAbsPos.y = target.y;
              else if (drag.type === "h-center") snappedAbsPos.y = target.y - draggingRect.height / 2;
              else if (drag.type === "bottom") snappedAbsPos.y = target.y - draggingRect.height;
            }
          }
        }
      }

      if (shouldUpdateState) {
        setHelperLines(result);
      }

      // Convert absolute snapped position back to relative for the dragging node
      const snappedRelativePos = { ...snappedAbsPos };
      if (draggingNode.parentId) {
        const parent = allNodes.find((n) => n.id === draggingNode.parentId);
        if (parent) {
          const parentAbsPos = getAbsolutePosition(parent, allNodes);
          snappedRelativePos.x -= parentAbsPos.x;
          snappedRelativePos.y -= parentAbsPos.y;
        }
      }
      return { helperLines: result, snappedPosition: snappedRelativePos };
    },
    [],
  );

  return {
    calculateLines,
    helperLines,
    setHelperLines,
  };
}
