import { useCallback, useState } from "react";
import { type Node, type XYPosition } from "@xyflow/react";

export interface HelperLines {
  horizontal?: number;
  vertical?: number;
}

const SNAP_DISTANCE = 10;

/**
 * Hook to calculate snapping and helper lines using absolute coordinates.
 */
export function useHelperLines(): {
  helperLines: HelperLines;
  setHelperLines: React.Dispatch<React.SetStateAction<HelperLines>>;
  calculateLines: (
    draggingNode: Node,
    allNodes: Node[],
    shouldUpdateState: boolean,
  ) => { snappedPosition: XYPosition; helperLines: HelperLines };
} {
  const [helperLines, setHelperLines] = useState<HelperLines>({});

  // Helper to find absolute position of a node
  const getAbsolutePosition = (node: Node, allNodes: Node[]): XYPosition => {
    let x = node.position.x;
    let y = node.position.y;
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
    (draggingNode: Node, allNodes: Node[], shouldUpdateState: boolean) => {
      const result: HelperLines = {
        horizontal: undefined,
        vertical: undefined,
      };

      // Calculate dragging node's absolute position
      const draggingAbsPos = getAbsolutePosition(draggingNode, allNodes);

      const draggingRect = {
        width: draggingNode.measured?.width ?? 0,
        height: draggingNode.measured?.height ?? 0,
        x: draggingAbsPos.x,
        y: draggingAbsPos.y,
      };

      const draggingNodes = [
        { x: draggingRect.x, type: "left" },
        { x: draggingRect.x + draggingRect.width / 2, type: "v-center" },
        { x: draggingRect.x + draggingRect.width, type: "right" },
        { y: draggingRect.y, type: "top" },
        { y: draggingRect.y + draggingRect.height / 2, type: "h-center" },
        { y: draggingRect.y + draggingRect.height, type: "bottom" },
      ];

      // Final snapped absolute position
      const snappedAbsPos: XYPosition = { ...draggingAbsPos };

      for (const node of allNodes) {
        if (node.id === draggingNode.id) continue;

        const nodeAbsPos = getAbsolutePosition(node, allNodes);
        const nodeRect = {
          width: node.measured?.width ?? 0,
          height: node.measured?.height ?? 0,
          x: nodeAbsPos.x,
          y: nodeAbsPos.y,
        };

        const targetNodes = [
          { x: nodeRect.x, type: "left" },
          { x: nodeRect.x + nodeRect.width / 2, type: "v-center" },
          { x: nodeRect.x + nodeRect.width, type: "right" },
          { y: nodeRect.y, type: "top" },
          { y: nodeRect.y + nodeRect.height / 2, type: "h-center" },
          { y: nodeRect.y + nodeRect.height, type: "bottom" },
        ];

        // Snap X
        for (const drag of draggingNodes) {
          if (drag.x === undefined) continue;
          for (const target of targetNodes) {
            if (target.x === undefined) continue;
            if (Math.abs(drag.x - target.x) < SNAP_DISTANCE) {
              result.vertical = target.x;
              if (drag.type === "left") snappedAbsPos.x = target.x;
              else if (drag.type === "v-center")
                snappedAbsPos.x = target.x - draggingRect.width / 2;
              else if (drag.type === "right")
                snappedAbsPos.x = target.x - draggingRect.width;
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
              else if (drag.type === "h-center")
                snappedAbsPos.y = target.y - draggingRect.height / 2;
              else if (drag.type === "bottom")
                snappedAbsPos.y = target.y - draggingRect.height;
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
      return { snappedPosition: snappedRelativePos, helperLines: result };
    },
    [],
  );

  return {
    helperLines,
    setHelperLines,
    calculateLines,
  };
}
