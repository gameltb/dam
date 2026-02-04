import { useStore } from "@xyflow/react";
import React from "react";

import { type HelperLines } from "@/hooks/graph/useHelperLines";

interface HelperLinesRendererProps {
  lines: HelperLines;
}

/**
 * Renders alignment lines.
 * Uses the internal React Flow transform to ensure lines stay aligned with nodes
 * during panning and zooming.
 */
export const HelperLinesRenderer: React.FC<HelperLinesRendererProps> = ({ lines }) => {
  // Access the current viewport transform directly from the store
  const transform = useStore((s) => s.transform);
  let [tx, ty, zoom] = transform;

  // Defensive checks for NaN in transform
  if (isNaN(tx)) tx = 0;
  if (isNaN(ty)) ty = 0;
  if (isNaN(zoom)) zoom = 1;

  return (
    <svg
      style={{
        height: "100%",
        left: 0,
        overflow: "visible",
        pointerEvents: "none",
        position: "absolute",
        top: 0,
        width: "100%",
        zIndex: 10000, // Ensure lines are above everything else
      }}
    >
      <g transform={`translate(${tx.toString()}, ${ty.toString()}) scale(${zoom.toString()})`}>
        {lines.vertical !== undefined && (
          <line
            stroke="var(--primary-color)"
            strokeDasharray={`${(5 / zoom).toString()} ${(5 / zoom).toString()}`}
            strokeWidth={(1.5 / zoom).toString()}
            style={{ opacity: 1 }}
            x1={lines.vertical}
            x2={lines.vertical}
            y1="-1000000"
            y2="1000000"
          />
        )}
        {lines.horizontal !== undefined && (
          <line
            stroke="var(--primary-color)"
            strokeDasharray={`${(5 / zoom).toString()} ${(5 / zoom).toString()}`}
            strokeWidth={(1.5 / zoom).toString()}
            style={{ opacity: 1 }}
            x1="-1000000"
            x2="1000000"
            y1={lines.horizontal}
            y2={lines.horizontal}
          />
        )}
      </g>
    </svg>
  );
};
