import { useStore } from "@xyflow/react";
import React from "react";

import { type HelperLines } from "@/hooks/useHelperLines";

interface HelperLinesRendererProps {
  lines: HelperLines;
}

/**
 * Renders alignment lines.
 * Uses the internal React Flow transform to ensure lines stay aligned with nodes
 * during panning and zooming.
 */
export const HelperLinesRenderer: React.FC<HelperLinesRendererProps> = ({
  lines,
}) => {
  // Access the current viewport transform directly from the store
  const transform = useStore((s) => s.transform);
  const [tx, ty, zoom] = transform;

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
        zIndex: 10,
      }}
    >
      <g
        transform={`translate(${tx.toString()}, ${ty.toString()}) scale(${zoom.toString()})`}
      >
        {lines.vertical !== undefined && (
          <line
            stroke="var(--primary-color)"
            strokeDasharray={`${(4 / zoom).toString()} ${(4 / zoom).toString()}`}
            strokeWidth={(1 / zoom).toString()} // Maintain constant line thickness
            style={{ opacity: 0.8 }}
            x1={lines.vertical}
            x2={lines.vertical}
            y1="-1000000"
            y2="1000000"
          />
        )}
        {lines.horizontal !== undefined && (
          <line
            stroke="var(--primary-color)"
            strokeDasharray={`${(4 / zoom).toString()} ${(4 / zoom).toString()}`}
            strokeWidth={(1 / zoom).toString()} // Maintain constant line thickness
            style={{ opacity: 0.8 }}
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
