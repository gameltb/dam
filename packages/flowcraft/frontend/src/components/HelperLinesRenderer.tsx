import React from "react";
import { useStore } from "@xyflow/react";
import { type HelperLines } from "../hooks/useHelperLines";

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
        position: "absolute",
        width: "100%",
        height: "100%",
        pointerEvents: "none",
        zIndex: 10,
        overflow: "visible",
        left: 0,
        top: 0,
      }}
    >
      <g transform={`translate(${tx}, ${ty}) scale(${zoom})`}>
        {lines.vertical !== undefined && (
          <line
            x1={lines.vertical}
            y1="-1000000"
            x2={lines.vertical}
            y2="1000000"
            stroke="var(--primary-color)"
            strokeWidth={1 / zoom} // Maintain constant line thickness
            strokeDasharray={`${4 / zoom} ${4 / zoom}`}
            style={{ opacity: 0.8 }}
          />
        )}
        {lines.horizontal !== undefined && (
          <line
            x1="-1000000"
            y1={lines.horizontal}
            x2="1000000"
            y2={lines.horizontal}
            stroke="var(--primary-color)"
            strokeWidth={1 / zoom} // Maintain constant line thickness
            strokeDasharray={`${4 / zoom} ${4 / zoom}`}
            style={{ opacity: 0.8 }}
          />
        )}
      </g>
    </svg>
  );
};
