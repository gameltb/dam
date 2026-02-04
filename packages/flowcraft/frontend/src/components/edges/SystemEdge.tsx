import { BaseEdge, type EdgeProps, getBezierPath } from "@xyflow/react";

export function SystemEdge({
  id,
  markerEnd,
  sourcePosition,
  sourceX,
  sourceY,
  style = {},
  targetPosition,
  targetX,
  targetY,
}: EdgeProps) {
  const [edgePath] = getBezierPath({
    sourcePosition,
    sourceX,
    sourceY,
    targetPosition,
    targetX,
    targetY,
  });

  return (
    <BaseEdge
      id={id}
      markerEnd={markerEnd}
      path={edgePath}
      style={{
        ...style,
        opacity: 0.6,
        stroke: "#b1b1b7", // Grey
        strokeDasharray: "5, 5", // Dashed effect
        strokeWidth: 2,
      }}
    />
  );
}
