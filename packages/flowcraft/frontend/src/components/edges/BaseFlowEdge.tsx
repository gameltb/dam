import { BaseEdge, type EdgeProps, getBezierPath } from "@xyflow/react";

/**
 * Standard edge with support for custom styling and animations.
 */
export const BaseFlowEdge = ({
  data,
  id,
  markerEnd,
  sourcePosition,
  sourceX,
  sourceY,
  style = {},
  targetPosition,
  targetX,
  targetY,
}: EdgeProps) => {
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
        stroke: (data?.color as string) || "#646cff",
        strokeWidth: 2,
        transition: "stroke 0.3s, stroke-width 0.3s",
      }}
    />
  );
};
