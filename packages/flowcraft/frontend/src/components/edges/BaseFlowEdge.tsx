import { BaseEdge, getBezierPath, type EdgeProps } from "@xyflow/react";

/**
 * Standard edge with support for custom styling and animations.
 */
export const BaseFlowEdge = ({
  id,
  sourceX,
  sourceY,
  targetX,
  targetY,
  sourcePosition,
  targetPosition,
  style = {},
  markerEnd,
  data,
}: EdgeProps) => {
  const [edgePath] = getBezierPath({
    sourceX,
    sourceY,
    sourcePosition,
    targetX,
    targetY,
    targetPosition,
  });

  return (
    <BaseEdge
      id={id}
      path={edgePath}
      markerEnd={markerEnd}
      style={{
        ...style,
        strokeWidth: 2,
        stroke: (data?.color as string) || "#646cff",
        transition: "stroke 0.3s, stroke-width 0.3s",
      }}
    />
  );
};
