import {
  BaseEdge,
  EdgeLabelRenderer,
  type EdgeProps,
  getBezierPath,
} from "@xyflow/react";

export default function SystemEdge({
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
    <>
      <BaseEdge
        markerEnd={markerEnd}
        path={edgePath}
        style={{
          ...style,
          opacity: 0.6,
          stroke: "#b1b1b7", // 灰色
          strokeDasharray: "5, 5", // 虚线效果
          strokeWidth: 2,
        }}
      />
      <EdgeLabelRenderer children={null} />
    </>
  );
}
