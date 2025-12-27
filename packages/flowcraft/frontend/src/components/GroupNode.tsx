import { type NodeProps, NodeResizer, type Node } from "@xyflow/react";
import { memo } from "react";
import { useTheme } from "../hooks/useTheme";

export type GroupNodeData = {
  label?: string;
} & Record<string, unknown>;

export type GroupNodeType = Node<GroupNodeData, "groupNode">;

const GroupNode = ({ selected, data, id }: NodeProps<GroupNodeType>) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: isDark
          ? "rgba(255, 255, 255, 0.05)"
          : "rgba(0, 0, 0, 0.05)",
        border: `1px dashed ${selected ? "#646cff" : isDark ? "#444" : "#ccc"}`,
        borderRadius: "8px",
        padding: "10px",
        boxSizing: "border-box",
        position: "relative",
      }}
    >
      <NodeResizer
        color="#646cff"
        isVisible={selected}
        minWidth={100}
        minHeight={100}
      />
      <div
        style={{
          position: "absolute",
          top: "-25px",
          left: "0",
          fontSize: "12px",
          fontWeight: "bold",
          color: isDark ? "#aaa" : "#666",
        }}
      >
        {data.label ?? `Group ${id.slice(0, 4)}`}
      </div>
    </div>
  );
};

export default memo(GroupNode);
