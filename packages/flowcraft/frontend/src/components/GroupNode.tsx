import { type NodeProps, NodeResizer, type Node } from "@xyflow/react";
import { memo, useState } from "react";
import { useTheme } from "../hooks/useTheme";
import { useFlowStore } from "../store/flowStore";
import { useShallow } from "zustand/react/shallow";
import { AppNodeType } from "../types";

export type GroupNodeData = {
  label?: string;
} & Record<string, unknown>;

export type GroupNodeType = Node<GroupNodeData, AppNodeType.GROUP>;

const GroupNode = ({ selected, data, id }: NodeProps<GroupNodeType>) => {
  const { theme } = useTheme();
  const isDark = theme === "dark";
  const updateNodeData = useFlowStore(useShallow((s) => s.updateNodeData));

  const [label, setLabel] = useState(data.label ?? "");
  const [isFocused, setIsFocused] = useState(false);

  // Update local state when external data changes
  const [prevDataLabel, setPrevDataLabel] = useState(data.label);
  if (data.label !== prevDataLabel) {
    setLabel(data.label ?? "");
    setPrevDataLabel(data.label);
  }

  const handleBlur = () => {
    setIsFocused(false);
    if (label !== data.label) {
      // Correctly update the label within the data object
      updateNodeData(id, { label });
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") {
      (e.target as HTMLInputElement).blur();
      e.stopPropagation();
    }
  };

  return (
    <div
      style={{
        width: "100%",
        height: "100%",
        backgroundColor: isDark
          ? "rgba(100, 108, 255, 0.08)"
          : "rgba(100, 108, 255, 0.05)",
        border: `2px ${selected ? "solid" : "dashed"} ${selected ? "var(--primary-color)" : isDark ? "rgba(100, 108, 255, 0.3)" : "rgba(100, 108, 255, 0.4)"}`,
        borderRadius: "12px",
        padding: "10px",
        boxSizing: "border-box",
        position: "relative",
        transition: "all 0.2s ease",
      }}
    >
      <NodeResizer
        color="var(--primary-color)"
        isVisible={selected}
        minWidth={150}
        minHeight={100}
      />
      <div
        className="nodrag nopan"
        style={{
          position: "absolute",
          top: "-32px",
          left: "0",
          width: "100%",
          display: "flex",
          zIndex: 10,
        }}
      >
        <input
          className="nodrag nopan"
          value={label}
          placeholder="Unnamed Group"
          onFocus={() => {
            setIsFocused(true);
          }}
          onChange={(e) => {
            setLabel(e.target.value);
          }}
          onBlur={handleBlur}
          onKeyDown={handleKeyDown}
          onClick={(e) => {
            e.stopPropagation();
          }}
          style={{
            background: isFocused ? "var(--panel-bg)" : "rgba(20, 20, 20, 0.4)",
            border: `1px solid ${isFocused ? "var(--primary-color)" : "transparent"}`,
            borderRadius: "6px",
            fontSize: "12px",
            fontWeight: 600,
            color: isDark ? "#fff" : "#000",
            outline: "none",
            width: "fit-content",
            minWidth: "100px",
            padding: "4px 10px",
            backdropFilter: "blur(8px)",
            boxShadow: isFocused ? "0 4px 12px rgba(0,0,0,0.4)" : "none",
            transition: "all 0.2s ease",
            cursor: "text",
          }}
        />
      </div>
    </div>
  );
};
export default memo(GroupNode);
