import { type Node, type NodeProps, NodeResizer } from "@xyflow/react";
import { memo, useState } from "react";
import { useShallow } from "zustand/react/shallow";

import { useTheme } from "@/hooks/useTheme";
import { useFlowStore } from "@/store/flowStore";
import { AppNodeType } from "@/types";

export type GroupNodeData = Record<string, unknown> & {
  label?: string;
};

export type GroupNodeType = Node<GroupNodeData, AppNodeType.GROUP>;

const GroupNode = ({ data, id, selected }: NodeProps<GroupNodeType>) => {
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
        backgroundColor: isDark
          ? "rgba(100, 108, 255, 0.08)"
          : "rgba(100, 108, 255, 0.05)",
        border: `2px ${selected ? "solid" : "dashed"} ${selected ? "var(--primary-color)" : isDark ? "rgba(100, 108, 255, 0.3)" : "rgba(100, 108, 255, 0.4)"}`,
        borderRadius: "12px",
        boxSizing: "border-box",
        height: "100%",
        padding: "10px",
        position: "relative",
        transition: "all 0.2s ease",
        width: "100%",
      }}
    >
      <NodeResizer
        color="var(--primary-color)"
        isVisible={selected}
        minHeight={100}
        minWidth={150}
      />
      <div
        className="nodrag nopan"
        style={{
          display: "flex",
          left: "0",
          position: "absolute",
          top: "-32px",
          width: "100%",
          zIndex: 10,
        }}
      >
        <input
          className="nodrag nopan"
          onBlur={handleBlur}
          onChange={(e) => {
            setLabel(e.target.value);
          }}
          onClick={(e) => {
            e.stopPropagation();
          }}
          onFocus={() => {
            setIsFocused(true);
          }}
          onKeyDown={handleKeyDown}
          placeholder="Unnamed Group"
          style={{
            backdropFilter: "blur(8px)",
            background: isFocused ? "var(--panel-bg)" : "rgba(20, 20, 20, 0.4)",
            border: `1px solid ${isFocused ? "var(--primary-color)" : "transparent"}`,
            borderRadius: "6px",
            boxShadow: isFocused ? "0 4px 12px rgba(0,0,0,0.4)" : "none",
            color: isDark ? "#fff" : "#000",
            cursor: "text",
            fontSize: "12px",
            fontWeight: 600,
            minWidth: "100px",
            outline: "none",
            padding: "4px 10px",
            transition: "all 0.2s ease",
            width: "fit-content",
          }}
          value={label}
        />
      </div>
    </div>
  );
};
export default memo(GroupNode);
