import React from "react";

import { PortStyle } from "@/generated/flowcraft/v1/core/node_pb";

export const PortIcon: React.FC<{
  color: string;
  isConnected: boolean;
  mainType?: string;
  style: PortStyle;
}> = ({ color, isConnected, mainType, style }) => {
  const baseStyle: React.CSSProperties = {
    alignItems: "center",
    background: isConnected ? color : "var(--node-bg)",
    border: `2px solid ${color}`,
    boxSizing: "border-box",
    display: "flex",
    height: "10px",
    justifyContent: "center",
    overflow: "hidden",
    transition: "all 0.2s ease",
    width: "10px",
  };

  if (mainType === "list") {
    return (
      <div
        style={{
          ...baseStyle,
          borderRadius: "2px",
          flexDirection: "column",
          gap: "1px",
          padding: "1px",
        }}
      >
        <div
          style={{
            background: isConnected ? "white" : color,
            height: "2px",
            opacity: 0.8,
            width: "100%",
          }}
        />
        <div
          style={{
            background: isConnected ? "white" : color,
            height: "2px",
            opacity: 0.8,
            width: "100%",
          }}
        />
      </div>
    );
  }
  if (mainType === "set") {
    return (
      <div style={{ ...baseStyle, borderRadius: "50%" }}>
        <div
          style={{
            background: isConnected ? "white" : color,
            borderRadius: "50%",
            height: "4px",
            width: "4px",
          }}
        />
      </div>
    );
  }
  switch (style) {
    case PortStyle.DASH:
      return (
        <div
          style={{
            ...baseStyle,
            background: "transparent",
            borderRadius: "50%",
            borderStyle: "dashed",
          }}
        />
      );
    case PortStyle.DIAMOND:
      return (
        <div
          style={{
            ...baseStyle,
            borderRadius: "1px",
            transform: "rotate(45deg) scale(0.8)",
          }}
        />
      );
    case PortStyle.SQUARE:
      return <div style={{ ...baseStyle, borderRadius: "2px" }} />;
    default:
      return <div style={{ ...baseStyle, borderRadius: "50%" }} />;
  }
};
