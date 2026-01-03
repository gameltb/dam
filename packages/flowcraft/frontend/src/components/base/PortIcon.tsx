import React from "react";
import { PortStyle } from "../../generated/flowcraft/v1/node_pb";

export const PortIcon: React.FC<{
  style: PortStyle;
  mainType?: string;
  color: string;
  isConnected: boolean;
}> = ({ style, mainType, color, isConnected }) => {
  const baseStyle: React.CSSProperties = {
    width: "10px",
    height: "10px",
    background: isConnected ? color : "var(--node-bg)",
    border: `2px solid ${color}`,
    boxSizing: "border-box",
    transition: "all 0.2s ease",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    overflow: "hidden",
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
            width: "100%",
            height: "2px",
            background: isConnected ? "white" : color,
            opacity: 0.8,
          }}
        />
        <div
          style={{
            width: "100%",
            height: "2px",
            background: isConnected ? "white" : color,
            opacity: 0.8,
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
            width: "4px",
            height: "4px",
            borderRadius: "50%",
            background: isConnected ? "white" : color,
          }}
        />
      </div>
    );
  }
  switch (style) {
    case PortStyle.SQUARE:
      return <div style={{ ...baseStyle, borderRadius: "2px" }} />;
    case PortStyle.DIAMOND:
      return (
        <div
          style={{
            ...baseStyle,
            transform: "rotate(45deg) scale(0.8)",
            borderRadius: "1px",
          }}
        />
      );
    case PortStyle.DASH:
      return (
        <div
          style={{
            ...baseStyle,
            borderStyle: "dashed",
            borderRadius: "50%",
            background: "transparent",
          }}
        />
      );
    default:
      return <div style={{ ...baseStyle, borderRadius: "50%" }} />;
  }
};
