import React from "react";
import { useTheme } from "../hooks/useTheme";

type StatusPanelProps = {
  status: string;
  url: string;
  onClick: () => void;
  isOutOfSync?: boolean;
};

export const StatusPanel = ({
  status,
  url,
  onClick,
  isOutOfSync,
}: StatusPanelProps) => {
  const { theme } = useTheme();

  const panelStyle: React.CSSProperties = {
    position: "absolute",
    bottom: 5,
    left: 5,
    zIndex: 10,
    cursor: "pointer",
    fontSize: "12px",
    color:
      theme === "dark" ? "rgba(240, 240, 240, 0.5)" : "rgba(33, 53, 71, 0.5)",
    backgroundColor: isOutOfSync ? "rgba(255, 0, 0, 0.2)" : "transparent",
    padding: isOutOfSync ? "5px" : "0",
    borderRadius: isOutOfSync ? "5px" : "0",
  };

  const statusIndicatorStyle: React.CSSProperties = {
    display: "inline-block",
    width: 8,
    height: 8,
    borderRadius: "50%",
    marginRight: 6,
    backgroundColor:
      status === "Connected"
        ? isOutOfSync
          ? "orange"
          : "rgba(0, 255, 0, 0.7)"
        : "rgba(255, 0, 0, 0.7)",
  };

  return (
    <div style={panelStyle} onClick={onClick}>
      <span style={statusIndicatorStyle}></span>
      <span>{status}</span>
      <span style={{ marginLeft: 10, color: "#888" }}>{url}</span>
    </div>
  );
};
