import React from "react";

export const itemStyle: React.CSSProperties = {
  padding: "8px 12px",
  cursor: "pointer",
  fontSize: "12px",
  color: "var(--text-color)",
  display: "flex",
  alignItems: "center",
  gap: "8px",
  transition: "background 0.2s",
};

export const sectionStyle: React.CSSProperties = {
  borderBottom: "1px solid var(--node-border)",
  paddingBottom: "4px",
  marginBottom: "4px",
};

export const labelStyle: React.CSSProperties = {
  ...itemStyle,
  cursor: "default",
  color: "var(--sub-text)",
};

export const handleMouseEnter = (e: React.MouseEvent) => {
  (e.currentTarget as HTMLElement).style.backgroundColor =
    "rgba(100, 108, 255, 0.15)";
};

export const handleMouseLeave = (e: React.MouseEvent) => {
  (e.currentTarget as HTMLElement).style.backgroundColor = "transparent";
};
