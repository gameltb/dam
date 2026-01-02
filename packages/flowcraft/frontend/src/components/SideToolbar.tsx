import React from "react";
import { Settings } from "lucide-react";
import { useUiStore } from "../store/uiStore";

export const SideToolbar: React.FC = () => {
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen);

  return (
    <div
      style={{
        position: "fixed",
        left: "12px",
        top: "50%",
        transform: "translateY(-50%)",
        display: "flex",
        flexDirection: "column",
        gap: "6px",
        padding: "6px",
        backgroundColor: "var(--panel-bg)",
        border: "1px solid var(--node-border)",
        borderRadius: "10px",
        boxShadow: "0 4px 15px rgba(0,0,0,0.3)",
        zIndex: 5000,
        backdropFilter: "blur(10px)",
      }}
    >
      <button
        onClick={() => {
          setSettingsOpen(true);
        }}
        title="Settings"
        style={{
          width: "32px",
          height: "32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background: "none",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          color: "var(--sub-text)",
          transition: "all 0.2s",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "rgba(100, 108, 255, 0.1)";
          e.currentTarget.style.color = "var(--primary-color)";
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
          e.currentTarget.style.color = "var(--sub-text)";
        }}
      >
        <Settings size={18} />
      </button>
    </div>
  );
};
