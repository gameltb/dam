import React from "react";
import { Settings } from "lucide-react";
import { useUiStore } from "../store/uiStore";
import { SocketStatus } from "../utils/SocketClient";

interface SideToolbarProps {
  connectionStatus: SocketStatus;
}

const getStatusColor = (status: SocketStatus) => {
  switch (status) {
    case SocketStatus.CONNECTED:
      return "rgba(76, 175, 80, 0.4)";
    case SocketStatus.INITIALIZING:
      return "rgba(33, 150, 243, 0.4)"; // Blue for initializing
    case SocketStatus.CONNECTING:
      return "rgba(255, 235, 59, 0.4)";
    case SocketStatus.ERROR:
      return "rgba(244, 67, 54, 0.4)";
    case SocketStatus.DISCONNECTED:
    default:
      return "transparent";
  }
};

const getStatusText = (status: SocketStatus) => {
  switch (status) {
    case SocketStatus.CONNECTED:
      return "Ready";
    case SocketStatus.INITIALIZING:
      return "Synchronizing State...";
    case SocketStatus.CONNECTING:
      return "Connecting to Server...";
    case SocketStatus.ERROR:
      return "Connection Error";
    case SocketStatus.DISCONNECTED:
    default:
      return "Offline";
  }
};

export const SideToolbar: React.FC<SideToolbarProps> = ({
  connectionStatus,
}) => {
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen);

  const statusColor = getStatusColor(connectionStatus);
  const statusText = getStatusText(connectionStatus);

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
        title={`Settings (${statusText})`}
        style={{
          width: "32px",
          height: "32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          background:
            connectionStatus !== SocketStatus.DISCONNECTED
              ? `radial-gradient(circle, ${statusColor} 0%, transparent 80%)`
              : "none",
          border: "none",
          borderRadius: "6px",
          cursor: "pointer",
          color:
            connectionStatus === SocketStatus.CONNECTED
              ? "var(--primary-color)"
              : "var(--sub-text)",
          transition: "all 0.3s ease",
        }}
        onMouseEnter={(e) => {
          e.currentTarget.style.backgroundColor = "rgba(100, 108, 255, 0.1)";
          if (connectionStatus === SocketStatus.DISCONNECTED) {
            e.currentTarget.style.color = "var(--primary-color)";
          }
        }}
        onMouseLeave={(e) => {
          e.currentTarget.style.backgroundColor = "transparent";
          if (connectionStatus === SocketStatus.DISCONNECTED) {
            e.currentTarget.style.color = "var(--sub-text)";
          }
        }}
      >
        <Settings size={18} />
      </button>
    </div>
  );
};
