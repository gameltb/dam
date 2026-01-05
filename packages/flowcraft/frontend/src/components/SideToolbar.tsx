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
    <div className="fc-panel fixed left-3 top-1/2 -translate-y-1/2 flex flex-col gap-1.5 p-1.5 z-[5000] shadow-lg backdrop-blur-md">
      <button
        onClick={() => {
          setSettingsOpen(true);
        }}
        title={`Settings (${statusText})`}
        className="w-8 h-8 flex items-center justify-center rounded-md cursor-pointer transition-all duration-300 hover:bg-primary/10"
        style={{
          background:
            connectionStatus !== SocketStatus.DISCONNECTED
              ? `radial-gradient(circle, ${statusColor} 0%, transparent 80%)`
              : "none",
          color:
            connectionStatus === SocketStatus.CONNECTED
              ? "var(--primary-color)"
              : "var(--sub-text)",
        }}
        onMouseEnter={(e) => {
          if (connectionStatus === SocketStatus.DISCONNECTED) {
            e.currentTarget.style.color = "var(--primary-color)";
          }
        }}
        onMouseLeave={(e) => {
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
