import { ArrowLeftCircle, FileJson, Settings } from "lucide-react";
import React, { useRef } from "react";

import { useImportExport } from "@/hooks/graph/useImportExport";
import { useNavigation } from "@/hooks/graph/useNavigation";
import { useNavigationStore } from "@/store/ui/navigationStore";
import { useUiStore } from "@/store/uiStore";
import { SocketStatus } from "@/utils/SocketClient";

interface SideToolbarProps {
  connectionStatus: SocketStatus;
}

const getStatusColor = (status: SocketStatus) => {
  switch (status) {
    case SocketStatus.CONNECTED:
      return "rgba(76, 175, 80, 0.4)";
    case SocketStatus.CONNECTING:
      return "rgba(255, 235, 59, 0.4)";
    case SocketStatus.ERROR:
      return "rgba(244, 67, 54, 0.4)";
    case SocketStatus.INITIALIZING:
      return "rgba(33, 150, 243, 0.4)";
    case SocketStatus.DISCONNECTED:
    default:
      return "transparent";
  }
};

const getStatusText = (status: SocketStatus) => {
  switch (status) {
    case SocketStatus.CONNECTED:
      return "Ready";
    case SocketStatus.CONNECTING:
      return "Connecting to Server…";
    case SocketStatus.ERROR:
      return "Connection Error";
    case SocketStatus.INITIALIZING:
      return "Synchronizing State…";
    case SocketStatus.DISCONNECTED:
    default:
      return "Offline";
  }
};

export const SideToolbar: React.FC<SideToolbarProps> = ({ connectionStatus }) => {
  const setSettingsOpen = useUiStore((s) => s.setSettingsOpen);
  const activeScopeId = useNavigationStore((s) => s.activeScopeId);
  const { goBack } = useNavigation();

  const { importConversations } = useImportExport();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const statusColor = getStatusColor(connectionStatus);
  const statusText = getStatusText(connectionStatus);

  const handleImportClick = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      importConversations(content);
      e.target.value = "";
    };
    reader.readAsText(file);
  };

  return (
    <div className="fc-panel fixed left-3 top-1/2 -translate-y-1/2 flex flex-col gap-1.5 p-1.5 z-[5000] shadow-lg backdrop-blur-md">
      <input accept=".json" className="hidden" onChange={handleFileChange} ref={fileInputRef} type="file" />

      {activeScopeId && (
        <button
          className="w-8 h-8 flex items-center justify-center rounded-md cursor-pointer transition-all duration-300 hover:bg-primary/10 text-primary animate-in fade-in zoom-in"
          onClick={goBack}
          title="Exit Subgraph (Go to Parent)"
          type="button"
        >
          <ArrowLeftCircle size={20} />
        </button>
      )}

      <button
        className="w-8 h-8 flex items-center justify-center rounded-md cursor-pointer transition-all duration-300 hover:bg-primary/10 text-muted-foreground hover:text-primary"
        onClick={handleImportClick}
        title="Import Conversations (.json)"
        type="button"
      >
        <FileJson size={18} />
      </button>

      <button
        className="w-8 h-8 flex items-center justify-center rounded-md cursor-pointer transition-all duration-300 hover:bg-primary/10"
        onClick={() => {
          setSettingsOpen(true);
        }}
        style={{
          background:
            connectionStatus !== SocketStatus.DISCONNECTED
              ? `radial-gradient(circle, ${statusColor} 0%, transparent 80%)`
              : "none",
          color: connectionStatus === SocketStatus.CONNECTED ? "var(--primary-color)" : "var(--sub-text)",
        }}
        title={`Settings (${statusText})`}
        type="button"
      >
        <Settings size={18} />
      </button>
    </div>
  );
};
