import React from "react";
import { Bot, Maximize2, RotateCcw, PanelRight } from "lucide-react";
import { useUiStore } from "../../store/uiStore";
import { useFlowStore } from "../../store/flowStore";
import { Button } from "../ui/button";
import { ChatBot } from "./ChatBot";

interface ChatRendererProps {
  id: string;
}

/**
 * ChatRenderer acts as a display wrapper for the core ChatBot component.
 * It handles the frame, header, and switching between display modes.
 */
export const ChatRenderer: React.FC<ChatRendererProps> = ({ id: nodeId }) => {
  const { setActiveChat, chatViewMode } = useUiStore();
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));

  const isSidebar = chatViewMode === "sidebar";

  const resetHistory = () => {
    updateNodeData(nodeId, {
      metadata: { chat_history: "[]" },
    });
  };

  return (
    <div className="ai-theme-container shadcn-lookup flex flex-col h-full bg-node-bg text-text-color rounded-lg overflow-hidden relative">
      {/* Header - Acts as drag handle except for buttons */}
      <div className="shrink-0 p-3 bg-muted/20 flex justify-between items-center border-b border-node-border cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2 pointer-events-none">
          <Bot size={16} className="text-primary-color" />
          <h3 className="text-xs font-bold uppercase tracking-wider">
            {node?.data.label || "Chat Assistant"}
          </h3>
        </div>
        <div className="flex items-center gap-1 nodrag">
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={resetHistory}
            title="Reset History"
          >
            <RotateCcw size={14} />
          </Button>
          <Button
            variant="ghost"
            size="icon-sm"
            onClick={() => {
              setActiveChat(nodeId, "fullscreen");
            }}
          >
            <Maximize2 size={14} />
          </Button>
          {!isSidebar && (
            <Button
              variant="ghost"
              size="icon-sm"
              onClick={() => {
                setActiveChat(nodeId, "sidebar");
              }}
            >
              <PanelRight size={14} />
            </Button>
          )}
        </div>
      </div>

      {/* Core Chat Logic Component - Non-draggable area */}
      <div className="flex-1 overflow-hidden nodrag">
        {isSidebar ? (
          <div className="flex flex-col items-center justify-center h-full p-6 text-center text-muted-foreground bg-muted/5">
            <Bot size={32} className="mb-4 opacity-20" />
            <p className="text-sm">
              This conversation is currently active in the sidebar.
            </p>
            <Button
              variant="outline"
              size="sm"
              className="mt-4"
              onClick={() => {
                setActiveChat(nodeId, "inline");
              }}
            >
              Dock back to node
            </Button>
          </div>
        ) : (
          <ChatBot nodeId={nodeId} />
        )}
      </div>
    </div>
  );
};
