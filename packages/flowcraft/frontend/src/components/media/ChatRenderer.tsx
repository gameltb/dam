import { Bot, Maximize2, PanelRight, RotateCcw } from "lucide-react";
import React from "react";

import { useFlowStore } from "@/store/flowStore";
import { useUiStore } from "@/store/uiStore";
import { ChatViewMode, type DynamicNodeData } from "@/types";

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
  const { chatViewMode, setActiveChat } = useUiStore();
  const updateNodeData = useFlowStore((s) => s.updateNodeData);
  const node = useFlowStore((s) => s.nodes.find((n) => n.id === nodeId));

  const isSidebar = chatViewMode === ChatViewMode.SIDEBAR;

  const startNewBranch = () => {
    const dynData = node?.data as DynamicNodeData | undefined;
    const chatExtension =
      dynData?.extension?.case === "chat" ? dynData.extension.value : null;
    updateNodeData(nodeId, {
      extension: {
        case: "chat",
        value: {
          conversationHeadId: "",
          isHistoryCleared: false,
          treeId: chatExtension?.treeId ?? "",
        },
      },
    });
  };

  return (
    <div className="ai-theme-container shadcn-lookup flex flex-col h-full bg-node-bg text-text-color rounded-lg overflow-hidden relative">
      {/* Header - Acts as drag handle except for buttons */}
      <div className="shrink-0 p-3 bg-muted/20 flex justify-between items-center border-b border-node-border cursor-grab active:cursor-grabbing">
        <div className="flex items-center gap-2 pointer-events-none">
          <Bot className="text-primary-color" size={16} />
          <h3 className="text-xs font-bold uppercase tracking-wider">
            {node?.data.label ?? "Chat Assistant"}
          </h3>
        </div>
        <div className="flex items-center gap-1 nodrag">
          <Button
            onClick={startNewBranch}
            size="icon-sm"
            title="Start New Branch (CoW)"
            variant="ghost"
          >
            <RotateCcw size={14} />
          </Button>
          <Button
            onClick={() => {
              setActiveChat(nodeId, ChatViewMode.FULLSCREEN);
            }}
            size="icon-sm"
            variant="ghost"
          >
            <Maximize2 size={14} />
          </Button>
          {!isSidebar && (
            <Button
              onClick={() => {
                setActiveChat(nodeId, ChatViewMode.SIDEBAR);
              }}
              size="icon-sm"
              variant="ghost"
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
            <Bot className="mb-4 opacity-20" size={32} />
            <p className="text-sm">
              This conversation is currently active in the sidebar.
            </p>
            <Button
              className="mt-4"
              onClick={() => {
                setActiveChat(nodeId, ChatViewMode.INLINE);
              }}
              size="sm"
              variant="outline"
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
