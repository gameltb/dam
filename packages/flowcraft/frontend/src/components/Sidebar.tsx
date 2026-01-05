import React, { useRef, useCallback } from "react";
import { useUiStore } from "../store/uiStore";
import { useFlowStore } from "../store/flowStore";
import { ChatBot } from "./media/ChatBot";
import { X, PanelLeftClose } from "lucide-react";
import { Button } from "./ui/button";

export const Sidebar: React.FC = () => {
  const { 
    isSidebarOpen, 
    sidebarWidth, 
    activeChatNodeId, 
    setSidebarWidth,
    setActiveChat 
  } = useUiStore();
  
  const node = useFlowStore(s => s.nodes.find(n => n.id === activeChatNodeId));
  
  const isResizing = useRef(false);

  const handleMouseMove = useCallback((e: MouseEvent) => {
    if (!isResizing.current) return;
    const newWidth = window.innerWidth - e.clientX;
    if (newWidth > 300 && newWidth < 800) {
      setSidebarWidth(newWidth);
    }
  }, [setSidebarWidth]);

  const stopResizing = useCallback(() => {
    isResizing.current = false;
    document.removeEventListener("mousemove", handleMouseMove);
    document.removeEventListener("mouseup", stopResizing);
  }, [handleMouseMove]);

  const startResizing = useCallback((e: React.MouseEvent) => {
    e.preventDefault();
    isResizing.current = true;
    document.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseup", stopResizing);
  }, [handleMouseMove, stopResizing]);

  if (!isSidebarOpen) return null;

  return (
    <div 
      className="flex h-full border-l border-border bg-background transition-all duration-300 ease-in-out relative shadow-xl z-[4000]"
      style={{ width: sidebarWidth }}
    >
      {/* Resize Handle */}
      <div
        className="absolute left-0 top-0 w-1 h-full cursor-col-resize hover:bg-primary-color/50 transition-colors z-50"
        onMouseDown={startResizing}
      />

      <div className="flex flex-col w-full h-full overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-node-border bg-muted/30">
          <div className="flex items-center gap-2">
            <PanelLeftClose size={18} className="text-primary-color" />
            <h2 className="text-sm font-semibold truncate">
              {node ? `Chat: ${node.data.label || node.id}` : "AI Assistant"}
            </h2>
          </div>
          <Button 
            variant="ghost" 
            size="icon" 
            className="h-8 w-8"
            onClick={() => setActiveChat(null)}
          >
            <X size={16} />
          </Button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-hidden">
          {activeChatNodeId ? (
            <ChatBot nodeId={activeChatNodeId} />
          ) : (
            <div className="flex flex-col items-center justify-center h-full text-muted-foreground p-8 text-center">
              <p className="text-sm">Select a node with Chat enabled to view conversation here.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
