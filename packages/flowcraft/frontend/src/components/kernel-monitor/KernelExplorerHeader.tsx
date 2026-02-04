import { Cpu, X } from "lucide-react";
import React from "react";

import { Button } from "@/components/ui/button";

interface KernelExplorerHeaderProps {
  onClose: () => void;
  onMouseDown: (e: React.MouseEvent) => void;
  taskListCount: number;
  uptime: number;
  workersCount: number;
}

export const KernelExplorerHeader: React.FC<KernelExplorerHeaderProps> = ({
  onClose,
  onMouseDown,
  taskListCount,
  uptime,
  workersCount,
}) => {
  return (
    <>
      {/* Resize Handle */}
      <div
        className="absolute top-0 left-0 right-0 h-1 cursor-ns-resize hover:bg-primary/50 transition-colors z-50"
        onMouseDown={onMouseDown}
      />

      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 h-11 border-b border-border bg-muted/30">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Cpu className="text-primary" size={16} />
            <span className="text-xs font-black tracking-widest uppercase">Spacetime Kernel Explorer</span>
          </div>

          <div className="flex gap-4 items-center">
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/50 border border-border/50">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-[9px] font-mono font-bold uppercase tracking-tight">System Online</span>
            </div>
            <div className="flex gap-3 text-[10px] font-mono text-muted-foreground">
              <span>
                UPTIME: <span className="text-foreground">{uptime}s</span>
              </span>
              <span>
                TASKS: <span className="text-foreground">{taskListCount}</span>
              </span>
              <span>
                WORKERS: <span className="text-foreground">{workersCount}</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            aria-label="Close Explorer"
            className="h-7 w-7 rounded-md hover:bg-destructive/10 hover:text-destructive transition-colors"
            onClick={onClose}
            size="icon"
            variant="ghost"
          >
            <X size={16} />
          </Button>
        </div>
      </div>
    </>
  );
};
