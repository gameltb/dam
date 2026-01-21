import { AlertCircle, Loader2, RefreshCcw } from "lucide-react";
import React from "react";

import { useNodeController } from "@/hooks/useNodeController";
import { cn } from "@/lib/utils";

export interface BaseNodeProps {
  children?: React.ReactNode;
  className?: string;
  nodeId?: string; // Optional: enable controller integration
  style?: React.CSSProperties;
}

/**
 * BaseNode
 * A consistent container for all node types with built-in runtime feedback.
 */
export const BaseNode: React.FC<BaseNodeProps> = ({ children, className, nodeId, style }) => {
  const controller = nodeId ? useNodeController(nodeId) : null;
  const isBusy = controller?.status === "busy";
  const isError = controller?.status === "error";

  return (
    <div
      className={cn(
        "relative w-full h-full rounded-xl bg-background border border-node-border shadow-md flex flex-col transition-all",
        isBusy && "border-primary/50 shadow-lg",
        isError && "border-destructive/50",
        className,
      )}
      style={{
        overflow: "visible",
        ...style,
      }}
    >
      {children}

      {/* Synchronized Status Overlay */}
      {(isBusy || isError) && (
        <div className="absolute inset-0 z-[100] flex flex-col items-center justify-center rounded-[inherit] bg-background/60 backdrop-blur-[2px] p-4 text-center animate-in fade-in duration-200">
          {isBusy ? (
            <>
              <Loader2 className="w-6 h-6 text-primary animate-spin mb-2" />
              <div className="text-[10px] font-bold text-primary uppercase tracking-wider mb-1">
                {controller.message || "Processing..."}
              </div>
              <div className="w-2/3 h-1 bg-muted rounded-full overflow-hidden">
                <div
                  className="h-full bg-primary transition-all duration-300 ease-out"
                  style={{ width: `${controller.progress}%` }}
                />
              </div>
            </>
          ) : (
            <>
              <AlertCircle className="w-6 h-6 text-destructive mb-2" />
              <div className="text-[10px] font-bold text-destructive uppercase mb-1">Execution Failed</div>
              <div className="text-[9px] text-muted-foreground line-clamp-2 px-2 italic">
                {controller.error || "Unknown runtime error"}
              </div>
              <button
                className="mt-3 flex items-center gap-1.5 px-3 py-1.5 bg-destructive/10 hover:bg-destructive/20 text-destructive rounded-full text-[10px] font-medium transition-colors"
                onClick={(e) => {
                  e.stopPropagation();
                  controller.reset();
                }}
              >
                <RefreshCcw className="w-3 h-3" />
                Force Reset
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
};
