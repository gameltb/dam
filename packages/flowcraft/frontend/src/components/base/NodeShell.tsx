import React, { memo, useMemo, useRef } from "react";

import { NodeProvider } from "@/contexts/NodeContext";
import { useNodeController } from "@/hooks/nodes/useNodeController";
import { SizingStrategy, useNodeDimensionManager } from "@/hooks/nodes/useNodeDimensionManager";
import { useNodeVisibility } from "@/hooks/nodes/useNodeVisibility";
import { cn } from "@/lib/utils";

import { NodeStatus } from "@/types";

import { NodeStatusOverlay } from "../nodes/parts/NodeStatusOverlay";

export interface NodeShellProps {
  aspectRatio?: number;
  autoHeight?: boolean;
  children?: React.ReactNode;
  className?: string;
  isNested?: boolean; // prevents double styling when used inside another shell
  nodeId: string;
  onDoubleClick?: (e: React.MouseEvent) => void;
  selected?: boolean;
  sizingStrategy?: SizingStrategy;
  style?: React.CSSProperties;
}

/**
 * NodeShell
 * The authoritative visual foundation for all Flowcraft nodes.
 * Now acts as a NodeProvider to inject identity into the sub-tree.
 */
export const NodeShell: React.FC<NodeShellProps> = memo(
  ({
    aspectRatio,
    autoHeight,
    children,
    className,
    isNested = false,
    nodeId,
    onDoubleClick,
    selected,
    sizingStrategy,
    style,
  }) => {
    const controller = useNodeController(nodeId);
    const isBusy = controller.status === NodeStatus.PENDING || controller.status === NodeStatus.RUNNING;
    const isError = controller.status === NodeStatus.FAILED;

    const containerRef = useRef<HTMLDivElement>(null);
    const contentRef = useRef<HTMLDivElement>(null);

    // Viewport-driven hydration detection
    const isVisible = useNodeVisibility(containerRef);

    const effectiveStrategy = useMemo(() => {
      if (sizingStrategy) return sizingStrategy;
      if (aspectRatio && aspectRatio > 0) return SizingStrategy.ASPECT_RATIO;
      if (autoHeight) return SizingStrategy.CONTENT_FIT;
      return SizingStrategy.MANUAL;
    }, [sizingStrategy, aspectRatio, autoHeight]);

    useNodeDimensionManager(nodeId, contentRef, {
      ratio: aspectRatio,
      strategy: effectiveStrategy,
    });

    return (
      <NodeProvider nodeId={nodeId}>
        <div
          className={cn(
            "relative w-full h-full flex flex-col transition-all overflow-visible",
            !isNested && "rounded-xl bg-background border border-node-border shadow-md",
            !isNested && selected && "border-primary ring-2 ring-primary/20",
            !isNested && isBusy && "border-primary/50 shadow-lg",
            !isNested && isError && "border-destructive/50 shadow-[0_0_15px_rgba(239,68,68,0.2)]",
            className,
          )}
          onDoubleClick={onDoubleClick}
          ref={containerRef}
          style={{
            minHeight: isNested ? "0" : "80px",
            ...style,
          }}
        >
          <div
            className={cn(
              "flex flex-col flex-1 overflow-hidden",
              !isNested && "rounded-[inherit]",
              autoHeight ? "h-auto" : "h-full",
            )}
            ref={contentRef}
          >
            {isVisible ? (
              children
            ) : (
              <div className="flex-1 flex items-center justify-center bg-muted/5 animate-pulse">
                <div className="w-8 h-8 rounded-full border-2 border-primary/10 border-t-primary/30 animate-spin" />
              </div>
            )}
          </div>

          {!isNested && (
            <NodeStatusOverlay
              error={controller.error}
              isBusy={isBusy}
              isError={isError}
              message={controller.message}
              onReset={() => {
                controller.reset();
              }}
              progress={controller.progress}
            />
          )}
        </div>
      </NodeProvider>
    );
  },
);

NodeShell.displayName = "NodeShell";
