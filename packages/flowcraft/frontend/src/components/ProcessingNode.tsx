import { type NodeProps } from "@xyflow/react";
import { Loader2, Terminal, XCircle } from "lucide-react";
import { memo } from "react";

import { TaskStatus } from "@/generated/flowcraft/v1/core/kernel_pb";
import { cn } from "@/lib/utils";
import { type ProcessingNodeType } from "@/types";

import { BaseNode } from "./base/BaseNode";
import { Button } from "./ui/button";
import { Progress } from "./ui/progress";

export const ProcessingNode = memo(({ data, selected: _selected }: NodeProps<ProcessingNodeType>) => {
  const isFailed = data.status === TaskStatus.FAILED;
  const isCancelled = data.status === TaskStatus.CANCELLED;
  const isPending = data.status === TaskStatus.PENDING;
  const isRunning = data.status === TaskStatus.RUNNING;

  return (
    <BaseNode
      className={cn(
        "min-w-[280px] border-2",
        isRunning && "border-primary shadow-[0_0_15px_rgba(var(--primary-rgb),0.3)]",
        isFailed && "border-destructive",
        isCancelled && "border-muted-foreground/50",
      )}
    >
      <div className="p-4 flex flex-col gap-4">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            {isRunning ? (
              <Loader2 className="w-4 h-4 text-primary animate-spin" />
            ) : (
              <Terminal className="w-4 h-4 text-muted-foreground" />
            )}
            <span className="text-xs font-bold uppercase tracking-tight truncate max-w-[180px]">
              {data.displayName || "Processing..."}
            </span>
          </div>
          {isFailed && <XCircle className="w-4 h-4 text-destructive" />}
        </div>

        {/* Progress & Message */}
        <div className="flex flex-col gap-2">
          <div className="flex justify-between items-end text-[10px]">
            <span className="text-muted-foreground font-mono uppercase italic">
              {isPending ? "Waiting..." : isRunning ? "In Progress" : "Done"}
            </span>
            <span className="font-bold text-primary">{Math.round(data.progress || 0)}%</span>
          </div>
          <Progress className="h-1.5" value={data.progress} />
          <p className="text-[10px] text-muted-foreground italic line-clamp-2 min-h-[2.5em]">
            {data.message || "Initializing task pipeline..."}
          </p>
        </div>

        {/* Footer Actions */}
        {(isRunning || isPending) && (
          <div className="flex justify-end pt-2 border-t border-border/50">
            <Button className="h-7 text-[10px] text-destructive hover:bg-destructive/10" size="sm" variant="ghost">
              Cancel Task
            </Button>
          </div>
        )}
      </div>
    </BaseNode>
  );
});
