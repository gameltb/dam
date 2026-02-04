import { type NodeProps } from "@xyflow/react";
import { Loader2, X } from "lucide-react";
import { memo } from "react";

import { useFlowSocket } from "@/hooks/integration/useFlowSocket";
import { type ProcessingNodeType } from "@/types";

import { NodeShell } from "../base/NodeShell";

const ProcessingNodeContent: React.FC<{
  data: ProcessingNodeType["data"];
  id: string;
}> = memo(({ data, id }) => {
  const { cancelTask } = useFlowSocket();
  const progress = data.progress || 0;

  return (
    <div className="flex flex-col flex-1 p-4 min-w-[200px]">
      <div className="flex items-center justify-between mb-3">
        <div className="flex items-center gap-2">
          <Loader2 className="w-4 h-4 text-primary animate-spin" />
          <span className="text-[11px] font-bold uppercase tracking-wider text-primary">
            {data.displayName || "Processing"}
          </span>
        </div>
        <button
          className="p-1 hover:bg-destructive/10 text-muted-foreground hover:text-destructive rounded transition-colors"
          onClick={(e) => {
            e.stopPropagation();
            if (data.taskId) cancelTask(data.taskId);
          }}
          title="Cancel Task"
        >
          <X size={14} />
        </button>
      </div>

      <div className="flex flex-col gap-1.5 mb-4">
        <div className="text-[10px] text-muted-foreground font-medium truncate">
          {data.message || "Running remote task…"}
        </div>
        <div className="w-full h-1.5 bg-muted rounded-full overflow-hidden">
          <div className="h-full bg-primary transition-all duration-500 ease-out" style={{ width: `${progress}%` }} />
        </div>
        <div className="flex justify-end">
          <span className="text-[9px] font-mono text-primary/70">{Math.round(progress)}%</span>
        </div>
      </div>

      <div className="mt-auto pt-2 border-t border-node-border/50 text-[8px] text-muted-foreground/40 font-mono truncate">
        ID: {id}
      </div>
    </div>
  );
});

export const ProcessingNode = memo(({ data, id, selected }: NodeProps<ProcessingNodeType>) => {
  return (
    <NodeShell
      className="border-primary/30 shadow-[0_0_20px_rgba(var(--primary-rgb),0.15)]"
      nodeId={id}
      selected={selected}
    >
      <ProcessingNodeContent data={data} id={id} />
    </NodeShell>
  );
});

ProcessingNode.displayName = "ProcessingNode";
