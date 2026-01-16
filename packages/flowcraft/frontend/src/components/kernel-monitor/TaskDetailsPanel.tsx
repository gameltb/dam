import { History } from "lucide-react";
import React from "react";

import { MutationSource, type TaskDefinition } from "@/types";

interface TaskDetailsPanelProps {
  identity: string;
  task: TaskDefinition;
}

export const TaskDetailsPanel: React.FC<TaskDetailsPanelProps> = ({ identity, task }) => {
  return (
    <div className="w-64 bg-muted/5 p-4 flex flex-col gap-4 overflow-y-auto">
      <section>
        <h4 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Context Identity</h4>
        <div className="p-2 rounded bg-muted/20 border border-border border-dashed">
          <code className="text-[9px] break-all block text-primary">{identity}</code>
        </div>
      </section>

      <section>
        <h4 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">Metadata</h4>
        <div className="space-y-2">
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">Source</span>
            <span className="font-mono">
              {task.source === MutationSource.SOURCE_USER ? "CLIENT_USER" : "WORKER_TASK"}
            </span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">Node ID</span>
            <span className="font-mono truncate ml-4 text-primary">{task.nodeId ?? "GLOBAL"}</span>
          </div>
          <div className="flex justify-between text-[10px]">
            <span className="text-muted-foreground">Registered</span>
            <span className="font-mono">{new Date(task.createdAt).toLocaleTimeString()}</span>
          </div>
        </div>
      </section>

      <div className="mt-auto p-3 rounded border border-border bg-background flex flex-col gap-2">
        <div className="flex items-center gap-2 text-[10px] font-bold">
          <History size={12} />
          <span>STDB_AUDIT_READY</span>
        </div>
        <p className="text-[9px] text-muted-foreground leading-relaxed italic">
          This task is being audited via the implicit client-task assignment protocol.
        </p>
      </div>
    </div>
  );
};
