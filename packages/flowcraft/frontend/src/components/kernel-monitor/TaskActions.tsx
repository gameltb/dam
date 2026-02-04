import { RotateCw, Square } from "lucide-react";
import React from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { type TaskDefinition, TaskStatus } from "@/types";

import { StatusBadge } from "./StatusIndicators";

interface TaskActionsProps {
  onCancel: (taskId: string) => void;
  onRestart: (nodeId: string) => void;
  task: TaskDefinition;
}

export const TaskActions: React.FC<TaskActionsProps> = ({ onCancel, onRestart, task }) => {
  return (
    <div className="px-6 py-4 border-b border-border bg-muted/5 flex justify-between items-center">
      <div className="flex items-center gap-4">
        <div className="flex flex-col gap-0.5">
          <div className="flex items-center gap-2">
            <h3 className="text-sm font-black tracking-tight">{task.label}</h3>
            <Badge
              className="text-[9px] font-black uppercase h-4 bg-primary/10 text-primary border-none"
              variant="outline"
            >
              {task.type}
            </Badge>
          </div>
          <code className="text-[10px] text-muted-foreground/60 font-mono tracking-tighter">ID: {task.taskId}</code>
        </div>

        <div className="h-8 w-[1px] bg-border mx-2" />

        <div className="flex flex-col gap-1">
          <div className="flex items-center gap-3">
            <StatusBadge status={task.status} />
            {(task.status === TaskStatus.RUNNING || task.status === TaskStatus.CLAIMED) && (
              <div className="flex items-center gap-2 w-40">
                <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden border border-border/30">
                  <div
                    className="h-full bg-primary transition-all duration-700 ease-out shadow-[0_0_8px_rgba(var(--primary),0.5)]"
                    style={{
                      width: `${String(task.progress)}%`,
                    }}
                  />
                </div>
                <span className="text-[10px] font-black font-mono w-8">{Math.round(task.progress)}%</span>
              </div>
            )}
          </div>
        </div>
      </div>

      <div className="flex gap-2">
        {task.nodeId && (
          <Button
            className="h-8 gap-2 text-[10px] font-bold uppercase tracking-wider"
            onClick={() => {
              if (task.nodeId) onRestart(task.nodeId);
            }}
            size="sm"
            variant="outline"
          >
            <RotateCw size={12} /> Restart
          </Button>
        )}
        {(task.status === TaskStatus.RUNNING ||
          task.status === TaskStatus.PENDING ||
          task.status === TaskStatus.CLAIMED) && (
          <Button
            className="h-8 gap-2 text-[10px] font-bold uppercase tracking-wider"
            onClick={() => {
              onCancel(task.taskId);
            }}
            size="sm"
            variant="destructive"
          >
            <Square size={12} /> Terminate
          </Button>
        )}
      </div>
    </div>
  );
};
