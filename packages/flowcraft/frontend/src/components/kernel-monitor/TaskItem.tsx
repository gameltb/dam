import { Settings, User } from "lucide-react";
import React from "react";

import { cn } from "@/lib/utils";
import { MutationSource, type TaskDefinition, TaskStatus } from "@/types";

import { StatusDot } from "./StatusIndicators";

interface TaskItemProps {
  isSelected: boolean;
  onClick: () => void;
  task: TaskDefinition;
}

export const TaskItem: React.FC<TaskItemProps> = ({ isSelected, onClick, task }) => {
  return (
    <div
      className={cn(
        "flex flex-col p-3 border-b border-border/50 cursor-pointer transition-all border-l-2",
        isSelected ? "bg-accent border-l-primary" : "hover:bg-muted/50 border-l-transparent",
      )}
      onClick={onClick}
    >
      <div className="flex justify-between items-center mb-1">
        <div className="flex items-center gap-1.5 overflow-hidden">
          {task.source === MutationSource.SOURCE_USER ? (
            <User className="text-blue-400 shrink-0" size={10} />
          ) : (
            <Settings className="text-green-400 shrink-0" size={10} />
          )}
          <span className="text-[10px] font-mono text-muted-foreground truncate">{task.taskId.split("-")[0]}</span>
        </div>
        <span className="text-[9px] font-mono opacity-40">{new Date(task.createdAt).toLocaleTimeString()}</span>
      </div>
      <div className="text-xs font-semibold truncate leading-none mb-2">{task.label}</div>
      <div className="flex items-center gap-2">
        <StatusDot status={task.status} />
        <span className="text-[10px] text-muted-foreground capitalize">
          {TaskStatus[task.status] ? TaskStatus[task.status].replace("TASK_", "").toLowerCase() : "unknown"}
        </span>
      </div>
    </div>
  );
};
