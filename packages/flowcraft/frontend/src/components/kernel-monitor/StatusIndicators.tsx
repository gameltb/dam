import React from "react";

import { cn } from "@/lib/utils";
import { TaskStatus } from "@/types";

export const StatusBadge: React.FC<{ status: TaskStatus }> = ({ status }) => {
  const config: Record<number, { color: string; label: string }> = {
    [TaskStatus.TASK_CANCELLED]: {
      color: "bg-muted text-muted-foreground border-border",
      label: "Cancelled",
    },
    [TaskStatus.TASK_COMPLETED]: {
      color: "bg-green-500/10 text-green-500 border-green-500/20",
      label: "Completed",
    },
    [TaskStatus.TASK_FAILED]: {
      color: "bg-destructive/10 text-destructive border-destructive/20",
      label: "Failed",
    },
    [TaskStatus.TASK_PENDING]: {
      color: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
      label: "Pending",
    },
    [TaskStatus.TASK_PROCESSING]: {
      color: "bg-primary/10 text-primary border-primary/20",
      label: "Processing",
    },
    [TaskStatus.TASK_RESTARTING]: {
      color: "bg-blue-500/10 text-blue-500 border-blue-500/20",
      label: "Restarting",
    },
  };

  const { color, label } = config[status] ?? {
    color: "bg-muted",
    label: "Unknown",
  };

  return (
    <span className={cn("text-[10px] px-2 py-0.5 rounded border font-bold uppercase tracking-wider", color)}>
      {label}
    </span>
  );
};

export const StatusDot: React.FC<{ status: TaskStatus }> = ({ status }) => {
  const colors: Record<number, string> = {
    [TaskStatus.TASK_CANCELLED]: "bg-muted-foreground",
    [TaskStatus.TASK_COMPLETED]: "bg-green-500",
    [TaskStatus.TASK_FAILED]: "bg-destructive",
    [TaskStatus.TASK_PENDING]: "bg-yellow-500",
    [TaskStatus.TASK_PROCESSING]: "bg-primary animate-pulse",
    [TaskStatus.TASK_RESTARTING]: "bg-blue-500 animate-bounce",
  };

  return <div className={cn("w-1.5 h-1.5 rounded-full", colors[status] ?? "bg-muted")} />;
};
