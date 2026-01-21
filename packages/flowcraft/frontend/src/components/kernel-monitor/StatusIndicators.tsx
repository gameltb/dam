import React from "react";

import { cn } from "@/lib/utils";
import { TaskStatus } from "@/types";

export const StatusBadge: React.FC<{ status: TaskStatus }> = ({ status }) => {
  const config: Record<number, { color: string; label: string }> = {
    [TaskStatus.CANCELLED]: {
      color: "bg-muted text-muted-foreground border-border",
      label: "Cancelled",
    },
    [TaskStatus.COMPLETED]: {
      color: "bg-green-500/10 text-green-500 border-green-500/20",
      label: "Completed",
    },
    [TaskStatus.FAILED]: {
      color: "bg-destructive/10 text-destructive border-destructive/20",
      label: "Failed",
    },
    [TaskStatus.PENDING]: {
      color: "bg-yellow-500/10 text-yellow-500 border-yellow-500/20",
      label: "Pending",
    },
    [TaskStatus.RUNNING]: {
      color: "bg-primary/10 text-primary border-primary/20",
      label: "Processing",
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
    [TaskStatus.CANCELLED]: "bg-muted-foreground",
    [TaskStatus.COMPLETED]: "bg-green-500",
    [TaskStatus.FAILED]: "bg-destructive",
    [TaskStatus.PENDING]: "bg-yellow-500",
    [TaskStatus.RUNNING]: "bg-primary animate-pulse",
  };

  return <div className={cn("w-1.5 h-1.5 rounded-full", colors[status] ?? "bg-muted")} />;
};
