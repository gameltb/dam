import { AlertTriangle, Clock, Info, List, XCircle } from "lucide-react";
import React from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";
import { cn } from "@/lib/utils";

import { ScrollArea } from "../ui/scroll-area";

export const KernelLogViewer: React.FC = () => {
  const [logs] = useTable(tables.taskAuditLog);

  const sortedLogs = [...logs].sort((a, b) => Number(b.timestamp - a.timestamp));

  const getLogIcon = (type: string) => {
    switch (type.toLowerCase()) {
      case "error":
        return <XCircle className="text-destructive" size={12} />;
      case "warn":
      case "warning":
        return <AlertTriangle className="text-yellow-500" size={12} />;
      default:
        return <Info className="text-blue-500" size={12} />;
    }
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background font-mono">
      <div className="px-4 py-2 border-b border-border bg-muted/20 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <List className="text-muted-foreground" size={14} />
          <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">Kernel Events Stream</span>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="flex flex-col">
          {sortedLogs.map((log) => (
            <div
              className="px-4 py-1.5 border-b border-border/30 hover:bg-muted/10 transition-colors flex items-start gap-4"
              key={log.id}
            >
              <div className="flex items-center gap-2 min-w-[80px] shrink-0 pt-0.5">
                <Clock className="text-muted-foreground/50" size={10} />
                <span className="text-[10px] text-muted-foreground">
                  {new Date(Number(log.timestamp)).toLocaleTimeString([], {
                    hour: "2-digit",
                    hour12: false,
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
              </div>
              <div className="shrink-0 pt-0.5">{getLogIcon(log.eventType)}</div>
              <div className="flex flex-col min-w-0 flex-1">
                <div className="flex items-center gap-2 mb-0.5">
                  <span
                    className={cn(
                      "text-[10px] font-bold px-1 rounded",
                      log.eventType.toLowerCase() === "error"
                        ? "bg-destructive/10 text-destructive"
                        : "bg-muted text-muted-foreground",
                    )}
                  >
                    {log.eventType.toUpperCase()}
                  </span>
                  <span className="text-[10px] text-primary/70">TASK:{log.taskId.slice(0, 8)}</span>
                </div>
                <span className="text-[11px] leading-relaxed break-words">{log.message}</span>
              </div>
            </div>
          ))}
          {sortedLogs.length === 0 && (
            <div className="py-20 text-center text-muted-foreground text-xs italic">Awaiting kernel events...</div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};
