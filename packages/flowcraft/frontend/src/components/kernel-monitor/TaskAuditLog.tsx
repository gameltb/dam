import { ChevronRight, Terminal } from "lucide-react";
import React from "react";

import { type MutationLogEntry } from "@/types";

import { ScrollArea } from "../ui/scroll-area";

interface TaskAuditLogProps {
  logs: MutationLogEntry[];
}

export const TaskAuditLog: React.FC<TaskAuditLogProps> = ({ logs }) => {
  return (
    <div className="flex-1 flex flex-col border-r border-border min-w-0">
      <div className="px-4 py-2 border-b border-border bg-muted/20 flex items-center gap-2">
        <Terminal className="text-muted-foreground" size={12} />
        <span className="text-[10px] font-bold uppercase">Audit & Mutation Stream</span>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4 space-y-4">
          {logs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground text-xs italic">
              No mutations emitted by this instance.
            </div>
          ) : (
            logs.map((log) => (
              <div className="group" key={log.id}>
                <div className="flex items-center gap-2 mb-1">
                  <span className="text-[10px] font-mono text-orange-500/80">
                    [{new Date(log.timestamp).toLocaleTimeString()}]
                  </span>
                  <span className="text-xs font-medium group-hover:text-primary transition-colors cursor-default">
                    {log.description}
                  </span>
                </div>
                <div className="ml-4 pl-4 border-l-2 border-muted space-y-1">
                  {log.mutations.map((m, idx) => {
                    const type = Object.keys(m).find((k) => k !== "toJSON");
                    return (
                      <div className="text-[10px] font-mono text-muted-foreground flex items-center gap-2" key={idx}>
                        <ChevronRight size={8} /> {type}
                      </div>
                    );
                  })}
                </div>
              </div>
            ))
          )}
        </div>
      </ScrollArea>
    </div>
  );
};
