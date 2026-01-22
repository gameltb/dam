import { Activity, ChevronRight, Terminal } from "lucide-react";
import React, { useMemo } from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";
import { cn } from "@/lib/utils";
import { type MutationLogEntry } from "@/types";

import { ScrollArea } from "../ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "../ui/tabs";

interface TaskConsoleProps {
  mutationLogs: MutationLogEntry[];
  taskId: string;
}

export const TaskConsole: React.FC<TaskConsoleProps> = ({ mutationLogs, taskId }) => {
  const [stAuditLogs] = useTable(tables.taskAuditLog);

  const taskAuditLogs = useMemo(() => {
    return stAuditLogs.filter((log) => log.taskId === taskId).sort((a, b) => Number(a.timestamp - b.timestamp));
  }, [stAuditLogs, taskId]);

  const relatedMutations = useMemo(() => {
    return mutationLogs.filter((log) => log.taskId === taskId);
  }, [mutationLogs, taskId]);

  return (
    <div className="flex-1 flex flex-col h-full bg-background overflow-hidden border-r border-border">
      <Tabs className="flex-1 flex flex-col" defaultValue="events">
        <div className="px-4 py-1 border-b border-border bg-muted/20 flex items-center justify-between">
          <TabsList className="h-8 bg-transparent">
            <TabsTrigger
              className="text-[10px] uppercase font-bold px-3 py-1 data-[state=active]:bg-background"
              value="events"
            >
              <Terminal className="mr-1.5" size={12} /> Execution Events
            </TabsTrigger>
            <TabsTrigger
              className="text-[10px] uppercase font-bold px-3 py-1 data-[state=active]:bg-background"
              value="mutations"
            >
              <Activity className="mr-1.5" size={12} /> Mutation Stream
            </TabsTrigger>
          </TabsList>
        </div>

        <TabsContent className="flex-1 m-0 overflow-hidden" value="events">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-2">
              {taskAuditLogs.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground text-[10px] italic">
                  No execution events recorded for this task.
                </div>
              ) : (
                taskAuditLogs.map((log) => (
                  <div className="flex gap-3 text-[11px] font-mono leading-relaxed" key={log.id}>
                    <span className="text-muted-foreground shrink-0">
                      [{new Date(Number(log.timestamp)).toLocaleTimeString([], { hour12: false })}]
                    </span>
                    <span
                      className={cn(
                        "font-bold shrink-0",
                        log.eventType.toLowerCase() === "error" ? "text-destructive" : "text-primary/80",
                      )}
                    >
                      {log.eventType.toUpperCase()}
                    </span>
                    <span className="break-words">{log.message}</span>
                  </div>
                ))
              )}
            </div>
          </ScrollArea>
        </TabsContent>

        <TabsContent className="flex-1 m-0 overflow-hidden" value="mutations">
          <ScrollArea className="h-full">
            <div className="p-4 space-y-4">
              {relatedMutations.length === 0 ? (
                <div className="text-center py-12 text-muted-foreground text-[10px] italic">
                  No state mutations emitted by this task.
                </div>
              ) : (
                relatedMutations.map((log) => {
                  let parsedMutations: any[] = [];
                  try {
                    parsedMutations = JSON.parse(log.mutationsJson);
                  } catch (e) {
                    console.error("Failed to parse mutation log", e);
                  }

                  return (
                    <div className="group border-l-2 border-muted pl-4 py-1" key={log.id}>
                      <div className="flex items-center gap-2 mb-1">
                        <span className="text-[10px] font-mono text-orange-500/80">
                          [{new Date(log.timestamp).toLocaleTimeString()}]
                        </span>
                        <span className="text-xs font-bold text-primary/90">{log.description}</span>
                      </div>
                      <div className="space-y-1">
                        {parsedMutations.map((m, idx) => {
                          const fullName = m.$typeName || "Unknown Type";
                          const shortName = fullName.split(".").pop() || fullName;
                          return (
                            <div
                              className="text-[10px] font-mono text-muted-foreground flex items-center gap-2"
                              key={idx}
                            >
                              <ChevronRight className="opacity-50" size={10} />
                              <span className="bg-muted/50 px-1 rounded text-[9px]">{shortName}</span>
                              <span className="truncate opacity-50">{m.nodeId || m.targetId || m.id || ""}</span>
                            </div>
                          );
                        })}
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </ScrollArea>
        </TabsContent>
      </Tabs>
    </div>
  );
};
