import { Cpu, HardDrive, ShieldCheck } from "lucide-react";
import React from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";

import { Badge } from "../ui/badge";
import { ScrollArea } from "../ui/scroll-area";

export const WorkerMonitor: React.FC = () => {
  const [workers] = useTable(tables.workers);

  const getStatusColor = (lastHeartbeat: bigint) => {
    const diff = Date.now() - Number(lastHeartbeat);
    if (diff < 10000) return "bg-green-500";
    if (diff < 30000) return "bg-yellow-500";
    return "bg-red-500";
  };

  return (
    <div className="flex-1 flex flex-col h-full bg-background">
      <div className="px-4 py-3 border-b border-border bg-muted/20 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Cpu className="text-primary" size={14} />
          <span className="text-xs font-bold uppercase tracking-wider">Cluster Nodes ({workers.length})</span>
        </div>
      </div>
      <ScrollArea className="flex-1">
        <div className="p-4 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
          {workers.map((worker) => (
            <div
              className="p-3 rounded-lg border border-border bg-muted/5 hover:border-primary/30 transition-colors"
              key={worker.workerId}
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex flex-col">
                  <span className="text-[10px] font-mono text-muted-foreground mb-1">
                    ID: {worker.workerId.slice(0, 18)}...
                  </span>
                  <div className="flex items-center gap-2">
                    <ShieldCheck className="text-primary" size={12} />
                    <span className="text-xs font-bold uppercase tracking-tight">
                      {worker.lang.tag.replace("WORKER_LANG_", "")} Worker
                    </span>
                  </div>
                </div>
                <div
                  className={`w-2 h-2 rounded-full ${getStatusColor(worker.lastHeartbeat)} shadow-[0_0_5px_currentColor]`}
                />
              </div>

              <div className="space-y-3">
                <div>
                  <div className="text-[9px] font-bold text-muted-foreground uppercase mb-1 flex items-center gap-1">
                    <HardDrive size={10} /> Capabilities
                  </div>
                  <div className="flex flex-wrap gap-1">
                    {worker.capabilities.split(",").map((cap) => (
                      <Badge className="text-[9px] px-1 py-0 h-4 bg-primary/10 text-primary border-none" key={cap}>
                        {cap.trim()}
                      </Badge>
                    ))}
                  </div>
                </div>

                <div className="flex justify-between items-center pt-2 border-t border-border/50 text-[9px] text-muted-foreground">
                  <span>HEARTBEAT</span>
                  <span>{new Date(Number(worker.lastHeartbeat)).toLocaleTimeString()}</span>
                </div>
              </div>
            </div>
          ))}

          {workers.length === 0 && (
            <div className="col-span-full py-20 flex flex-col items-center justify-center text-muted-foreground">
              <Cpu className="opacity-10 mb-2" size={48} />
              <p className="text-xs italic">No active workers registered in the cluster.</p>
            </div>
          )}
        </div>
      </ScrollArea>
    </div>
  );
};
