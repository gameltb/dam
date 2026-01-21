import { Activity, Bug, Cpu, RotateCw, Square, X } from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { cn } from "@/lib/utils";
import { useTaskStore } from "@/store/taskStore";
import { type TaskDefinition, TaskStatus } from "@/types";

import { SpacetimeTableBrowser } from "./debug/SpacetimeTableBrowser";
import { StatusBadge } from "./kernel-monitor/StatusIndicators";
import { TaskAuditLog } from "./kernel-monitor/TaskAuditLog";
import { TaskDetailsPanel } from "./kernel-monitor/TaskDetailsPanel";
import { TaskItem } from "./kernel-monitor/TaskItem";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

export const TaskHistoryDrawer: React.FC = () => {
  const { isDrawerOpen, mutationLogs, selectedTaskId, setDrawerOpen, setSelectedTaskId, tasks } = useTaskStore();
  const { cancelTask, restartTask } = useFlowSocket();
  const [uptime, setUptime] = useState(0);
  const [stAssignments] = useTable(tables.clientTaskAssignments);

  useEffect(() => {
    const start = Date.now();
    const interval = setInterval(() => {
      setUptime(Math.floor((Date.now() - start) / 1000));
    }, 1000);
    return () => {
      clearInterval(interval);
    };
  }, []);

  const taskList = useMemo(() => Object.values(tasks).sort((a, b) => b.createdAt - a.createdAt), [tasks]);

  const activeInstances = useMemo(
    () => taskList.filter((t) => t.status === TaskStatus.RUNNING || t.status === TaskStatus.PENDING),
    [taskList],
  );

  const selectedTaskIdentity = useMemo(() => {
    if (!selectedTaskId) return null;
    const assignment = stAssignments.find((a) => a.taskId === selectedTaskId);
    return assignment?.clientIdentity ?? "N/A";
  }, [stAssignments, selectedTaskId]);

  const selectedTask = selectedTaskId ? (tasks[selectedTaskId] as TaskDefinition | undefined) : null;

  const relatedLogs = mutationLogs.filter((log) => log.taskId === selectedTaskId);

  if (!isDrawerOpen) {
    return (
      <div
        className="fixed bottom-0 right-5 z-[1000] flex items-center gap-2 px-4 py-2 bg-background border border-border border-b-0 rounded-t-lg cursor-pointer hover:bg-muted transition-colors shadow-2xl"
        onClick={() => {
          setDrawerOpen(true);
        }}
      >
        <Activity className={cn(activeInstances.length > 0 && "text-primary animate-pulse")} size={14} />
        <span className="text-xs font-bold uppercase tracking-tighter">Kernel Monitor</span>
        {activeInstances.length > 0 && (
          <Badge className="h-4 px-1 min-w-[16px] flex justify-center text-[10px]" variant="default">
            {activeInstances.length}
          </Badge>
        )}
      </div>
    );
  }

  return (
    <div className="fixed bottom-0 left-0 right-0 h-[450px] bg-background border-t border-border z-[2000] flex flex-col shadow-2xl animate-in slide-in-from-bottom duration-300">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Cpu className="text-primary" size={16} />
            <span className="text-xs font-black tracking-widest uppercase">Spacetime Kernel Explorer</span>
          </div>
          <div className="h-4 w-[1px] bg-border" />
          <div className="flex gap-3 text-[10px] font-mono text-muted-foreground">
            <span>UPTIME: {uptime}s</span>
            <span>TASKS: {taskList.length}</span>
            <span>LOGS: {mutationLogs.length}</span>
          </div>
        </div>
        <Button
          onClick={() => {
            setDrawerOpen(false);
          }}
          size="icon-sm"
          variant="ghost"
        >
          <X size={16} />
        </Button>
      </div>

      <div className="flex flex-1 overflow-hidden">
        <Tabs className="flex-1 flex flex-col h-full" defaultValue="tasks">
          <div className="px-4 border-b border-border bg-muted/10">
            <TabsList className="h-10 bg-transparent gap-4">
              <TabsTrigger
                className="text-[10px] uppercase font-bold tracking-wider data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4"
                value="tasks"
              >
                <Activity className="mr-2" size={12} /> Tasks & Instances
              </TabsTrigger>
              <TabsTrigger
                className="text-[10px] uppercase font-bold tracking-wider data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4"
                value="debug"
              >
                <Bug className="mr-2" size={12} /> SpacetimeDB Browser
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent className="flex-1 m-0 overflow-hidden" value="tasks">
            <div className="flex h-full">
              <div className="w-[320px] border-r border-border flex flex-col bg-muted/5">
                <Tabs className="flex flex-col h-full" defaultValue="all">
                  <div className="p-2">
                    <TabsList className="grid w-full grid-cols-2">
                      <TabsTrigger className="text-[10px]" value="all">
                        ALL HISTORY
                      </TabsTrigger>
                      <TabsTrigger className="text-[10px]" value="active">
                        ACTIVE
                      </TabsTrigger>
                    </TabsList>
                  </div>

                  <TabsContent className="flex-1 mt-0" value="all">
                    <ScrollArea className="h-[310px]">
                      <div className="flex flex-col">
                        {taskList.map((task) => (
                          <TaskItem
                            isSelected={selectedTaskId === task.taskId}
                            key={task.taskId}
                            onClick={() => {
                              setSelectedTaskId(task.taskId);
                            }}
                            task={task}
                          />
                        ))}
                      </div>
                    </ScrollArea>
                  </TabsContent>

                  <TabsContent className="flex-1 mt-0" value="active">
                    <ScrollArea className="h-[310px]">
                      <div className="flex flex-col">
                        {activeInstances.map((task) => (
                          <TaskItem
                            isSelected={selectedTaskId === task.taskId}
                            key={task.taskId}
                            onClick={() => {
                              setSelectedTaskId(task.taskId);
                            }}
                            task={task}
                          />
                        ))}
                        {activeInstances.length === 0 && (
                          <div className="p-8 text-center text-muted-foreground text-xs italic">
                            No active instances.
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              </div>

              <div className="flex-1 flex flex-col bg-background">
                {selectedTask ? (
                  <div className="flex flex-col h-full overflow-hidden">
                    <div className="p-4 border-b border-border flex justify-between items-start">
                      <div>
                        <div className="flex items-center gap-2 mb-1">
                          <Badge className="text-[10px] font-mono uppercase" variant="outline">
                            {selectedTask.type}
                          </Badge>
                          <h3 className="text-sm font-bold tracking-tight">{selectedTask.label}</h3>
                        </div>
                        <code className="text-[10px] text-muted-foreground block mb-2">ID: {selectedTask.taskId}</code>

                        <div className="flex items-center gap-4">
                          <StatusBadge status={selectedTask.status} />
                          {selectedTask.status === TaskStatus.RUNNING && (
                            <div className="flex items-center gap-2 w-32">
                              <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary transition-all duration-500"
                                  style={{
                                    width: `${String(selectedTask.progress)}%`,
                                  }}
                                />
                              </div>
                              <span className="text-[10px] font-mono">{Math.round(selectedTask.progress)}%</span>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex gap-2">
                        {selectedTask.nodeId && (
                          <Button
                            className="h-8 gap-1.5 text-xs"
                            onClick={() => {
                              if (selectedTask.nodeId) restartTask(selectedTask.nodeId);
                            }}
                            size="sm"
                            variant="outline"
                          >
                            <RotateCw size={12} /> Restart
                          </Button>
                        )}
                        {(selectedTask.status === TaskStatus.RUNNING || selectedTask.status === TaskStatus.PENDING) && (
                          <Button
                            className="h-8 gap-1.5 text-xs"
                            onClick={() => {
                              cancelTask(selectedTask.taskId);
                            }}
                            size="sm"
                            variant="destructive"
                          >
                            <Square size={12} /> Terminate
                          </Button>
                        )}
                      </div>
                    </div>

                    <div className="flex-1 flex overflow-hidden">
                      <TaskAuditLog logs={relatedLogs} />
                      <TaskDetailsPanel identity={selectedTaskIdentity ?? "N/A"} task={selectedTask} />
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground p-12">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                      <Activity className="opacity-20" size={24} />
                    </div>
                    <p className="text-sm font-medium">Monitor active kernel operations</p>
                    <p className="text-xs opacity-50 mt-1">Select a task from history or browser STDB schemas</p>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent className="flex-1 m-0 overflow-hidden" value="debug">
            <SpacetimeTableBrowser />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
