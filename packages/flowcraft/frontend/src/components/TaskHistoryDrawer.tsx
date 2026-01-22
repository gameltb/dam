import { Activity, Cpu, Database, List, RotateCw, Shield, Square, Terminal, X } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTable } from "spacetimedb/react";

import { tables } from "@/generated/spacetime";
import { useFlowSocket } from "@/hooks/useFlowSocket";
import { cn } from "@/lib/utils";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { type TaskDefinition, TaskStatus } from "@/types";

import { SpacetimeTableBrowser } from "./debug/SpacetimeTableBrowser";
import { KernelLogViewer } from "./kernel-monitor/KernelLogViewer";
import { StatusBadge } from "./kernel-monitor/StatusIndicators";
import { TaskConsole } from "./kernel-monitor/TaskConsole";
import { TaskDetailsPanel } from "./kernel-monitor/TaskDetailsPanel";
import { TaskItem } from "./kernel-monitor/TaskItem";
import { WorkerMonitor } from "./kernel-monitor/WorkerMonitor";
import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";

export const TaskHistoryDrawer: React.FC = () => {
  const { isDrawerOpen, mutationLogs, selectedTaskId, setDrawerOpen, setSelectedTaskId, tasks } = useTaskStore();
  const { kernelExplorerHeight, setKernelExplorerHeight } = useUiStore();
  const { cancelTask, restartTask } = useFlowSocket();
  const [uptime, setUptime] = useState(0);
  const [stAssignments] = useTable(tables.clientTaskAssignments);
  const [workers] = useTable(tables.workers);

  const isResizing = useRef(false);

  const handleMouseMove = useCallback(
    (e: MouseEvent) => {
      if (!isResizing.current) return;
      const newHeight = window.innerHeight - e.clientY;
      if (newHeight > 200 && newHeight < window.innerHeight * 0.8) {
        setKernelExplorerHeight(newHeight);
      }
    },
    [setKernelExplorerHeight],
  );

  const startResizing = useCallback(
    (e: React.MouseEvent) => {
      e.preventDefault();
      isResizing.current = true;

      const stopResizing = () => {
        isResizing.current = false;
        document.removeEventListener("mousemove", handleMouseMove);
        document.removeEventListener("mouseup", stopResizing);
      };

      document.addEventListener("mousemove", handleMouseMove);
      document.addEventListener("mouseup", stopResizing);
    },
    [handleMouseMove],
  );

  useEffect(() => {
    const handleResize = () => {
      const maxHeight = window.innerHeight * 0.85;
      if (kernelExplorerHeight > maxHeight) {
        setKernelExplorerHeight(maxHeight);
      }
    };
    window.addEventListener("resize", handleResize);
    return () => {
      window.removeEventListener("resize", handleResize);
    };
  }, [kernelExplorerHeight, setKernelExplorerHeight]);

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
    () =>
      taskList.filter(
        (t) => t.status === TaskStatus.RUNNING || t.status === TaskStatus.PENDING || t.status === TaskStatus.CLAIMED,
      ),
    [taskList],
  );

  const selectedTaskIdentity = useMemo(() => {
    if (!selectedTaskId) return null;
    const assignment = stAssignments.find((a) => a.taskId === selectedTaskId);
    return assignment?.clientIdentity ?? "N/A";
  }, [stAssignments, selectedTaskId]);

  const selectedTask = selectedTaskId ? (tasks[selectedTaskId] as TaskDefinition | undefined) : null;

  if (!isDrawerOpen) {
    return (
      <div
        className="fixed bottom-0 right-5 z-[1000] flex items-center gap-2 px-4 py-2 bg-background border border-border border-b-0 rounded-t-lg cursor-pointer hover:bg-muted transition-colors shadow-2xl group"
        onClick={() => {
          setDrawerOpen(true);
        }}
      >
        <Activity
          className={cn(
            "transition-colors",
            activeInstances.length > 0
              ? "text-primary animate-pulse"
              : "text-muted-foreground group-hover:text-foreground",
          )}
          size={14}
        />
        <span className="text-[10px] font-black uppercase tracking-widest">Kernel Monitor</span>
        {activeInstances.length > 0 && (
          <Badge className="h-4 px-1 min-w-[16px] flex justify-center text-[9px] font-bold" variant="default">
            {activeInstances.length}
          </Badge>
        )}
      </div>
    );
  }

  return (
    <div
      className="fixed bottom-0 left-0 right-0 bg-background border-t border-border z-[6000] flex flex-col shadow-[0_-10px_40px_rgba(0,0,0,0.3)] animate-in slide-in-from-bottom duration-300"
      style={{ height: kernelExplorerHeight, maxHeight: "85vh" }}
    >
      {/* Resize Handle */}
      <div
        className="absolute top-0 left-0 right-0 h-1 cursor-ns-resize hover:bg-primary/50 transition-colors z-50"
        onMouseDown={startResizing}
      />

      {/* Header Bar */}
      <div className="flex items-center justify-between px-4 h-11 border-b border-border bg-muted/30">
        <div className="flex items-center gap-6">
          <div className="flex items-center gap-2">
            <Cpu className="text-primary" size={16} />
            <span className="text-xs font-black tracking-widest uppercase">Spacetime Kernel Explorer</span>
          </div>

          <div className="flex gap-4 items-center">
            <div className="flex items-center gap-1.5 px-2 py-0.5 rounded bg-muted/50 border border-border/50">
              <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
              <span className="text-[9px] font-mono font-bold uppercase tracking-tight">System Online</span>
            </div>
            <div className="flex gap-3 text-[10px] font-mono text-muted-foreground">
              <span>
                UPTIME: <span className="text-foreground">{uptime}s</span>
              </span>
              <span>
                TASKS: <span className="text-foreground">{taskList.length}</span>
              </span>
              <span>
                WORKERS: <span className="text-foreground">{workers.length}</span>
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <Button
            className="h-7 w-7 rounded-md hover:bg-destructive/10 hover:text-destructive transition-colors"
            onClick={() => {
              setDrawerOpen(false);
            }}
            size="icon"
            variant="ghost"
          >
            <X size={16} />
          </Button>
        </div>
      </div>

      <div className="flex flex-1 overflow-hidden min-w-0 max-w-full">
        <Tabs className="flex-1 flex flex-col h-full min-w-0" defaultValue="tasks">
          {/* Main Tab Switcher */}
          <div className="px-4 border-b border-border bg-muted/10 flex items-center justify-between">
            <TabsList className="h-10 bg-transparent gap-0">
              <TabsTrigger
                className="text-[10px] uppercase font-black tracking-wider data-[state=active]:bg-transparent data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-5 h-10 transition-all opacity-60 data-[state=active]:opacity-100"
                value="tasks"
              >
                <Terminal className="mr-2" size={12} /> Task Console
              </TabsTrigger>
              <TabsTrigger
                className="text-[10px] uppercase font-black tracking-wider data-[state=active]:bg-transparent data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-5 h-10 transition-all opacity-60 data-[state=active]:opacity-100"
                value="workers"
              >
                <Shield className="mr-2" size={12} /> Cluster Workers
              </TabsTrigger>
              <TabsTrigger
                className="text-[10px] uppercase font-black tracking-wider data-[state=active]:bg-transparent data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-5 h-10 transition-all opacity-60 data-[state=active]:opacity-100"
                value="logs"
              >
                <List className="mr-2" size={12} /> Kernel Events
              </TabsTrigger>
              <TabsTrigger
                className="text-[10px] uppercase font-black tracking-wider data-[state=active]:bg-transparent data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-5 h-10 transition-all opacity-60 data-[state=active]:opacity-100"
                value="debug"
              >
                <Database className="mr-2" size={12} /> DB Browser
              </TabsTrigger>
            </TabsList>
          </div>

          <TabsContent className="flex-1 m-0 overflow-hidden outline-none" value="tasks">
            <div className="flex h-full">
              {/* Task List Sidebar */}
              <div className="w-[300px] border-r border-border flex flex-col bg-muted/5">
                <Tabs className="flex flex-col h-full" defaultValue="all">
                  <div className="p-2 border-b border-border/50 flex items-center justify-between">
                    <TabsList className="h-7 p-0.5 bg-muted/50 border border-border">
                      <TabsTrigger className="text-[9px] font-bold px-3 py-0 h-6" value="all">
                        HISTORY
                      </TabsTrigger>
                      <TabsTrigger className="text-[9px] font-bold px-3 py-0 h-6" value="active">
                        ACTIVE
                      </TabsTrigger>
                    </TabsList>
                  </div>

                  <TabsContent className="flex-1 mt-0 overflow-hidden" value="all">
                    <ScrollArea className="h-full">
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

                  <TabsContent className="flex-1 mt-0 overflow-hidden" value="active">
                    <ScrollArea className="h-full">
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
                          <div className="p-12 text-center text-muted-foreground flex flex-col items-center gap-2">
                            <Activity className="opacity-10" size={32} />
                            <span className="text-[10px] italic">No active operations.</span>
                          </div>
                        )}
                      </div>
                    </ScrollArea>
                  </TabsContent>
                </Tabs>
              </div>

              {/* Task Details & Console */}
              <div className="flex-1 flex flex-col bg-background min-w-0">
                {selectedTask ? (
                  <div className="flex flex-col h-full overflow-hidden">
                    <div className="px-6 py-4 border-b border-border bg-muted/5 flex justify-between items-center">
                      <div className="flex items-center gap-4">
                        <div className="flex flex-col gap-0.5">
                          <div className="flex items-center gap-2">
                            <h3 className="text-sm font-black tracking-tight">{selectedTask.label}</h3>
                            <Badge
                              className="text-[9px] font-black uppercase h-4 bg-primary/10 text-primary border-none"
                              variant="outline"
                            >
                              {selectedTask.type}
                            </Badge>
                          </div>
                          <code className="text-[10px] text-muted-foreground/60 font-mono tracking-tighter">
                            ID: {selectedTask.taskId}
                          </code>
                        </div>

                        <div className="h-8 w-[1px] bg-border mx-2" />

                        <div className="flex flex-col gap-1">
                          <div className="flex items-center gap-3">
                            <StatusBadge status={selectedTask.status} />
                            {(selectedTask.status === TaskStatus.RUNNING ||
                              selectedTask.status === TaskStatus.CLAIMED) && (
                              <div className="flex items-center gap-2 w-40">
                                <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden border border-border/30">
                                  <div
                                    className="h-full bg-primary transition-all duration-700 ease-out shadow-[0_0_8px_rgba(var(--primary),0.5)]"
                                    style={{
                                      width: `${String(selectedTask.progress)}%`,
                                    }}
                                  />
                                </div>
                                <span className="text-[10px] font-black font-mono w-8">
                                  {Math.round(selectedTask.progress)}%
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>

                      <div className="flex gap-2">
                        {selectedTask.nodeId && (
                          <Button
                            className="h-8 gap-2 text-[10px] font-bold uppercase tracking-wider"
                            onClick={() => {
                              if (selectedTask.nodeId) restartTask(selectedTask.nodeId);
                            }}
                            size="sm"
                            variant="outline"
                          >
                            <RotateCw size={12} /> Restart
                          </Button>
                        )}
                        {(selectedTask.status === TaskStatus.RUNNING ||
                          selectedTask.status === TaskStatus.PENDING ||
                          selectedTask.status === TaskStatus.CLAIMED) && (
                          <Button
                            className="h-8 gap-2 text-[10px] font-bold uppercase tracking-wider"
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
                      <TaskConsole mutationLogs={mutationLogs} taskId={selectedTask.taskId} />
                      <TaskDetailsPanel identity={selectedTaskIdentity ?? "N/A"} task={selectedTask} />
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground p-12 bg-muted/5">
                    <div className="w-16 h-16 rounded-3xl bg-background border border-border flex items-center justify-center mb-6 shadow-sm rotate-3">
                      <Activity className="opacity-20 text-primary" size={32} />
                    </div>
                    <h3 className="text-sm font-black uppercase tracking-widest mb-2">Kernel Monitor Active</h3>
                    <p className="text-[10px] opacity-50 text-center max-w-[200px] leading-relaxed">
                      Select an operation from the sidebar to inspect execution events and state mutations.
                    </p>
                  </div>
                )}
              </div>
            </div>
          </TabsContent>

          <TabsContent className="flex-1 m-0 overflow-hidden outline-none" value="workers">
            <WorkerMonitor />
          </TabsContent>

          <TabsContent className="flex-1 m-0 overflow-hidden outline-none" value="logs">
            <KernelLogViewer />
          </TabsContent>

          <TabsContent className="flex-1 m-0 overflow-hidden outline-none min-w-0" value="debug">
            <SpacetimeTableBrowser />
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
};
