import {
  Activity,
  Bug,
  ChevronRight,
  Cpu,
  History,
  RotateCw,
  Settings,
  Square,
  Terminal,
  User,
  X,
} from "lucide-react";
import React, { useEffect, useMemo, useState } from "react";
import { useTable } from "spacetimedb/react";
import { tables } from "@/generated/spacetime";

import { useFlowSocket } from "@/hooks/useFlowSocket";
import { cn } from "@/lib/utils";
import { useTaskStore } from "@/store/taskStore";
import { MutationSource, type TaskDefinition, TaskStatus } from "@/types";

import { Badge } from "./ui/badge";
import { Button } from "./ui/button";
import { ScrollArea } from "./ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "./ui/tabs";
import { SpacetimeTableBrowser } from "./debug/SpacetimeTableBrowser";

export const TaskHistoryDrawer: React.FC = () => {
  const {
    isDrawerOpen,
    mutationLogs,
    selectedTaskId,
    setDrawerOpen,
    setSelectedTaskId,
    tasks,
  } = useTaskStore();
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

  const taskList = useMemo(
    () => Object.values(tasks).sort((a, b) => b.createdAt - a.createdAt),
    [tasks],
  );

  const activeInstances = useMemo(
    () =>
      taskList.filter(
        (t) =>
          t.status === TaskStatus.TASK_PROCESSING ||
          t.status === TaskStatus.TASK_PENDING,
      ),
    [taskList],
  );

  const selectedTaskIdentity = useMemo(() => {
    if (!selectedTaskId) return null;
    const assignment = stAssignments.find((a) => a.taskId === selectedTaskId);
    return assignment?.clientIdentity ?? "N/A";
  }, [stAssignments, selectedTaskId]);

  if (!isDrawerOpen) {
    return (
      <div
        className="fixed bottom-0 right-5 z-[1000] flex items-center gap-2 px-4 py-2 bg-background border border-border border-b-0 rounded-t-lg cursor-pointer hover:bg-muted transition-colors shadow-2xl"
        onClick={() => {
          setDrawerOpen(true);
        }}
      >
        <Activity
          className={cn(
            activeInstances.length > 0 && "text-primary animate-pulse",
          )}
          size={14}
        />
        <span className="text-xs font-bold uppercase tracking-tighter">
          Kernel Monitor
        </span>
        {activeInstances.length > 0 && (
          <Badge
            className="h-4 px-1 min-w-[16px] flex justify-center text-[10px]"
            variant="default"
          >
            {activeInstances.length}
          </Badge>
        )}
      </div>
    );
  }

  const selectedTask = selectedTaskId
    ? (tasks[selectedTaskId] as TaskDefinition | undefined)
    : null;

  const relatedLogs = mutationLogs.filter(
    (log) => log.taskId === selectedTaskId,
  );

  return (
    <div className="fixed bottom-0 left-0 right-0 h-[450px] bg-background border-t border-border z-[2000] flex flex-col shadow-2xl animate-in slide-in-from-bottom duration-300">
      <div className="flex items-center justify-between px-4 py-2 border-b border-border bg-muted/30">
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-2">
            <Cpu className="text-primary" size={16} />
            <span className="text-xs font-black tracking-widest uppercase">
              Spacetime Kernel Explorer
            </span>
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
              <TabsTrigger className="text-[10px] uppercase font-bold tracking-wider data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4" value="tasks">
                <Activity size={12} className="mr-2" /> Tasks & Instances
              </TabsTrigger>
              <TabsTrigger className="text-[10px] uppercase font-bold tracking-wider data-[state=active]:bg-primary/10 data-[state=active]:text-primary rounded-none border-b-2 border-transparent data-[state=active]:border-primary px-4" value="debug">
                <Bug size={12} className="mr-2" /> SpacetimeDB Browser
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
                          <Badge
                            className="text-[10px] font-mono uppercase"
                            variant="outline"
                          >
                            {selectedTask.type}
                          </Badge>
                          <h3 className="text-sm font-bold tracking-tight">
                            {selectedTask.label}
                          </h3>
                        </div>
                        <code className="text-[10px] text-muted-foreground block mb-2">
                          ID: {selectedTask.taskId}
                        </code>

                        <div className="flex items-center gap-4">
                          <StatusBadge status={selectedTask.status} />
                          {selectedTask.status === TaskStatus.TASK_PROCESSING && (
                            <div className="flex items-center gap-2 w-32">
                              <div className="flex-1 h-1 bg-muted rounded-full overflow-hidden">
                                <div
                                  className="h-full bg-primary transition-all duration-500"
                                  style={{
                                    width: `${String(selectedTask.progress)}%`,
                                  }}
                                />
                              </div>
                              <span className="text-[10px] font-mono">
                                {Math.round(selectedTask.progress)}%
                              </span>
                            </div>
                          )}
                        </div>
                      </div>

                      <div className="flex gap-2">
                        {selectedTask.nodeId && (
                          <Button
                            className="h-8 gap-1.5 text-xs"
                            onClick={() => {
                              if (selectedTask.nodeId)
                                restartTask(selectedTask.nodeId);
                            }}
                            size="sm"
                            variant="outline"
                          >
                            <RotateCw size={12} /> Restart
                          </Button>
                        )}
                        {(selectedTask.status === TaskStatus.TASK_PROCESSING ||
                          selectedTask.status === TaskStatus.TASK_PENDING) && (
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
                      <div className="flex-1 flex flex-col border-r border-border min-w-0">
                        <div className="px-4 py-2 border-b border-border bg-muted/20 flex items-center gap-2">
                          <Terminal className="text-muted-foreground" size={12} />
                          <span className="text-[10px] font-bold uppercase">
                            Audit & Mutation Stream
                          </span>
                        </div>
                        <ScrollArea className="flex-1">
                          <div className="p-4 space-y-4">
                            {relatedLogs.length === 0 ? (
                              <div className="text-center py-12 text-muted-foreground text-xs italic">
                                No mutations emitted by this instance.
                              </div>
                            ) : (
                              relatedLogs.map((log) => (
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
                                      const type = Object.keys(m).find(
                                        (k) => k !== "toJSON",
                                      );
                                      return (
                                        <div
                                          className="text-[10px] font-mono text-muted-foreground flex items-center gap-2"
                                          key={idx}
                                        >
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

                      <div className="w-64 bg-muted/5 p-4 flex flex-col gap-4 overflow-y-auto">
                        <section>
                          <h4 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">
                            Context Identity
                          </h4>
                          <div className="p-2 rounded bg-muted/20 border border-border border-dashed">
                            <code className="text-[9px] break-all block text-primary">
                              {selectedTaskIdentity}
                            </code>
                          </div>
                        </section>

                        <section>
                          <h4 className="text-[10px] font-bold text-muted-foreground uppercase mb-2">
                            Metadata
                          </h4>
                          <div className="space-y-2">
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Source</span>
                              <span className="font-mono">
                                {selectedTask.source === MutationSource.SOURCE_USER
                                  ? "CLIENT_USER"
                                  : "WORKER_TASK"}
                              </span>
                            </div>
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">Node ID</span>
                              <span className="font-mono truncate ml-4 text-primary">
                                {selectedTask.nodeId ?? "GLOBAL"}
                              </span>
                            </div>
                            <div className="flex justify-between text-[10px]">
                              <span className="text-muted-foreground">
                                Registered
                              </span>
                              <span className="font-mono">
                                {new Date(
                                  selectedTask.createdAt,
                                ).toLocaleTimeString()}
                              </span>
                            </div>
                          </div>
                        </section>

                        <div className="mt-auto p-3 rounded border border-border bg-background flex flex-col gap-2">
                          <div className="flex items-center gap-2 text-[10px] font-bold">
                            <History size={12} />
                            <span>STDB_AUDIT_READY</span>
                          </div>
                          <p className="text-[9px] text-muted-foreground leading-relaxed italic">
                            This task is being audited via the implicit client-task assignment protocol.
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground p-12">
                    <div className="w-12 h-12 rounded-full bg-muted flex items-center justify-center mb-4">
                      <Activity className="opacity-20" size={24} />
                    </div>
                    <p className="text-sm font-medium">
                      Monitor active kernel operations
                    </p>
                    <p className="text-xs opacity-50 mt-1">
                      Select a task from history or browser STDB schemas
                    </p>
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

const TaskItem: React.FC<{
  isSelected: boolean;
  onClick: () => void;
  task: TaskDefinition;
}> = ({ isSelected, onClick, task }) => {
  return (
    <div
      className={cn(
        "flex flex-col p-3 border-b border-border/50 cursor-pointer transition-all border-l-2",
        isSelected
          ? "bg-accent border-l-primary"
          : "hover:bg-muted/50 border-l-transparent",
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
          <span className="text-[10px] font-mono text-muted-foreground truncate">
            {task.taskId.split("-")[0]}
          </span>
        </div>
        <span className="text-[9px] font-mono opacity-40">
          {new Date(task.createdAt).toLocaleTimeString()}
        </span>
      </div>
      <div className="text-xs font-semibold truncate leading-none mb-2">
        {task.label}
      </div>
      <div className="flex items-center gap-2">
        <StatusDot status={task.status} />
        <span className="text-[10px] text-muted-foreground capitalize">
          {TaskStatus[task.status]
            ? TaskStatus[task.status].replace("TASK_", "").toLowerCase()
            : "unknown"}
        </span>
      </div>
    </div>
  );
};

const StatusBadge: React.FC<{ status: TaskStatus }> = ({ status }) => {
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
    <span
      className={cn(
        "text-[10px] px-2 py-0.5 rounded border font-bold uppercase tracking-wider",
        color,
      )}
    >
      {label}
    </span>
  );
};

const StatusDot: React.FC<{ status: TaskStatus }> = ({ status }) => {
  const colors: Record<number, string> = {
    [TaskStatus.TASK_CANCELLED]: "bg-muted-foreground",
    [TaskStatus.TASK_COMPLETED]: "bg-green-500",
    [TaskStatus.TASK_FAILED]: "bg-destructive",
    [TaskStatus.TASK_PENDING]: "bg-yellow-500",
    [TaskStatus.TASK_PROCESSING]: "bg-primary animate-pulse",
    [TaskStatus.TASK_RESTARTING]: "bg-blue-500 animate-bounce",
  };

  return (
    <div
      className={cn("w-1.5 h-1.5 rounded-full", colors[status] ?? "bg-muted")}
    />
  );
};
