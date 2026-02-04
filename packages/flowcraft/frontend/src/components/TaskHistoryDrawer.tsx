import { Activity, Database, List, Shield, Terminal } from "lucide-react";
import React, { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useTable } from "spacetimedb/react";
import { useShallow } from "zustand/react/shallow";

import { SpacetimeTableBrowser } from "@/components/kernel-monitor/debug/SpacetimeTableBrowser";
import { KernelLogViewer } from "@/components/kernel-monitor/KernelLogViewer";
import { TaskConsole } from "@/components/kernel-monitor/TaskConsole";
import { TaskDetailsPanel } from "@/components/kernel-monitor/TaskDetailsPanel";
import { WorkerMonitor } from "@/components/kernel-monitor/WorkerMonitor";
import { Badge } from "@/components/ui/badge";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { tables } from "@/generated/spacetime";
import { useFlowSocket } from "@/hooks/integration/useFlowSocket";
import { cn } from "@/lib/utils";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { type TaskDefinition, TaskStatus } from "@/types";

import { KernelExplorerHeader } from "./kernel-monitor/KernelExplorerHeader";
import { TaskActions } from "./kernel-monitor/TaskActions";
import { TaskHistorySidebar } from "./kernel-monitor/TaskHistorySidebar";

export const TaskHistoryDrawer: React.FC = () => {
  const { isDrawerOpen, mutationLogs, selectedTaskId, setDrawerOpen, setSelectedTaskId, tasks } = useTaskStore(
    useShallow((s) => ({
      isDrawerOpen: s.isDrawerOpen,
      mutationLogs: s.mutationLogs,
      selectedTaskId: s.selectedTaskId,
      setDrawerOpen: s.setDrawerOpen,
      setSelectedTaskId: s.setSelectedTaskId,
      tasks: s.tasks,
    })),
  );

  const { kernelExplorerHeight, setKernelExplorerHeight } = useUiStore(
    useShallow((s) => ({
      kernelExplorerHeight: s.kernelExplorerHeight,
      setKernelExplorerHeight: s.setKernelExplorerHeight,
    })),
  );
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
      <KernelExplorerHeader
        onClose={() => {
          setDrawerOpen(false);
        }}
        onMouseDown={startResizing}
        taskListCount={taskList.length}
        uptime={uptime}
        workersCount={workers.length}
      />

      <div className="flex flex-1 overflow-hidden min-w-0 max-w-full">
        <Tabs className="flex-1 flex flex-col h-full min-w-0" defaultValue="tasks">
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
              <TaskHistorySidebar
                activeInstances={activeInstances}
                onSelectTask={setSelectedTaskId}
                selectedTaskId={selectedTaskId}
                taskList={taskList}
              />

              <div className="flex-1 flex flex-col bg-background min-w-0">
                {selectedTask ? (
                  <div className="flex flex-col h-full overflow-hidden">
                    <TaskActions onCancel={cancelTask} onRestart={restartTask} task={selectedTask} />

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
