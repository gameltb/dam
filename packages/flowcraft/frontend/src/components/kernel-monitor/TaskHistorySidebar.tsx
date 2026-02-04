import { Activity } from "lucide-react";
import React from "react";

import type { TaskDefinition } from "@/types";

import { ScrollArea } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

import { TaskItem } from "./TaskItem";

interface TaskHistorySidebarProps {
  activeInstances: TaskDefinition[];
  onSelectTask: (taskId: string) => void;
  selectedTaskId: null | string;
  taskList: TaskDefinition[];
}

export const TaskHistorySidebar: React.FC<TaskHistorySidebarProps> = ({
  activeInstances,
  onSelectTask,
  selectedTaskId,
  taskList,
}) => {
  return (
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
                    onSelectTask(task.taskId);
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
                    onSelectTask(task.taskId);
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
  );
};
