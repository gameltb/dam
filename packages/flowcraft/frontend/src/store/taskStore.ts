import { create } from "zustand";

import { type MutationLogEntry, MutationSource, type TaskDefinition, TaskStatus, TaskType } from "@/types";

interface TaskState {
  addMutationLog: (log: MutationLogEntry) => void;
  isDrawerOpen: boolean;
  linkMutationToTask: (taskId: string, logId: string) => void;
  mutationLogs: MutationLogEntry[];

  // Actions
  registerTask: (task: Partial<TaskDefinition> & { label: string; taskId: string }) => void;
  resetStore: () => void;
  selectedTaskId: null | string;
  setDrawerOpen: (open: boolean) => void;
  setSelectedTaskId: (taskId: null | string) => void;
  tasks: Record<string, TaskDefinition>;
  updateTask: (taskId: string, update: Partial<TaskDefinition>) => void;
}

export const useTaskStore = create<TaskState>((set) => ({
  addMutationLog: (log) => {
    set((state) => ({
      mutationLogs: [log, ...state.mutationLogs],
    }));
  },
  isDrawerOpen: false,
  linkMutationToTask: (taskId, logId) => {
    set((state) => {
      const task = state.tasks[taskId];
      if (!task) return state;
      return {
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...task,
            mutationIds: [...task.mutationIds, logId],
          },
        },
      };
    });
  },
  mutationLogs: [],

  registerTask: (task) => {
    set((state) => ({
      tasks: {
        ...state.tasks,
        [task.taskId]: {
          createdAt: Date.now(),
          label: task.label,
          message: task.message ?? "Registered",
          mutationIds: [],
          nodeId: task.nodeId,
          progress: task.progress ?? 0,
          source: task.source ?? MutationSource.SOURCE_SYSTEM,
          status: task.status ?? TaskStatus.PENDING,
          taskId: task.taskId,
          type: task.type ?? TaskType.UNKNOWN,
          updatedAt: Date.now(),
        },
      },
    }));
  },

  resetStore: () => {
    set({ mutationLogs: [], selectedTaskId: null, tasks: {} });
  },

  selectedTaskId: null,

  setDrawerOpen: (open) => {
    set({ isDrawerOpen: open });
  },

  setSelectedTaskId: (taskId) => {
    set({ selectedTaskId: taskId });
  },
  tasks: {},
  updateTask: (taskId, update) => {
    set((state) => {
      const existing = state.tasks[taskId];
      if (!existing) return state;
      return {
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...existing,
            ...update,
            updatedAt: Date.now(),
          },
        },
      };
    });
  },
}));
