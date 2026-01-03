import { create } from "zustand";
import {
  type TaskDefinition,
  type MutationLogEntry,
  MutationSource,
  TaskStatus,
} from "../types";

interface TaskState {
  tasks: Record<string, TaskDefinition>;
  mutationLogs: MutationLogEntry[];
  isDrawerOpen: boolean;
  selectedTaskId: string | null;

  // Actions
  registerTask: (
    task: Partial<TaskDefinition> & { taskId: string; label: string },
  ) => void;
  updateTask: (taskId: string, update: Partial<TaskDefinition>) => void;
  addMutationLog: (log: MutationLogEntry) => void;
  linkMutationToTask: (taskId: string, logId: string) => void;
  setDrawerOpen: (open: boolean) => void;
  setSelectedTaskId: (taskId: string | null) => void;
  resetStore: () => void;
}

export const useTaskStore = create<TaskState>((set) => ({
  tasks: {},
  mutationLogs: [],
  isDrawerOpen: false,
  selectedTaskId: null,

  registerTask: (task) => {
    set((state) => ({
      tasks: {
        ...state.tasks,
        [task.taskId]: {
          taskId: task.taskId,
          type: task.type ?? "unknown",
          label: task.label,
          status: task.status ?? TaskStatus.TASK_PENDING,
          progress: task.progress ?? 0,
          message: task.message ?? "Registered",
          source: task.source ?? MutationSource.SOURCE_SYSTEM,
          createdAt: Date.now(),
          updatedAt: Date.now(),
          mutationIds: [],
        },
      },
    }));
  },

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

  addMutationLog: (log) => {
    set((state) => ({
      mutationLogs: [log, ...state.mutationLogs],
    }));
  },

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

  setDrawerOpen: (open) => {
    set({ isDrawerOpen: open });
  },
  setSelectedTaskId: (taskId) => {
    set({ selectedTaskId: taskId });
  },
  resetStore: () => {
    set({ tasks: {}, mutationLogs: [], selectedTaskId: null });
  },
}));
