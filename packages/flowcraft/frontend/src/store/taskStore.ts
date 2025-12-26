import { create } from "zustand";
import {
  TaskStatus,
  type TaskDefinition,
  type MutationLogEntry,
  MutationSource,
} from "../types";

interface TaskStoreState {
  tasks: Record<string, TaskDefinition>;
  mutationLogs: MutationLogEntry[];

  // Task Management
  registerTask: (task: Partial<TaskDefinition> & { taskId: string }) => void;
  updateTask: (taskId: string, update: Partial<TaskDefinition>) => void;

  // Mutation Logging
  logMutation: (entry: Omit<MutationLogEntry, "id" | "timestamp">) => void;

  // UI State
  isDrawerOpen: boolean;
  setDrawerOpen: (open: boolean) => void;
  selectedTaskId: string | null;
  setSelectedTaskId: (id: string | null) => void;
}

export const useTaskStore = create<TaskStoreState>((set) => ({
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
          type: task.type || "generic",
          label: task.label || "New Task",
          source: task.source || MutationSource.SYSTEM,
          status: TaskStatus.TASK_PENDING,
          progress: 0,
          message: "Initialized",
          createdAt: Date.now(),
          updatedAt: Date.now(),
          mutationIds: [],
          ...task,
        },
      },
    }));
  },

  updateTask: (taskId, update) => {
    set((state) => {
      const current = state.tasks[taskId];
      if (!current) return state;
      return {
        tasks: {
          ...state.tasks,
          [taskId]: {
            ...current,
            ...update,
            updatedAt: Date.now(),
          },
        },
      };
    });
  },

  logMutation: (entry) => {
    const logId = Math.random().toString(36).substring(7);
    const newEntry: MutationLogEntry = {
      ...entry,
      id: logId,
      timestamp: Date.now(),
    };

    set((state) => {
      const task = state.tasks[entry.taskId];
      const updatedTasks = { ...state.tasks };

      if (task) {
        updatedTasks[entry.taskId] = {
          ...task,
          mutationIds: [...task.mutationIds, logId],
          updatedAt: Date.now(),
        };
      }

      return {
        mutationLogs: [newEntry, ...state.mutationLogs].slice(0, 1000), // Keep last 1000
        tasks: updatedTasks,
      };
    });
  },

  setDrawerOpen: (isDrawerOpen) => set({ isDrawerOpen }),
  setSelectedTaskId: (selectedTaskId) => set({ selectedTaskId }),
}));
