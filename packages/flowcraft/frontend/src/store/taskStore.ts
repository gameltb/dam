import { create } from "zustand";
import { TaskStatus } from "../types";

interface TaskState {
  status: TaskStatus;
  progress: number;
  message: string;
  result?: unknown;
}

interface TaskStoreState {
  tasks: Record<string, TaskState>; // Map taskId -> State

  registerTask: (taskId: string) => void;
  updateTask: (taskId: string, update: Partial<TaskState>) => void;
  removeTask: (taskId: string) => void;
  getTask: (taskId: string) => TaskState | undefined;
}

export const useTaskStore = create<TaskStoreState>((set, get) => ({
  tasks: {},

  registerTask: (taskId: string) => {
    set((state) => ({
      tasks: {
        ...state.tasks,
        [taskId]: {
          status: TaskStatus.TASK_PENDING,
          progress: 0,
          message: "Queued...",
        },
      },
    }));
  },

  updateTask: (taskId: string, update: Partial<TaskState>) => {
    set((state) => {
      const current = state.tasks[taskId];
      if (!current) return state; // Task might have been removed

      return {
        tasks: {
          ...state.tasks,
          [taskId]: { ...current, ...update },
        },
      };
    });
  },

  removeTask: (taskId: string) => {
    set((state) => {
      const { [taskId]: deletedTask, ...rest } = state.tasks;
      console.log("Removing task", deletedTask);
      return { tasks: rest };
    });
  },

  getTask: (taskId: string) => get().tasks[taskId],
}));
