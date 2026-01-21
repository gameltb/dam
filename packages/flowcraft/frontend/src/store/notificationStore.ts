import { create } from "zustand";

export enum NotificationType {
  ERROR = "error",
  INFO = "info",
  SUCCESS = "success",
}

export interface Notification {
  id: number;
  message: string;
  timestamp: Date;
  type: NotificationType;
}

interface NotificationState {
  addNotification: (notification: Omit<Notification, "id" | "timestamp">) => void;
  clearNotifications: () => void;
  notifications: Notification[];
}

export const useNotificationStore = create<NotificationState>((set) => ({
  addNotification: (notification) => {
    set((state) => ({
      notifications: [...state.notifications, { ...notification, id: Date.now(), timestamp: new Date() }],
    }));
  },
  clearNotifications: () => {
    set({ notifications: [] });
  },
  notifications: [],
}));
