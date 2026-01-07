import { create } from "zustand";

export interface Notification {
  id: number;
  message: string;
  timestamp: Date;
  type: "error" | "info" | "success";
}

interface NotificationState {
  addNotification: (
    notification: Omit<Notification, "id" | "timestamp">,
  ) => void;
  clearNotifications: () => void;
  notifications: Notification[];
}

export const useNotificationStore = create<NotificationState>((set) => ({
  addNotification: (notification) => {
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: Date.now(), timestamp: new Date() },
      ],
    }));
  },
  clearNotifications: () => {
    set({ notifications: [] });
  },
  notifications: [],
}));
