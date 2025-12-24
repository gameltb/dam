import { create } from "zustand";

export interface Notification {
  id: number;
  message: string;
  type: "success" | "error" | "info";
  timestamp: Date;
}

interface NotificationState {
  notifications: Notification[];
  addNotification: (
    notification: Omit<Notification, "id" | "timestamp">,
  ) => void;
  clearNotifications: () => void;
}

export const useNotificationStore = create<NotificationState>((set) => ({
  notifications: [],
  addNotification: (notification) =>
    set((state) => ({
      notifications: [
        ...state.notifications,
        { ...notification, id: Date.now(), timestamp: new Date() },
      ],
    })),
  clearNotifications: () => set({ notifications: [] }),
}));
