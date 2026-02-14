/**
 * Log Management Utility
 *
 * Provides formatted, categorized logging for development.
 * Automatically disabled in production.
 */

type SyncDirection = "IN" | "OUT";

const COLORS = {
  ERROR: "#ef4444", // Red
  IN: "#3b82f6", // Blue
  OUT: "#10b981", // Green
};

export const log = {
  debug: (context: string, message: string, ...args: unknown[]) => {
    if (!import.meta.env.DEV) return;
    console.debug(
      `%c[DEBUG] %c[${context}] %c${message}`,
      "color: #8b5cf6; font-weight: bold;",
      "color: #6366f1; font-weight: bold;",
      "color: inherit;",
      ...args,
    );
  },

  error: (context: string, error: unknown) => {
    console.error(`%c[ERROR] ${context}`, `color: ${COLORS.ERROR}; font-weight: bold;`, error);
  },

  /**
   * Logs synchronization events with directional highlighting.
   */
  sync: (direction: SyncDirection, message: string, detail?: unknown) => {
    if (!import.meta.env.DEV) return;

    const color = COLORS[direction];
    const label = `[SYNC-${direction}]`;
    const timestamp = new Date().toLocaleTimeString();

    if (detail) {
      console.groupCollapsed(
        `%c${label} %c${timestamp} %c${message}`,
        `color: white; background: ${color}; padding: 2px 4px; border-radius: 2px; font-weight: bold;`,
        "color: #94a3b8; font-family: monospace; font-size: 10px;",
        "color: inherit; font-weight: bold;",
      );
      console.log("Detail:", detail);
      console.groupEnd();
    } else {
      console.log(
        `%c${label} %c${timestamp} %c${message}`,
        `color: white; background: ${color}; padding: 2px 4px; border-radius: 2px; font-weight: bold;`,
        "color: #94a3b8; font-family: monospace; font-size: 10px;",
        "color: inherit; font-weight: bold;",
      );
    }
  },

  warn: (context: string, ...args: unknown[]) => {
    console.warn(`%c[WARN] ${context}`, `color: ${COLORS.ERROR}; font-weight: bold;`, ...args);
  },
};
