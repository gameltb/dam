import { DbConnection } from "@/generated/spacetime";

import logger from "./utils/logger";

// Polyfill WebSocket if needed (Node.js 22 has it global, but safer to check)
if (!globalThis.WebSocket) {
  // @ts-ignore
  globalThis.WebSocket = require("ws");
}

let conn: DbConnection | null = null;
const connectListeners: ((c: DbConnection) => void)[] = [];

export const initSpacetime = () => {
  if (conn) return conn;

  logger.info("Connecting to SpacetimeDB at ws://127.0.0.1:3000");
  conn = DbConnection.builder()
    .withUri("ws://127.0.0.1:3000")
    .withModuleName("flowcraft")
    .withToken("")
    .onConnect((c) => {
      logger.info("Connected to SpacetimeDB");
      connectListeners.forEach((l) => {
        l(c);
      });
    })
    .onDisconnect(() => {
      logger.info("Disconnected from SpacetimeDB");
    })
    .build();

  return conn;
};

export const onSpacetimeConnect = (cb: (c: DbConnection) => void) => {
  if (conn && conn.isActive) {
    cb(conn);
  } else {
    connectListeners.push(cb);
  }
};

export const getSpacetimeConn = () => conn;

export const createTaskConnection = async (taskId: string): Promise<DbConnection> => {
  return new Promise((resolve) => {
    DbConnection.builder()
      .withUri("ws://127.0.0.1:3000")
      .withModuleName("flowcraft")
      .withToken("") // Isolation: Each task gets its own identity by not providing a persistent token
      .onConnect((c) => {
        logger.info(`Task client connected for taskId: ${taskId}`);
        c.reducers.assignCurrentTask({ taskId });
        resolve(c);
      })
      .build();
  });
};
