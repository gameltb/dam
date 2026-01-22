import { DbConnection } from "@/generated/spacetime";

import { type PbClient, wrapReducers } from "../utils/pb-client";
import logger from "./utils/logger";

// ... [polyfill code remains] ...

let conn: DbConnection | null = null;
let pbClient: null | PbClient = null;
const connectListeners: ((c: PbClient) => void)[] = [];

export const initSpacetime = () => {
  if (conn) return pbClient!;

  logger.info("Connecting to SpacetimeDB at ws://127.0.0.1:3000");
  conn = DbConnection.builder()
    .withUri("ws://127.0.0.1:3000")
    .withModuleName("flowcraft")
    .withToken("")
    .onConnect((c) => {
      logger.info("Connected to SpacetimeDB");

      // Global subscriptions for server services
      void c.subscriptionBuilder().subscribe(["SELECT * FROM nodes", "SELECT * FROM tasks", "SELECT * FROM workers"]);

      pbClient = wrapReducers(c);
      connectListeners.forEach((l) => {
        l(pbClient!);
      });
    })
    .onDisconnect(() => {
      logger.info("Disconnected from SpacetimeDB");
    })
    .build();

  return pbClient;
};

export const onSpacetimeConnect = (cb: (c: PbClient) => void) => {
  if (pbClient) {
    cb(pbClient);
  } else {
    connectListeners.push(cb);
  }
};

export const getSpacetimeConn = () => pbClient;

export const createTaskConnection = async (taskId: string): Promise<PbClient> => {
  return new Promise((resolve) => {
    DbConnection.builder()
      .withUri("ws://127.0.0.1:3000")
      .withModuleName("flowcraft")
      .withToken("")
      .onConnect((c) => {
        logger.info(`Task client connected for taskId: ${taskId}`);
        const wrapped = wrapReducers(c);
        wrapped.reducers.assignCurrentTask({ taskId });
        resolve(wrapped);
      })
      .build();
  });
};
