import { type GraphMiddleware, MutationDirection } from "./types";
import { socketClient, SocketStatus } from "../../utils/SocketClient";
import { create } from "@bufbuild/protobuf";
import { MutationListSchema } from "../../generated/flowcraft/v1/service_pb";
import { MutationSource } from "../../generated/flowcraft/v1/base_pb";

/**
 * SyncMiddleware
 * 负责将 OUTGOING 的变更同步到服务器
 */
export const syncMiddleware: GraphMiddleware = (event, next) => {
  if (
    event.direction === MutationDirection.OUTGOING &&
    socketClient.getStatus() === SocketStatus.CONNECTED
  ) {
    const source = event.context.source ?? MutationSource.SOURCE_USER;

    void socketClient.send({
      payload: {
        case: "mutations",
        value: create(MutationListSchema, {
          mutations: event.mutations,
          sequenceNumber: 0n,
          source: source,
        }),
      },
    });
  }
  next(event);
};
