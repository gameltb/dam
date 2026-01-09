import { create } from "@bufbuild/protobuf";

import { MutationSource } from "@/generated/flowcraft/v1/core/base_pb";
import { MutationListSchema } from "@/generated/flowcraft/v1/core/service_pb";
import { socketClient, SocketStatus } from "@/utils/SocketClient";

import {
  type GraphMiddleware,
  type GraphMutationEvent,
  MutationDirection,
} from "./types";

/**
 * SyncMiddleware
 * 负责将 OUTGOING 的变更同步到服务器
 */
export const syncMiddleware: GraphMiddleware = (
  event: GraphMutationEvent,
  next,
) => {
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
