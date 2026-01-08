import { create } from "@bufbuild/protobuf";
import { v4 as uuidv4 } from "uuid";

import { type TaskUpdate } from "@/generated/flowcraft/v1/core/node_pb";
import {
  type FlowMessage,
  FlowMessageSchema,
  type GraphSnapshot,
  GraphSnapshotSchema,
  type MutationList,
  type NodeEvent,
} from "@/generated/flowcraft/v1/core/service_pb";
import {
  type NodeSignal,
  type WidgetSignal,
} from "@/generated/flowcraft/v1/core/signals_pb";
import { toProtoEdge, toProtoNode } from "@/utils/protoAdapter";
import { eventBus, serverGraph, serverVersion } from "./PersistenceService";

export async function* watchGraph(_req: unknown, ctx: { signal: AbortSignal }) {
  console.log("[GraphWatcher] Client connected");

  try {
    const snapshot = create(FlowMessageSchema, {
      messageId: uuidv4(),
      payload: {
        case: "snapshot",
        value: create(GraphSnapshotSchema, {
          edges: serverGraph.edges.map(toProtoEdge),
          nodes: serverGraph.nodes.map(toProtoNode),
          version: BigInt(serverVersion),
        }),
      },
      timestamp: BigInt(Date.now()),
    });
    yield snapshot;
  } catch (err) {
    console.error("[GraphWatcher] Error generating snapshot:", err);
  }

  const queue: FlowMessage[] = [];
  let waitingPromise: null | { promise: Promise<void>; resolve: () => void } =
    null;

  const pushToQueue = <K extends NonNullable<FlowMessage["payload"]>["case"]>(
    type: K,
    payload: Extract<NonNullable<FlowMessage["payload"]>, { case: K }>["value"],
  ) => {
    console.log(
      `[GraphWatcher] Pushing message to client queue: ${String(type)}`,
    );
    queue.push(
      create(FlowMessageSchema, {
        messageId: uuidv4(),
        payload: { case: type, value: payload } as NonNullable<
          FlowMessage["payload"]
        >,
        timestamp: BigInt(Date.now()),
      }),
    );
    if (waitingPromise) {
      waitingPromise.resolve();
      waitingPromise = null;
    }
  };

  const onMutations = (m: MutationList) => {
    pushToQueue("mutations", m);
  };
  const onTaskUpdate = (t: TaskUpdate) => {
    pushToQueue("taskUpdate", t);
  };
  const onNodeEvent = (e: NodeEvent) => {
    pushToQueue("nodeEvent", e);
  };
  const onNodeSignal = (s: NodeSignal) => {
    pushToQueue("nodeSignal", s);
  };
  const onWidgetSignal = (s: WidgetSignal) => {
    pushToQueue("widgetSignal", s);
  };
  const onSnapshot = (s: GraphSnapshot) => {
    pushToQueue("snapshot", s);
  };

  eventBus.on("mutations", onMutations);
  eventBus.on("taskUpdate", onTaskUpdate);
  eventBus.on("nodeEvent", onNodeEvent);
  eventBus.on("nodeSignal", onNodeSignal);
  eventBus.on("widgetSignal", onWidgetSignal);
  eventBus.on("snapshot", onSnapshot);

  try {
    while (!ctx.signal.aborted) {
      while (queue.length > 0) {
        const msg = queue.shift();
        if (msg) yield msg;
      }
      if (!waitingPromise) {
        let resolve: () => void = () => {
          /* noop */
        };
        const promise = new Promise<void>((r) => {
          resolve = r;
        });
        waitingPromise = { promise, resolve };
      }
      await Promise.race([
        waitingPromise.promise,
        new Promise((r) => setTimeout(r, 10000)),
      ]);
    }
  } finally {
    eventBus.off("mutations", onMutations);
    eventBus.off("taskUpdate", onTaskUpdate);
    eventBus.off("nodeEvent", onNodeEvent);
    eventBus.off("nodeSignal", onNodeSignal);
    eventBus.off("widgetSignal", onWidgetSignal);
    eventBus.off("snapshot", onSnapshot);
  }
}
