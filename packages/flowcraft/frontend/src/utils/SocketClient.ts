/* eslint-disable @typescript-eslint/no-explicit-any, @typescript-eslint/no-unsafe-assignment, @typescript-eslint/no-unsafe-call */
import { createClient } from "@connectrpc/connect";
import { toast } from "react-hot-toast";
import {
  FlowService,
  type FlowMessage,
  type SyncRequest,
  type GraphSnapshot,
  type GraphMutation,
  type StreamChunk,
} from "../generated/core/service_pb";
import { type TaskUpdate } from "../generated/core/node_pb";
import { type ActionTemplate } from "../generated/action_pb";
import { type TaskDefinition, MutationSource } from "../types";
import { type WidgetSignal } from "../generated/core/signals_pb";
import { routerTransport } from "../mocks/routerTransport";

interface SocketEvents {
  snapshot: (data: GraphSnapshot) => void;
  yjsUpdate: (data: Uint8Array) => void;
  mutations: (data: GraphMutation[]) => void;
  actions: (data: ActionTemplate[]) => void;
  taskUpdate: (data: TaskDefinition) => void;
  widgetSignal: (data: WidgetSignal) => void;
  streamChunk: (data: StreamChunk) => void;
  error: (error: unknown) => void;
}

type Handler<K extends keyof SocketEvents> = SocketEvents[K];

function mapTaskUpdateToDefinition(update: TaskUpdate): TaskDefinition {
  return {
    taskId: update.taskId,
    type: "remote-task",
    label: update.message,
    source: MutationSource.REMOTE_TASK,
    status: update.status,
    progress: update.progress,
    message: update.message,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    mutationIds: [],
  };
}

class SocketClientImpl {
  private handlers: { [K in keyof SocketEvents]?: SocketEvents[K][] } = {};
  private streamHandlers: Record<string, (chunk: string) => void> = {};

  private transport = routerTransport;
  private client = createClient(FlowService, this.transport);

  on<K extends keyof SocketEvents>(event: K, handler: Handler<K>) {
    const handlers = (this.handlers[event] ?? []) as Handler<K>[];
    handlers.push(handler);
    this.handlers[event] = handlers as any;
  }

  off<K extends keyof SocketEvents>(event: K, handler: Handler<K>) {
    const list = this.handlers[event];
    if (!list) return;
    this.handlers[event] = (list as Handler<K>[]).filter(
      (h) => h !== handler,
    ) as any;
  }

  private emit<K extends keyof SocketEvents>(
    event: K,
    data: Parameters<SocketEvents[K]>[0],
  ) {
    const handlers = this.handlers[event];
    if (handlers) {
      handlers.forEach((h: any) => {
        h(data);
      });
    }
  }

  registerStreamHandler(
    nodeId: string,
    widgetId: string,
    handler: (chunk: string) => void,
  ) {
    this.streamHandlers[`${nodeId}-${widgetId}`] = handler;
  }

  async send(wrapper: { payload: FlowMessage["payload"] }) {
    const { case: type, value } = wrapper.payload;

    try {
      switch (type) {
        case "syncRequest":
          void this.startStream(value);
          break;

        case "nodeUpdate":
          await this.client.updateNode(value);
          break;

        case "widgetUpdate":
          await this.client.updateWidget(value);
          break;

        case "widgetSignal":
          await this.client.sendWidgetSignal(value);
          break;

        case "actionExecute":
          await this.client.executeAction(value);
          break;

        case "actionDiscovery": {
          const res = await this.client.discoverActions(value);
          this.emit("actions", res.actions);
          break;
        }

        case "taskCancel":
          await this.client.cancelTask(value);
          break;

        default:
          console.warn("Unknown message type:", type);
      }
    } catch (e) {
      console.error("gRPC Error:", e);
      toast.error(
        `Operation failed: ${e instanceof Error ? e.message : "Unknown error"}`,
      );
      this.emit("error", e);
    }
  }

  private async startStream(req: SyncRequest) {
    const toastId = "socket-stream";
    try {
      for await (const msg of this.client.watchGraph(req)) {
        toast.success("Connected to backend", { id: toastId });
        this.handleIncomingMessage(msg);
      }
    } catch (e) {
      console.error("Stream Error:", e);
      toast.error(
        `Connection lost: ${e instanceof Error ? e.message : "Unknown error"}`,
        { id: toastId },
      );
      this.emit("error", e);
    }
  }

  private handleIncomingMessage(msg: FlowMessage) {
    const payload = msg.payload;
    if (payload.case === undefined) return;

    switch (payload.case) {
      case "snapshot":
        this.emit("snapshot", payload.value);
        break;
      case "yjsUpdate":
        this.emit("yjsUpdate", payload.value);
        break;
      case "mutations":
        this.emit("mutations", payload.value.mutations);
        break;
      case "actions":
        this.emit("actions", payload.value.actions);
        break;
      case "taskUpdate":
        this.emit("taskUpdate", mapTaskUpdateToDefinition(payload.value));
        break;
      case "widgetSignal": {
        const signal = payload.value;
        void import("../store/flowStore").then(({ useFlowStore }) => {
          useFlowStore.getState().handleIncomingWidgetSignal(signal);
        });
        this.emit("widgetSignal", signal);
        break;
      }
      case "streamChunk": {
        const val = payload.value;
        const nodeId = val.nodeId;
        const widgetId = val.widgetId;
        const handlerKey = `${nodeId}-${widgetId}`;
        const handler = this.streamHandlers[handlerKey];
        if (handler) {
          handler(val.chunkData);
        }
        this.emit("streamChunk", val);
        break;
      }
    }
  }
}

export const socketClient = new SocketClientImpl();
