/* eslint-disable @typescript-eslint/no-unsafe-assignment */
/* eslint-disable @typescript-eslint/no-unsafe-argument */
import { createClient } from "@connectrpc/connect";
import { FlowService } from "../generated/core/service_pb";
import { type FlowMessage } from "../generated/core/service_pb";
import { routerTransport } from "../mocks/routerTransport";

type Handler = (data: unknown) => void;

class SocketClientImpl {
  private handlers: Record<string, Handler[]> = {};
  private streamHandlers: Record<string, (chunk: string) => void> = {};

  // In a real app, use createConnectTransport({ baseUrl: "..." })
  // Here we use the in-memory router transport to connect to our mock service directly.
  private transport = routerTransport;

  private client = createClient(FlowService, this.transport);

  on(event: string, handler: Handler) {
    this.handlers[event] = this.handlers[event] ?? [];
    this.handlers[event].push(handler);
  }

  off(event: string, handler: Handler) {
    const list = this.handlers[event];
    if (!list) return;
    this.handlers[event] = list.filter((h) => h !== handler);
  }

  private emit(event: string, data: unknown) {
    this.handlers[event]?.forEach((h) => {
      h(data);
    });
  }

  registerStreamHandler(
    nodeId: string,
    widgetId: string,
    handler: (chunk: string) => void,
  ) {
    this.streamHandlers[`${nodeId}-${widgetId}`] = handler;
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async send(wrapper: { payload: { case: string; value: any } }) {
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
      this.emit("error", e);
    }
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  private async startStream(req: any) {
    try {
      for await (const msg of this.client.watchGraph(req)) {
        this.handleIncomingMessage(msg);
      }
    } catch (e) {
      console.error("Stream Error:", e);
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
        this.emit("taskUpdate", payload.value);
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
