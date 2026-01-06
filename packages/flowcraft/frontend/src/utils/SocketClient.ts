/* eslint-disable @typescript-eslint/no-explicit-any */
import { createConnectTransport } from "@connectrpc/connect-web";
import { createClient, ConnectError, Code } from "@connectrpc/connect";
import { toast } from "react-hot-toast";
import {
  FlowService,
  type FlowMessage,
  type SyncRequest,
  type GraphSnapshot,
  type GraphMutation,
  type StreamChunk,
  type TemplateDiscoveryResponse,
} from "../generated/flowcraft/v1/core/service_pb";
import { type TaskUpdate } from "../generated/flowcraft/v1/core/node_pb";
import { type WidgetSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { type ActionTemplate } from "../generated/flowcraft/v1/core/action_pb";
import { type TaskDefinition, MutationSource } from "../types";

export enum SocketStatus {
  DISCONNECTED = "disconnected",
  CONNECTING = "connecting",
  INITIALIZING = "initializing",
  CONNECTED = "connected",
  ERROR = "error",
}

interface SocketEvents {
  snapshot: (data: GraphSnapshot) => void;
  yjsUpdate: (data: Uint8Array) => void;
  mutations: (data: GraphMutation[]) => void;
  actions: (data: ActionTemplate[]) => void;
  taskUpdate: (data: TaskDefinition) => void;
  widgetSignal: (data: WidgetSignal) => void;
  streamChunk: (data: StreamChunk) => void;
  templates: (data: TemplateDiscoveryResponse) => void;
  error: (error: unknown) => void;
  statusChange: (status: SocketStatus) => void;
}

type Handler<K extends keyof SocketEvents> = SocketEvents[K];

function mapTaskUpdateToDefinition(update: TaskUpdate): TaskDefinition {
  return {
    taskId: update.taskId,
    type: "remote-task",
    label: update.message,
    source: MutationSource.SOURCE_REMOTE_TASK,
    status: update.status,
    progress: update.progress,
    message: update.message,
    createdAt: Date.now(),
    updatedAt: Date.now(),
    mutationIds: [],
  };
}

class SocketClientImpl {
  private handlers: Partial<{ [K in keyof SocketEvents]: Handler<K>[] }> = {};
  private streamHandlers: Record<string, (chunk: string) => void> = {};
  private activeStreamAbort: AbortController | null = null;
  private currentBaseUrl =
    typeof window !== "undefined"
      ? window.location.origin
      : "http://localhost:3000";
  private currentStatus: SocketStatus = SocketStatus.DISCONNECTED;

  private transport = createConnectTransport({
    baseUrl: this.currentBaseUrl,
  });

  private client = createClient(FlowService, this.transport);

  public updateBaseUrl(newUrl: string) {
    if (!newUrl) return;

    // Resolve relative URL
    let resolvedUrl = newUrl;
    if (newUrl.startsWith("/") && typeof window !== "undefined") {
      resolvedUrl = window.location.origin + (newUrl === "/" ? "" : newUrl);
    }

    if (this.currentBaseUrl === resolvedUrl) return;
    console.log(
      `[SocketClient] Updating Base URL from ${this.currentBaseUrl} to ${resolvedUrl}`,
    );
    this.currentBaseUrl = resolvedUrl;

    try {
      // Create new transport and client
      this.transport = createConnectTransport({
        baseUrl: this.currentBaseUrl,
      });
      this.client = createClient(FlowService, this.transport);

      // If we have an active stream, restart it
      if (this.activeStreamAbort) {
        console.log("[SocketClient] Base URL changed, restarting stream...");
        this.activeStreamAbort.abort();
      }
    } catch (err) {
      console.error("[SocketClient] Failed to update transport:", err);
    }
  }

  public getStatus() {
    return this.currentStatus;
  }

  private setStatus(status: SocketStatus) {
    this.currentStatus = status;
    this.emit("statusChange", status);
  }

  on<K extends keyof SocketEvents>(event: K, handler: Handler<K>) {
    if (!this.handlers[event]) {
      this.handlers[event] = [] as any;
    }
    this.handlers[event]!.push(handler);
  }

  off<K extends keyof SocketEvents>(event: K, handler: Handler<K>) {
    const list = this.handlers[event];
    if (!list) return;
    this.handlers[event] = list.filter((h) => h !== handler) as any;
  }

  private emit<K extends keyof SocketEvents>(
    event: K,
    data: Parameters<SocketEvents[K]>[0],
  ) {
    const handlers = this.handlers[event];
    if (handlers) {
      (handlers as ((arg: typeof data) => void)[]).forEach((h) => {
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

  async getChatHistory(headId: string) {
    try {
      return await this.client.getChatHistory({ headId });
    } catch (e) {
      console.error("[SocketClient] Failed to get chat history:", e);
      throw e;
    }
  }

  async send(wrapper: { payload: FlowMessage["payload"] }) {
    if (wrapper.payload.case === undefined) return;

    try {
      switch (wrapper.payload.case) {
        case "syncRequest":
          void this.startStream(wrapper.payload.value);
          break;

        case "nodeUpdate":
          await this.client.updateNode(wrapper.payload.value);
          break;

        case "widgetUpdate":
          await this.client.updateWidget(wrapper.payload.value);
          break;

        case "widgetSignal":
          await this.client.sendWidgetSignal(wrapper.payload.value);
          break;

        case "actionExecute":
          await this.client.executeAction(wrapper.payload.value);
          break;

        case "actionDiscovery": {
          const res = await this.client.discoverActions(wrapper.payload.value);
          this.emit("actions", res.actions);
          break;
        }

        case "mutations": {
          await this.client.applyMutations(wrapper.payload.value);
          break;
        }

        case "templateDiscovery": {
          const res = await this.client.discoverTemplates(
            wrapper.payload.value,
          );
          this.emit("templates", res);
          break;
        }

        case "taskCancel":
          await this.client.cancelTask(wrapper.payload.value);
          break;

        case "viewportUpdate":
          await this.client.updateViewport(wrapper.payload.value);
          break;

        default:
          // Use type narrowing to ensure all cases are handled if possible
          console.warn("Unknown message type:", (wrapper.payload as any).case);
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
    if (this.activeStreamAbort) {
      this.activeStreamAbort.abort();
    }
    const abortController = new AbortController();
    this.activeStreamAbort = abortController;
    const signal = abortController.signal;

    this.setStatus(SocketStatus.CONNECTING);
    const toastId = "socket-stream";
    try {
      let isFirstMessage = true;
      for await (const msg of this.client.watchGraph(req, { signal })) {
        if (signal.aborted) break;

        if (isFirstMessage) {
          isFirstMessage = false;
          this.setStatus(SocketStatus.INITIALIZING);
          toast.success("Connection established, initializing data...", {
            id: toastId,
          });
        }

        this.handleIncomingMessage(msg);

        // After first message (usually snapshot), if we are still initializing, set to connected
        if (this.currentStatus === SocketStatus.INITIALIZING) {
          this.setStatus(SocketStatus.CONNECTED);
          toast.success("Ready", { id: toastId, duration: 2000 });
        }
      }
    } catch (e) {
      if (e instanceof Error && e.name === "AbortError") {
        this.setStatus(SocketStatus.DISCONNECTED);
        return;
      }
      if (e instanceof ConnectError && e.code === Code.Canceled) {
        this.setStatus(SocketStatus.DISCONNECTED);
        return;
      }
      console.error("Stream Error:", e);
      if (e instanceof ConnectError) {
        console.error("ConnectError Details:", {
          code: e.code,
          message: e.message,
          details: e.details,
          rawMessage: e.rawMessage,
        });
      }
      this.setStatus(SocketStatus.ERROR);
      toast.error(
        `Connection lost: ${e instanceof Error ? e.message : "Unknown error"}`,
        { id: toastId },
      );
      this.emit("error", e);
    } finally {
      if (this.activeStreamAbort === abortController) {
        this.activeStreamAbort = null;
        if (this.currentStatus === SocketStatus.CONNECTED) {
          this.setStatus(SocketStatus.DISCONNECTED);
        }
      }
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
      case "templates":
        this.emit("templates", payload.value);
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
