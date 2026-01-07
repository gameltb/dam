import { Code, ConnectError, createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";
import { toast } from "react-hot-toast";

import { type ActionTemplate } from "../generated/flowcraft/v1/core/action_pb";
import { type TaskUpdate } from "../generated/flowcraft/v1/core/node_pb";
import {
  type FlowMessage,
  FlowService,
  type GraphMutation,
  type GraphSnapshot,
  type NodeEvent,
  type SyncRequest,
  type TemplateDiscoveryResponse,
  type WidgetStreamEvent,
} from "../generated/flowcraft/v1/core/service_pb";
import { type WidgetSignal } from "../generated/flowcraft/v1/core/signals_pb";
import { MutationSource, type TaskDefinition } from "../types";

export enum SocketStatus {
  CONNECTED = "connected",
  CONNECTING = "connecting",
  DISCONNECTED = "disconnected",
  ERROR = "error",
  INITIALIZING = "initializing",
}

type Handler<K extends keyof SocketEventMap> = (
  data: SocketEventMap[K],
) => void;

interface SocketEventMap {
  actions: ActionTemplate[];
  error: unknown;
  mutations: GraphMutation[];
  nodeEvent: NodeEvent;
  "nodeEvent:widgetStream": {
    event: WidgetStreamEvent;
    nodeId: string;
  };
  snapshot: GraphSnapshot;
  statusChange: SocketStatus;
  taskUpdate: TaskDefinition;
  templates: TemplateDiscoveryResponse;
  widgetSignal: WidgetSignal;
  yjsUpdate: Uint8Array;
}

class SocketClientImpl {
  private activeStreamAbort: AbortController | null = null;
  private currentBaseUrl =
    typeof window !== "undefined"
      ? window.location.origin
      : "http://localhost:3000";
  private transport = createConnectTransport({
    baseUrl: this.currentBaseUrl,
  });
  private client = createClient(FlowService, this.transport);

  private currentStatus: SocketStatus = SocketStatus.DISCONNECTED;

  private handlers: Partial<{ [K in keyof SocketEventMap]: Handler<K>[] }> = {};

  async getChatHistory(headId: string) {
    try {
      return await this.client.getChatHistory({ headId });
    } catch (e) {
      console.error("[SocketClient] Failed to get chat history:", e);
      throw e;
    }
  }

  public getStatus() {
    return this.currentStatus;
  }

  off<K extends keyof SocketEventMap>(event: K, handler: Handler<K>) {
    const list = this.handlers[event];
    if (!list) return;
    // @ts-expect-error - TS cannot verify generic key assignment in Partial record
    this.handlers[event] = list.filter((h) => h !== handler);
  }

  on<K extends keyof SocketEventMap>(event: K, handler: Handler<K>) {
    let list = this.handlers[event];
    if (!list) {
      list = [];
      this.handlers[event] = list as (typeof this.handlers)[K];
    }
    list.push(handler);
  }

  async send(wrapper: { payload: FlowMessage["payload"] }) {
    const payload = wrapper.payload;
    if (payload.case === undefined) return;

    try {
      switch (payload.case) {
        case "actionDiscovery": {
          const res = await this.client.discoverActions(payload.value);
          this.emit("actions", res.actions);
          break;
        }

        case "actionExecute":
          await this.client.executeAction(payload.value);
          break;

        case "actions":
        case "error":
        case "nodeEvent":
        case "snapshot":
        case "taskUpdate":
        case "templates":
          console.warn(
            `[SocketClient] Client tried to send server-only message type: ${payload.case}`,
          );
          break;

        case "mutations": {
          await this.client.applyMutations(payload.value);
          break;
        }

        case "nodeUpdate":
          await this.client.updateNode(payload.value);
          break;

        case "syncRequest":
          void this.startStream(payload.value);
          break;

        case "taskCancel":
          await this.client.cancelTask(payload.value);
          break;

        case "templateDiscovery": {
          const res = await this.client.discoverTemplates(payload.value);
          this.emit("templates", res);
          break;
        }

        case "viewportUpdate":
          await this.client.updateViewport(payload.value);
          break;

        case "widgetSignal":
          await this.client.sendWidgetSignal(payload.value);
          break;

        case "widgetUpdate":
          await this.client.updateWidget(payload.value);
          break;

        case "yjsUpdate":
          // Client rarely sends Yjs updates directly via send, but it's possible
          break;

        default: {
          // Use type narrowing to ensure all cases are handled
          const _exhaustiveCheck: never = payload;
          return _exhaustiveCheck;
        }
      }
    } catch (e) {
      console.error("gRPC Error:", e);
      toast.error(
        `Operation failed: ${e instanceof Error ? e.message : "Unknown error"}`,
      );
      this.emit("error", e);
    }
  }

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

  private emit<K extends keyof SocketEventMap>(
    event: K,
    data: SocketEventMap[K],
  ) {
    const handlers = this.handlers[event];
    if (handlers) {
      handlers.forEach((h: Handler<K>) => {
        h(data);
      });
    }
  }

  private handleIncomingMessage(msg: FlowMessage) {
    const payload = msg.payload;
    if (payload.case === undefined) return;

    switch (payload.case) {
      case "actions":
        this.emit("actions", payload.value.actions);
        break;
      case "mutations":
        this.emit("mutations", payload.value.mutations);
        break;
      case "nodeEvent": {
        const event = payload.value;
        this.emit("nodeEvent", event);

        // Dispatch granular events based on payload case
        if (event.payload.case === "widgetStream") {
          this.emit("nodeEvent:widgetStream", {
            event: event.payload.value,
            nodeId: event.nodeId,
          });
        }
        break;
      }
      case "snapshot":
        this.emit("snapshot", payload.value);
        break;
      case "taskUpdate":
        this.emit("taskUpdate", mapTaskUpdateToDefinition(payload.value));
        break;
      case "templates":
        this.emit("templates", payload.value);
        break;
      case "widgetSignal": {
        const signal = payload.value;
        void import("../store/flowStore").then(({ useFlowStore }) => {
          useFlowStore.getState().handleIncomingWidgetSignal(signal);
        });
        this.emit("widgetSignal", signal);
        break;
      }
      case "yjsUpdate":
        this.emit("yjsUpdate", payload.value);
        break;
    }
  }

  private setStatus(status: SocketStatus) {
    this.currentStatus = status;
    this.emit("statusChange", status);
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
          toast.success("Ready", { duration: 2000, id: toastId });
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
          details: e.details,
          message: e.message,
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
}

function mapTaskUpdateToDefinition(update: TaskUpdate): TaskDefinition {
  return {
    createdAt: Date.now(),
    label: update.message,
    message: update.message,
    mutationIds: [],
    progress: update.progress,
    source: MutationSource.SOURCE_REMOTE_TASK,
    status: update.status,
    taskId: update.taskId,
    type: "remote-task",
    updatedAt: Date.now(),
  };
}

export const socketClient = new SocketClientImpl();
