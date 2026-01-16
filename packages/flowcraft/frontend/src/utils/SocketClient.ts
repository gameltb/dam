import { createClient } from "@connectrpc/connect";
import { createConnectTransport } from "@connectrpc/connect-web";

import { type ActionTemplate } from "@/generated/flowcraft/v1/core/action_pb";
import {
  type FlowMessage,
  FlowService,
  type InferenceConfigDiscoveryResponse,
  type TemplateDiscoveryResponse,
} from "@/generated/flowcraft/v1/core/service_pb";

export enum SocketStatus {
  CONNECTED = "connected",
  CONNECTING = "connecting",
  DISCONNECTED = "disconnected",
  ERROR = "error",
  INITIALIZING = "initializing",
}

type Handler<K extends keyof SocketEventMap> = (data: SocketEventMap[K]) => void;

interface SocketEventMap {
  actions: ActionTemplate[];
  error: unknown;
  inferenceConfig: InferenceConfigDiscoveryResponse;
  statusChange: SocketStatus;
  templates: TemplateDiscoveryResponse;
}

class SocketClientImpl {
  private currentBaseUrl = typeof window !== "undefined" ? window.location.origin : "http://localhost:3000";
  private transport = createConnectTransport({
    baseUrl: this.currentBaseUrl,
  });
  private client = createClient(FlowService, this.transport);

  private currentStatus: SocketStatus = SocketStatus.CONNECTED; // Default to connected for discovery

  private handlers: Partial<{ [K in keyof SocketEventMap]: Handler<K>[] }> = {};

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

        case "inferenceDiscovery": {
          const res = await this.client.discoverInferenceConfig(payload.value);
          this.emit("inferenceConfig", res);
          break;
        }

        case "templateDiscovery": {
          const res = await this.client.discoverTemplates(payload.value);
          this.emit("templates", res);
          break;
        }

        default: {
          console.warn(`[SocketClient] Handled case ${payload.case} is legacy/unsupported in discovery mode.`);
          break;
        }
      }
    } catch (e) {
      console.error("Discovery Error:", e);
      this.emit("error", e);
    }
  }

  public updateBaseUrl(newUrl: string) {
    if (!newUrl) return;

    let resolvedUrl = newUrl;
    if (newUrl.startsWith("/") && typeof window !== "undefined") {
      resolvedUrl = window.location.origin + (newUrl === "/" ? "" : newUrl);
    }

    if (this.currentBaseUrl === resolvedUrl) return;
    this.currentBaseUrl = resolvedUrl;

    try {
      this.transport = createConnectTransport({
        baseUrl: this.currentBaseUrl,
      });
      this.client = createClient(FlowService, this.transport);
    } catch (err) {
      console.error("[SocketClient] Failed to update transport:", err);
    }
  }

  private emit<K extends keyof SocketEventMap>(event: K, data: SocketEventMap[K]) {
    const handlers = this.handlers[event];
    if (handlers) {
      handlers.forEach((h: Handler<K>) => {
        h(data);
      });
    }
  }
}

export const socketClient = new SocketClientImpl();
