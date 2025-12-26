import { v4 as uuidv4 } from "uuid";
import { flowcraft } from "../generated/flowcraft";

type Handler = (data: unknown) => void;

class SocketClient {
  private static instance: SocketClient;
  private handlers: Record<string, Handler[]> = {};
  private streamHandlers: Record<string, (chunk: string) => void> = {};

  private constructor() {}

  static getInstance() {
    if (!SocketClient.instance) {
      SocketClient.instance = new SocketClient();
    }
    return SocketClient.instance;
  }

  on(event: string, handler: Handler) {
    if (!this.handlers[event]) this.handlers[event] = [];
    this.handlers[event].push(handler);
  }

  off(event: string, handler: Handler) {
    if (!this.handlers[event]) return;
    this.handlers[event] = this.handlers[event].filter((h) => h !== handler);
  }

  private emit(event: string, data: unknown) {
    if (!this.handlers[event]) return;
    this.handlers[event].forEach((h) => h(data));
  }

  registerStreamHandler(
    nodeId: string,
    widgetId: string,
    handler: (chunk: string) => void,
  ) {
    this.streamHandlers[`${nodeId}-${widgetId}`] = handler;
  }

  async send(payload: flowcraft.v1.IFlowMessage) {
    const message: flowcraft.v1.IFlowMessage = {
      messageId: uuidv4(),
      timestamp: Date.now() as unknown as number,
      ...payload,
    };

    try {
      const response = await fetch("/api/ws", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(message),
      });

      if (!response.ok) return;

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) return;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        const chunk = decoder.decode(value, { stream: true });
        const lines = chunk.split("\n").filter((l) => l.trim());

        for (const line of lines) {
          try {
            const serverMsg = JSON.parse(line) as flowcraft.v1.IFlowMessage;
            this.handleIncomingMessage(serverMsg);
          } catch (e) {
            console.error("Failed to parse server message", e);
          }
        }
      }
    } catch (e) {
      this.emit("error", e);
    }
  }

  private handleIncomingMessage(msg: flowcraft.v1.IFlowMessage) {
    if (msg.snapshot) {
      this.emit("snapshot", msg.snapshot);
    } else if (msg.yjsUpdate) {
      this.emit("yjsUpdate", msg.yjsUpdate);
    } else if (msg.mutations) {
      this.emit("mutations", msg.mutations.mutations);
    } else if (msg.taskUpdate) {
      this.emit("taskUpdate", msg.taskUpdate);
    } else if (msg.streamChunk) {
      const handlerKey = `${msg.streamChunk.nodeId}-${msg.streamChunk.widgetId}`;
      if (this.streamHandlers[handlerKey]) {
        this.streamHandlers[handlerKey](msg.streamChunk.chunkData!);
      }
      this.emit("streamChunk", msg.streamChunk);
    }
  }
}

export const socketClient = SocketClient.getInstance();
