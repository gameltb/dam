import { v4 as uuidv4 } from "uuid";
import { flowcraft_proto } from "../generated/flowcraft_proto";

type Handler = (data: unknown) => void;

class SocketClientImpl {
  private handlers: Record<string, Handler[]> = {};
  private streamHandlers: Record<string, (chunk: string) => void> = {};

  on(event: string, handler: Handler) {
    // eslint-disable-next-line @typescript-eslint/prefer-nullish-coalescing
    this.handlers[event] = this.handlers[event] || [];
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

  async send(payload: flowcraft_proto.v1.IFlowMessage) {
    const message: flowcraft_proto.v1.IFlowMessage = {
      messageId: uuidv4(),
      timestamp: Date.now(),
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

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");

        // The last element is either an empty string (if buffer ends with \n)
        // or a partial line. Keep it in the buffer for the next chunk.
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const serverMsg = JSON.parse(
              line,
            ) as flowcraft_proto.v1.IFlowMessage;
            this.handleIncomingMessage(serverMsg);
          } catch (e) {
            console.error(
              "Failed to parse server message",
              e instanceof Error ? e.message : String(e),
              line,
            );
          }
        }
      }
    } catch (e) {
      this.emit("error", e);
    }
  }

  private handleIncomingMessage(msg: flowcraft_proto.v1.IFlowMessage) {
    if (msg.snapshot) {
      this.emit("snapshot", msg.snapshot);
    } else if (msg.yjsUpdate) {
      this.emit("yjsUpdate", msg.yjsUpdate);
    } else if (msg.mutations) {
      this.emit("mutations", msg.mutations.mutations);
    } else if (msg.actions) {
      this.emit("actions", msg.actions.actions);
    } else if (msg.taskUpdate) {
      this.emit("taskUpdate", msg.taskUpdate);
    } else if (msg.streamChunk) {
      const handlerKey = `${msg.streamChunk.nodeId ?? ""}-${msg.streamChunk.widgetId ?? ""}`;
      const handler = this.streamHandlers[handlerKey];
      if (handler) {
        handler(msg.streamChunk.chunkData ?? "");
      }
      this.emit("streamChunk", msg.streamChunk);
    }
  }
}

export const socketClient = new SocketClientImpl();
