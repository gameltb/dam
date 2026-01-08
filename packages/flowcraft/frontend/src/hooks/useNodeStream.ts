import { useEffect } from "react";

import { type NodeEvent } from "@/generated/flowcraft/v1/core/service_pb";
import { socketClient } from "@/utils/SocketClient";

export function useNodeStream(
  nodeId: string,
  onChunk: (chunk: string, isDone: boolean) => void,
) {
  useEffect(() => {
    const handler = (data: NodeEvent) => {
      if (data.nodeId === nodeId) {
        if (data.payload.case === "chatStream") {
          onChunk(data.payload.value.chunkData, data.payload.value.isDone);
        }
      }
    };

    socketClient.on("nodeEvent", handler);
    return () => {
      socketClient.off("nodeEvent", handler);
    };
  }, [nodeId, onChunk]);
}
