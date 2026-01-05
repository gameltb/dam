import { useEffect } from "react";
import { socketClient } from "../utils/SocketClient";
import { type StreamChunk } from "../generated/flowcraft/v1/core/service_pb";

export function useNodeStream(
  nodeId: string,
  onChunk: (chunk: string, isDone: boolean) => void,
) {
  useEffect(() => {
    const handler = (data: StreamChunk) => {
      if (data.nodeId === nodeId) {
        onChunk(data.chunkData, data.isDone);
      }
    };

    socketClient.on("streamChunk", handler);
    return () => {
      socketClient.off("streamChunk", handler);
    };
  }, [nodeId, onChunk]);
}
