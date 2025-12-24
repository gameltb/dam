import { useState, useCallback, useEffect, useMemo } from "react";
import type { AppNode, NodeTemplate } from "../types";
import type { Edge } from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";
import { hydrateNodes } from "../utils/nodeUtils";

// --- Types ---

interface SyncGraphPayload {
  version: number;
  graph: {
    nodes: AppNode[];
    edges: Edge[];
  };
}

export interface WebSocketMessage {
  type: "sync_graph" | "execute_action" | "apply_changes";
  payload: SyncGraphPayload | any; // eslint-disable-line @typescript-eslint/no-explicit-any
  error?: string;
}

// --- Custom Hook (Refactored to use MSW via HTTP Polling) ---

export const useMockSocket = () => {
  const { updateNodeData, dispatchNodeEvent } = useFlowStore((state) => ({
    updateNodeData: state.updateNodeData,
    dispatchNodeEvent: state.dispatchNodeEvent,
  }));

  const [lastJsonMessage, setLastJsonMessage] =
    useState<WebSocketMessage | null>(null);
  const [templates, setTemplates] = useState<NodeTemplate[]>([]);

  const nodeHandlers = useMemo(
    () => ({
      onChange: (id: string, d: Record<string, unknown>) =>
        updateNodeData(id, d),
      onWidgetClick: (nodeId: string, widgetId: string) => {
        dispatchNodeEvent("widget-click", { nodeId, widgetId });
      },
      onGalleryItemContext: (
        nodeId: string,
        url: string,
        x: number,
        y: number,
      ) => {
        dispatchNodeEvent("gallery-context", { nodeId, url, x, y });
      },
    }),
    [updateNodeData, dispatchNodeEvent],
  );

  // Fetch templates on load
  useEffect(() => {
    fetch("/api/node-templates")
      .then((res) => res.json())
      .then(setTemplates)
      .catch((err) => console.error("Template fetch error:", err));
  }, []);

  // Poll for updates (Simulation of WebSocket push)
  useEffect(() => {
    const pollInterval = setInterval(async () => {
      try {
        const response = await fetch("/api/graph");
        if (response.ok) {
          const data = await response.json();

          if (data.type === "sync_graph" && data.payload.graph.nodes) {
            data.payload.graph.nodes = hydrateNodes(
              data.payload.graph.nodes,
              nodeHandlers,
            );
          }

          setLastJsonMessage(data);
        }
      } catch (error) {
        console.error("Polling error:", error);
      }
    }, 2000); // Poll every 2 seconds

    return () => clearInterval(pollInterval);
  }, [nodeHandlers]);

  const sendJsonMessage = useCallback(
    async (message: WebSocketMessage) => {
      const { type, payload } = message;

      if (type === "sync_graph") {
        try {
          const response = await fetch("/api/graph", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json();
          setLastJsonMessage(data);
        } catch (error) {
          console.error("Sync error:", error);
        }
      } else if (type === "execute_action") {
        try {
          const response = await fetch("/api/action", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
          });
          const data = await response.json();

          if (data.type === "apply_changes" && data.payload.add) {
            data.payload.add = hydrateNodes(data.payload.add, nodeHandlers);
          }
          setLastJsonMessage(data);
        } catch (error) {
          console.error("Action error:", error);
        }
      }
    },
    [nodeHandlers],
  );

  const mockServerState = useMemo(
    () => ({
      availableActions: {
        text: [{ id: "generate-children", name: "Generate Children" }],
      },
    }),
    [],
  );

  return {
    sendJsonMessage,
    lastJsonMessage,
    templates,
    readyState: 1, // Mock connected state
    mockServerState, // Static config
  };
};
