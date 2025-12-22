import { useState, useCallback } from "react";
import { v4 as uuidv4 } from "uuid";
import type { AppNode, NodeData } from "../types";
import type { Edge } from "@xyflow/react";
import { useFlowStore } from "../store/flowStore";

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

// --- Mock Backend ---

const mockServerState = {
  version: 0,
  graph: {
    nodes: [] as AppNode[],
    edges: [] as Edge[],
  },
  availableActions: {
    text: [{ id: "generate-children", name: "Generate Children" }],
  },
};

// --- Custom Hook ---

export const useMockSocket = () => {
  const { nodes, handleNodeDataChange } = useFlowStore((state) => ({
    nodes: state.nodes,
    handleNodeDataChange: (nodeId: string, data: Partial<NodeData>) => {
      const { setNodes } = useFlowStore.getState();
      setNodes(
        state.nodes.map((n) =>
          n.id === nodeId
            ? ({ ...n, data: { ...n.data, ...data } } as AppNode)
            : n,
        ),
      );
    },
  }));

  const [lastJsonMessage, setLastJsonMessage] =
    useState<WebSocketMessage | null>(null);

  const sendJsonMessage = useCallback(
    (message: WebSocketMessage) => {
      const { type, payload } = message;

      if (type === "sync_graph") {
        const { version, graph } = payload;

        if (version === mockServerState.version) {
          mockServerState.version += 1;
          mockServerState.graph = graph;

          const response: WebSocketMessage = {
            type: "sync_graph",
            payload: {
              version: mockServerState.version,
              graph: mockServerState.graph,
            },
          };
          setLastJsonMessage(response);
        } else {
          const response: WebSocketMessage = {
            type: "sync_graph",
            payload: {
              version: mockServerState.version,
              graph: mockServerState.graph,
            },
            error: "version_mismatch",
          };
          setLastJsonMessage(response);
        }
      } else if (type === "execute_action") {
        const { actionId, nodeId } = payload;
        if (actionId === "generate-children") {
          const parentNode = nodes.find((n) => n.id === nodeId);
          if (!parentNode) return;

          const newNodes: AppNode[] = [
            {
              id: uuidv4(),
              type: "text",
              position: {
                x: parentNode.position.x + 200,
                y: parentNode.position.y - 50,
              },
              data: {
                label: "Child 1",
                onChange: handleNodeDataChange,
                outputType: "text",
                inputType: "any",
              },
            },
            {
              id: uuidv4(),
              type: "text",
              position: {
                x: parentNode.position.x + 200,
                y: parentNode.position.y + 50,
              },
              data: {
                label: "Child 2",
                onChange: handleNodeDataChange,
                outputType: "text",
                inputType: "any",
              },
            },
          ];

          const response: WebSocketMessage = {
            type: "apply_changes",
            payload: {
              add: newNodes,
              update: [],
            },
          };
          setLastJsonMessage(response);
        }
      }
    },
    [nodes, handleNodeDataChange],
  );

  return {
    sendJsonMessage,
    lastJsonMessage,
    readyState: 1, // Mock connected state
    mockServerState, // Expose for use in ContextMenu
  };
};
