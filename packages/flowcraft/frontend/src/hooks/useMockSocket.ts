import { useState, useCallback, useEffect, useMemo } from "react";
import { flowcraft } from "../generated/flowcraft";
import type { NodeTemplate, AppNode } from "../types";
import { MediaType } from "../types";
import { useFlowStore } from "../store/flowStore";
import { hydrateNodes } from "../utils/nodeUtils";
import { useTaskStore } from "../store/taskStore";
import { v4 as uuidv4 } from "uuid";
import type { Edge } from "@xyflow/react";

export interface WebSocketMessage {
  type: string;
  payload: unknown;
}

export const useMockSocket = (config?: { disablePolling?: boolean }) => {
  const { updateNodeData, dispatchNodeEvent, applyMutations, setGraph } =
    useFlowStore((state) => ({
      updateNodeData: state.updateNodeData,
      dispatchNodeEvent: state.dispatchNodeEvent,
      applyMutations: state.applyMutations,
      setGraph: state.setGraph,
    }));

  const { updateTask } = useTaskStore();
  const [templates, setTemplates] = useState<NodeTemplate[]>([]);
  const [streamHandlers, setStreamHandlers] = useState<
    Record<string, (chunk: string) => void>
  >({});

  const nodeHandlers = useMemo(
    () => ({
      onChange: (id: string, d: Record<string, unknown>) =>
        updateNodeData(id, d),
      onWidgetClick: (nodeId: string, widgetId: string) =>
        dispatchNodeEvent("widget-click", { nodeId, widgetId }),
      onGalleryItemContext: (
        nodeId: string,
        url: string,
        mediaType: MediaType,
        x: number,
        y: number,
      ) => {
        dispatchNodeEvent("gallery-context", { nodeId, url, mediaType, x, y });
      },
    }),
    [updateNodeData, dispatchNodeEvent],
  );

  // Unified Message Sender
  const sendFlowMessage = useCallback(
    async (payload: flowcraft.v1.IFlowMessage) => {
      const message: flowcraft.v1.IFlowMessage = {
        messageId: uuidv4(),
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        timestamp: Date.now() as any,
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
            const serverMsg = JSON.parse(line) as flowcraft.v1.IFlowMessage;

            if (serverMsg.snapshot) {
              setGraph(
                serverMsg.snapshot as unknown as {
                  nodes: AppNode[];
                  edges: Edge[];
                },
                serverMsg.snapshot.version as unknown as number,
              );
            } else if (serverMsg.mutations) {
              const processed = serverMsg.mutations.mutations?.map((m) => {
                if (m.addNode?.node)
                  m.addNode.node = hydrateNodes(
                    [m.addNode.node as unknown as AppNode],
                    nodeHandlers,
                  )[0] as unknown as flowcraft.v1.INode;
                return m;
              });
              applyMutations(processed || []);
            } else if (serverMsg.taskUpdate) {
              updateTask(
                serverMsg.taskUpdate.taskId!,
                // eslint-disable-next-line @typescript-eslint/no-explicit-any
                serverMsg.taskUpdate as any,
              );
            } else if (serverMsg.streamChunk) {
              const handlerKey = `${serverMsg.streamChunk.nodeId}-${serverMsg.streamChunk.widgetId}`;
              streamHandlers[handlerKey]?.(serverMsg.streamChunk.chunkData!);
            }
          }
        }
      } catch (e) {
        console.error("WS error:", e);
      }
    },
    [setGraph, applyMutations, updateTask, streamHandlers, nodeHandlers],
  );

  // Initial Fetch
  useEffect(() => {
    if (config?.disablePolling) return;
    fetch("/api/node-templates")
      .then((r) => r.json())
      .then((data) => setTemplates(data));

    // Defer initial sync to avoid effect cascading render warning
    const t = setTimeout(() => {
      sendFlowMessage({ syncRequest: { graphId: "main" } });
    }, 0);
    return () => clearTimeout(t);
  }, [config?.disablePolling, sendFlowMessage]);

  // Public API Mapping
  const sendNodeUpdate = (nodeId: string, data: Partial<AppNode["data"]>) =>
    sendFlowMessage({ nodeUpdate: { nodeId, data: data as any } }); // eslint-disable-line @typescript-eslint/no-explicit-any

  const sendWidgetUpdate = (nodeId: string, widgetId: string, value: unknown) =>
    sendFlowMessage({
      widgetUpdate: { nodeId, widgetId, valueJson: JSON.stringify(value) },
    });

  const fetchWidgetOptions = async (nodeId: string, widgetId: string) => {
    try {
      const res = await fetch("/api/widget/options", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId, widgetId }),
      });
      return await res.json();
    } catch {
      return [];
    }
  };

  const executeAction = async (actionId: string, sourceNodeId: string) => {
    await sendFlowMessage({ actionExecute: { actionId, sourceNodeId } });
    return { type: "immediate" };
  };

  const streamAction = (
    nodeId: string,
    widgetId: string,
    onChunk: (c: string) => void,
  ) => {
    setStreamHandlers((prev) => ({
      ...prev,
      [`${nodeId}-${widgetId}`]: onChunk,
    }));
    sendFlowMessage({
      actionExecute: { actionId: "stream", sourceNodeId: nodeId },
    });
  };

  const executeTask = (
    taskId: string,
    type: string,
    params: { sourceNodeId: string },
  ) => {
    sendFlowMessage({
      actionExecute: {
        actionId: type,
        sourceNodeId: params.sourceNodeId,
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        paramsJson: JSON.stringify({ taskId }) as any,
      },
    });
  };

  const cancelTask = (taskId: string) =>
    sendFlowMessage({ taskCancel: { taskId } });

  const discoverActions = async (nodeId?: string) => {
    try {
      const res = await fetch("/api/actions/discover", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ nodeId }),
      });
      const data = await res.json();
      return data.actions as flowcraft.v1.IActionTemplate[];
    } catch {
      return [];
    }
  };

  return {
    sendNodeUpdate,
    sendWidgetUpdate,
    fetchWidgetOptions,
    executeAction,
    streamAction,
    executeTask,
    cancelTask,
    discoverActions,
    templates,
    readyState: 1,
    mockServerState: { availableActions: { text: [] } },
  };
};
