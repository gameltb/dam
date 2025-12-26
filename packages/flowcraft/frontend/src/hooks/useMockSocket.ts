import { useState, useEffect, useMemo } from "react";
import { flowcraft } from "../generated/flowcraft";
import type { NodeTemplate, AppNode, TaskDefinition } from "../types";
import { MediaType, MutationSource, TaskStatus } from "../types";
import { useFlowStore } from "../store/flowStore";
import { hydrateNodes } from "../utils/nodeUtils";
import { useTaskStore } from "../store/taskStore";
import { socketClient } from "../utils/SocketClient";
import { ProtoAdapter } from "../utils/protoAdapter";

export const useMockSocket = (config?: { disablePolling?: boolean }) => {
  const {
    updateNodeData,
    dispatchNodeEvent,
    applyMutations,
    setGraph,
    ydoc,
    applyYjsUpdate,
  } = useFlowStore((state) => ({
    updateNodeData: state.updateNodeData,
    dispatchNodeEvent: state.dispatchNodeEvent,
    applyMutations: state.applyMutations,
    setGraph: state.setGraph,
    ydoc: state.ydoc,
    applyYjsUpdate: state.applyYjsUpdate,
  }));

  const { updateTask } = useTaskStore();
  const [templates, setTemplates] = useState<NodeTemplate[]>([]);

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

  useEffect(() => {
    const onSnapshot = (snapshot: flowcraft.v1.IGraphSnapshot) => {
      const { nodes, edges } = ProtoAdapter.fromProtoGraph(snapshot);
      const hydratedNodes = hydrateNodes(nodes, nodeHandlers);

      const taskId = "initial-sync";
      useTaskStore.getState().registerTask({
        taskId,
        label: "Initial Graph Sync",
        source: MutationSource.SYNC,
        status: TaskStatus.TASK_COMPLETED,
      });

      setGraph({ nodes: hydratedNodes, edges }, Number(snapshot.version || 0));
    };

    const onYjsUpdate = (update: Uint8Array) => {
      applyYjsUpdate(update);
    };

    const onMutations = (mutations: flowcraft.v1.IGraphMutation[]) => {
      const processed = mutations.map((m) => {
        const cloned = { ...m };
        if (cloned.addNode?.node) {
          const appNode = ProtoAdapter.fromProtoNode(cloned.addNode.node);
          cloned.addNode.node = hydrateNodes(
            [appNode],
            nodeHandlers,
          )[0] as flowcraft.v1.INode;
        }
        if (cloned.addSubgraph) {
          if (cloned.addSubgraph.nodes) {
            const appNodes = cloned.addSubgraph.nodes.map((n) =>
              ProtoAdapter.fromProtoNode(n),
            );
            cloned.addSubgraph.nodes = hydrateNodes(
              appNodes,
              nodeHandlers,
            ) as flowcraft.v1.INode[];
          }
        }
        return cloned;
      });

      applyMutations(processed, {
        source: MutationSource.REMOTE_TASK,
        description: "Mutations from server",
      });
    };

    const onTaskUpdate = (update: TaskDefinition) => {
      updateTask(update.taskId, update);
    };

    socketClient.on("snapshot", onSnapshot);
    socketClient.on("yjsUpdate", onYjsUpdate);
    socketClient.on("mutations", onMutations);
    socketClient.on("taskUpdate", onTaskUpdate);

    const handleLocalUpdate = (update: Uint8Array, origin: unknown) => {
      if (origin === "local") {
        socketClient.send({ yjsUpdate: update });
      }
    };
    ydoc.on("update", handleLocalUpdate);

    return () => {
      socketClient.off("snapshot", onSnapshot);
      socketClient.off("yjsUpdate", onYjsUpdate);
      socketClient.off("mutations", onMutations);
      socketClient.off("taskUpdate", onTaskUpdate);
      ydoc.off("update", handleLocalUpdate);
    };
  }, [
    setGraph,
    applyMutations,
    updateTask,
    nodeHandlers,
    ydoc,
    applyYjsUpdate,
  ]);

  useEffect(() => {
    if (config?.disablePolling) return;
    fetch("/api/node-templates")
      .then((r) => r.json())
      .then((data) => setTemplates(data));

    socketClient.send({ syncRequest: { graphId: "main" } });
  }, [config?.disablePolling]);

  const sendNodeUpdate = (nodeId: string, data: Partial<AppNode["data"]>) =>
    socketClient.send({
      nodeUpdate: { nodeId, data: data as unknown as flowcraft.v1.INodeData },
    });

  const sendWidgetUpdate = (nodeId: string, widgetId: string, value: unknown) =>
    socketClient.send({
      widgetUpdate: { nodeId, widgetId, valueJson: JSON.stringify(value) },
    });

  const streamAction = (
    nodeId: string,
    widgetId: string,
    onChunk: (c: string) => void,
  ) => {
    socketClient.registerStreamHandler(nodeId, widgetId, onChunk);
    socketClient.send({
      actionExecute: { actionId: "stream", sourceNodeId: nodeId },
    });
  };

  const executeTask = (
    taskId: string,
    type: string,
    params: { sourceNodeId: string },
  ) => {
    socketClient.send({
      actionExecute: {
        actionId: type,
        sourceNodeId: params.sourceNodeId,
        paramsJson: JSON.stringify({ taskId }),
      },
    });
  };

  const cancelTask = (taskId: string) =>
    socketClient.send({ taskCancel: { taskId } });

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
    await socketClient.send({ actionExecute: { actionId, sourceNodeId } });
    return { type: "immediate" };
  };

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
