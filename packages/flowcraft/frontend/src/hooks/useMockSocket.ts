import { useState, useEffect } from "react";
import { flowcraft_proto } from "../generated/flowcraft_proto";
import type { NodeTemplate, AppNode, TaskDefinition } from "../types";
import { MutationSource, TaskStatus } from "../types";
import { useFlowStore } from "../store/flowStore";
import { useTaskStore } from "../store/taskStore";
import { socketClient } from "../utils/SocketClient";
import { fromProtoGraph, fromProtoNode } from "../utils/protoAdapter";

export const useMockSocket = (config?: { disablePolling?: boolean }) => {
  const { applyMutations, setGraph, ydoc, applyYjsUpdate } = useFlowStore(
    (state) => ({
      applyMutations: state.applyMutations,
      setGraph: state.setGraph,
      ydoc: state.ydoc,
      applyYjsUpdate: state.applyYjsUpdate,
    }),
  );

  const { updateTask } = useTaskStore();
  const [templates, setTemplates] = useState<NodeTemplate[]>([]);

  useEffect(() => {
    const onSnapshot = (snapshot: flowcraft_proto.v1.IGraphSnapshot) => {
      // Convert Proto -> AppNode (plain objects)
      const { nodes, edges } = fromProtoGraph(snapshot);

      const taskId = "initial-sync";
      useTaskStore.getState().registerTask({
        taskId,
        label: "Initial Graph Sync",
        source: MutationSource.SYNC,
        status: TaskStatus.TASK_COMPLETED,
      });

      // Pass plain nodes to store. Store will dehydrate them into Yjs.
      // Then syncFromYjs will read them back and hydrate them using registered handlers.
      setGraph({ nodes, edges }, Number(snapshot.version ?? 0));
    };

    const onYjsUpdate = (update: Uint8Array) => {
      applyYjsUpdate(update);
    };

    const onMutations = (mutations: flowcraft_proto.v1.IGraphMutation[]) => {
      const processed = mutations.map((m) => {
        const cloned = { ...m };
        if (cloned.addNode?.node) {
          // Convert INode -> AppNode (plain object)
          const appNode = fromProtoNode(cloned.addNode.node);
          // We must cast to INode to satisfy the strict Protobuf type,
          // even though it's actually an AppNode now.
          // The store expects AppNode properties (like 'id') inside.
          cloned.addNode.node = appNode as unknown as flowcraft_proto.v1.INode;
        }
        if (cloned.addSubgraph) {
          if (cloned.addSubgraph.nodes) {
            const appNodes = cloned.addSubgraph.nodes.map((n) =>
              fromProtoNode(n),
            );
            cloned.addSubgraph.nodes =
              appNodes as unknown as flowcraft_proto.v1.INode[];
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
        void socketClient.send({ yjsUpdate: update });
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
  }, [setGraph, applyMutations, updateTask, ydoc, applyYjsUpdate]);

  useEffect(() => {
    if (config?.disablePolling) return;
    fetch("/api/node-templates")
      .then((r) => r.json())
      .then((data) => {
        setTemplates(data as NodeTemplate[]);
      })
      .catch(() => {
        // ignore error
      });

    void socketClient.send({ syncRequest: { graphId: "main" } });
  }, [config?.disablePolling]);

  const sendNodeUpdate = (nodeId: string, data: Partial<AppNode["data"]>) =>
    socketClient.send({
      nodeUpdate: {
        nodeId,
        data: data as unknown as flowcraft_proto.v1.INodeData,
      },
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
    return socketClient.send({
      actionExecute: { actionId: "stream", sourceNodeId: nodeId },
    });
  };

  const executeTask = (
    taskId: string,
    type: string,
    params: { sourceNodeId: string },
  ) => {
    return socketClient.send({
      actionExecute: {
        actionId: type,
        sourceNodeId: params.sourceNodeId,
        paramsJson: JSON.stringify({ taskId }),
      },
    });
  };

  const cancelTask = (taskId: string) =>
    socketClient.send({ taskCancel: { taskId } });

  const fetchWidgetOptions = async (
    nodeId: string,
    widgetId: string,
  ): Promise<unknown> => {
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

  const executeAction = async (
    actionId: string,
    sourceNodeId: string,
  ): Promise<{ type: string }> => {
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
      const data = (await res.json()) as {
        actions: flowcraft_proto.v1.IActionTemplate[];
      };
      return data.actions;
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
