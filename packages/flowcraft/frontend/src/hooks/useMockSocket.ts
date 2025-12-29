/* eslint-disable @typescript-eslint/no-explicit-any */
import { useState, useEffect } from "react";
import { type NodeData } from "../generated/core/node_pb";
import {
  type GraphSnapshot,
  type GraphMutation,
} from "../generated/core/service_pb";
import {
  type ActionTemplate,
  ActionExecutionRequestSchema,
} from "../generated/action_pb";
import {
  SyncRequestSchema,
  UpdateNodeRequestSchema,
  UpdateWidgetRequestSchema,
  TaskCancelRequestSchema,
} from "../generated/core/service_pb";
import { create } from "@bufbuild/protobuf";
import type { NodeTemplate, TaskDefinition } from "../types";
import { MutationSource, TaskStatus } from "../types";
import { useFlowStore } from "../store/flowStore";
import { useTaskStore } from "../store/taskStore";
import { socketClient } from "../utils/SocketClient";
import { fromProtoGraph } from "../utils/protoAdapter";

export const useMockSocket = (_config?: { disablePolling?: boolean }) => {
  const { applyMutations, setGraph, applyYjsUpdate } = useFlowStore(
    (state) => ({
      applyMutations: state.applyMutations,
      setGraph: state.setGraph,
      applyYjsUpdate: state.applyYjsUpdate,
    }),
  );

  const { updateTask } = useTaskStore();
  const [templates] = useState<NodeTemplate[]>([]);

  useEffect(() => {
    const onSnapshot = (snapshot: GraphSnapshot) => {
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
      setGraph({ nodes, edges }, 0);
    };

    const onYjsUpdate = (update: Uint8Array) => {
      applyYjsUpdate(update);
    };

    const onMutations = (mutations: GraphMutation[]) => {
      applyMutations(mutations, { source: MutationSource.SYNC });
    };

    const onTaskUpdate = (update: TaskDefinition) => {
      updateTask(update.taskId, update);
    };

    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    socketClient.on("snapshot", onSnapshot as any);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    socketClient.on("yjsUpdate", onYjsUpdate as any);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    socketClient.on("mutations", onMutations as any);
    // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
    socketClient.on("taskUpdate", onTaskUpdate as any);

    // Initial Sync Request
    void socketClient.send({
      payload: {
        case: "syncRequest",
        value: create(SyncRequestSchema, {
          graphId: "default",
        }),
      },
    });

    return () => {
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      socketClient.off("snapshot", onSnapshot as any);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      socketClient.off("yjsUpdate", onYjsUpdate as any);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      socketClient.off("mutations", onMutations as any);
      // eslint-disable-next-line @typescript-eslint/no-unsafe-argument
      socketClient.off("taskUpdate", onTaskUpdate as any);
    };
  }, [applyMutations, setGraph, applyYjsUpdate, updateTask]);

  const executeTask = (action: ActionTemplate, nodeId?: string) => {
    void socketClient.send({
      payload: {
        case: "actionExecute",
        value: create(ActionExecutionRequestSchema, {
          actionId: action.id,
          sourceNodeId: nodeId ?? "",
          contextNodeIds: [],
          paramsJson: "{}",
        }),
      },
    });
  };

  const cancelTask = (taskId: string) => {
    void socketClient.send({
      payload: {
        case: "taskCancel",
        value: create(TaskCancelRequestSchema, {
          taskId,
        }),
      },
    });
  };

  const updateNodeData = (nodeId: string, data: NodeData) => {
    void socketClient.send({
      payload: {
        case: "nodeUpdate",
        value: create(UpdateNodeRequestSchema, {
          nodeId,
          data,
        }),
      },
    });
  };

  const updateWidget = (nodeId: string, widgetId: string, value: unknown) => {
    void socketClient.send({
      payload: {
        case: "widgetUpdate",
        value: create(UpdateWidgetRequestSchema, {
          nodeId,
          widgetId,
          valueJson: JSON.stringify(value),
        }),
      },
    });
  };

  const streamAction = (nodeId: string, actionId: string) => {
    void socketClient.send({
      payload: {
        case: "actionExecute",
        value: create(ActionExecutionRequestSchema, {
          actionId,
          sourceNodeId: nodeId,
          contextNodeIds: [],
          paramsJson: JSON.stringify({ stream: true }),
        }),
      },
    });
  };

  return {
    templates,
    executeTask,
    cancelTask,
    updateNodeData,
    updateWidget,
    streamAction,
  };
};
