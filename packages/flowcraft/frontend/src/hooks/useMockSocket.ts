import { useState, useEffect, useCallback, useMemo } from "react";
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
  ViewportUpdateSchema,
} from "../generated/core/service_pb";
import { create } from "@bufbuild/protobuf";
import type { NodeTemplate, TaskDefinition } from "../types";
import { MutationSource, TaskStatus } from "../types";
import { useFlowStore } from "../store/flowStore";
import { useTaskStore } from "../store/taskStore";
import { socketClient } from "../utils/SocketClient";
import { fromProtoGraph } from "../utils/protoAdapter";

import { useShallow } from "zustand/react/shallow";

export const useMockSocket = (_config?: { disablePolling?: boolean }) => {
  const { applyMutations, setGraph, applyYjsUpdate } = useFlowStore(
    useShallow((state) => ({
      applyMutations: state.applyMutations,
      setGraph: state.setGraph,
      applyYjsUpdate: state.applyYjsUpdate,
    })),
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

    socketClient.on("snapshot", onSnapshot);
    socketClient.on("yjsUpdate", onYjsUpdate);
    socketClient.on("mutations", onMutations);
    socketClient.on("taskUpdate", onTaskUpdate);

    const onError = (error: unknown) => {
      console.error("Socket Error Event:", error);
    };
    socketClient.on("error", onError);

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
      socketClient.off("snapshot", onSnapshot);
      socketClient.off("yjsUpdate", onYjsUpdate);
      socketClient.off("mutations", onMutations);
      socketClient.off("taskUpdate", onTaskUpdate);
      socketClient.off("error", onError);
    };
  }, [applyMutations, setGraph, applyYjsUpdate, updateTask]);

  const executeTask = useCallback((action: ActionTemplate, nodeId?: string) => {
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
  }, []);

  const cancelTask = useCallback((taskId: string) => {
    void socketClient.send({
      payload: {
        case: "taskCancel",
        value: create(TaskCancelRequestSchema, {
          taskId,
        }),
      },
    });
  }, []);

  const updateNodeData = useCallback((nodeId: string, data: NodeData) => {
    void socketClient.send({
      payload: {
        case: "nodeUpdate",
        value: create(UpdateNodeRequestSchema, {
          nodeId,
          data,
        }),
      },
    });
  }, []);

  const updateWidget = useCallback(
    (nodeId: string, widgetId: string, value: unknown) => {
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
    },
    [],
  );

  const streamAction = useCallback((nodeId: string, actionId: string) => {
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
  }, []);

  const updateViewport = useCallback(
    (x: number, y: number, zoom: number, width: number, height: number) => {
      void socketClient.send({
        payload: {
          case: "viewportUpdate",
          value: create(ViewportUpdateSchema, {
            viewport: { x, y, zoom },
            visibleBounds: { x, y, width, height },
          }),
        },
      });
    },
    [],
  );

  return useMemo(
    () => ({
      templates,
      executeTask,
      cancelTask,
      updateNodeData,
      updateWidget,
      streamAction,
      updateViewport,
    }),
    [
      templates,
      executeTask,
      cancelTask,
      updateNodeData,
      updateWidget,
      streamAction,
      updateViewport,
    ],
  );
};
