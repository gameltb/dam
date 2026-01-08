import {
  create,
  fromJson,
  type JsonObject,
  type JsonValue,
} from "@bufbuild/protobuf";
import { ValueSchema } from "@bufbuild/protobuf/wkt";
import { Code, ConnectError } from "@connectrpc/connect";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

import type { NodeTemplate, TaskDefinition } from "@/types";

import {
  ActionExecutionRequestSchema,
  type ActionTemplate,
} from "@/generated/flowcraft/v1/core/action_pb";
import { PresentationSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { type NodeData } from "@/generated/flowcraft/v1/core/node_pb";
import {
  type GraphMutation,
  type GraphSnapshot,
} from "@/generated/flowcraft/v1/core/service_pb";
import {
  InferenceConfigDiscoveryRequestSchema,
  type InferenceConfigDiscoveryResponse,
  SyncRequestSchema,
  TaskCancelRequestSchema,
  TemplateDiscoveryRequestSchema,
  type TemplateDiscoveryResponse,
  UpdateNodeRequestSchema,
  UpdateWidgetRequestSchema,
  ViewportUpdateSchema,
} from "@/generated/flowcraft/v1/core/service_pb";
import {
  NodeSignalSchema,
  RestartInstanceSchema,
} from "@/generated/flowcraft/v1/core/signals_pb";
import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { MutationSource, TaskStatus } from "@/types";
import { fromProtoGraph } from "@/utils/protoAdapter";
import { socketClient } from "@/utils/SocketClient";

export const useFlowSocket = (_config?: { disablePolling?: boolean }) => {
  const { applyMutations, applyYjsUpdate, setGraph } = useFlowStore(
    useShallow((state) => ({
      applyMutations: state.applyMutations,
      applyYjsUpdate: state.applyYjsUpdate,
      setGraph: state.setGraph,
    })),
  );

  const { serverAddress } = useUiStore(
    useShallow((state) => ({
      serverAddress: state.settings.serverAddress,
    })),
  );

  const { updateTask } = useTaskStore();
  const [templates, setTemplates] = useState<NodeTemplate[]>([]);
  const [inferenceConfig, setInferenceConfig] =
    useState<InferenceConfigDiscoveryResponse | null>(null);
  const hasRequestedSync = useRef(false);
  const lastServerAddress = useRef(serverAddress);

  useEffect(() => {
    const onSnapshot = (snapshot: GraphSnapshot) => {
      // Convert Proto -> AppNode (plain objects)
      const { edges, nodes } = fromProtoGraph(snapshot);

      const taskId = "initial-sync";
      useTaskStore.getState().registerTask({
        label: "Initial Graph Sync",
        source: MutationSource.SOURCE_SYNC,
        status: TaskStatus.TASK_COMPLETED,
        taskId,
      });

      // Pass plain nodes to store. Store will dehydrate them into Yjs.
      setGraph({ edges, nodes }, 0);
    };

    const onYjsUpdate = (update: Uint8Array) => {
      applyYjsUpdate(update);
    };

    const onMutations = (mutations: GraphMutation[]) => {
      applyMutations(mutations, { source: MutationSource.SOURCE_SYNC });
    };

    const onTaskUpdate = (update: TaskDefinition) => {
      updateTask(update.taskId, update);
    };

    const onTemplates = (response: TemplateDiscoveryResponse) => {
      setTemplates(response.templates);
    };

    const onInferenceConfig = (response: InferenceConfigDiscoveryResponse) => {
      setInferenceConfig(response);
    };

    socketClient.on("snapshot", onSnapshot);
    socketClient.on("yjsUpdate", onYjsUpdate);
    socketClient.on("mutations", onMutations);
    socketClient.on("taskUpdate", onTaskUpdate);
    socketClient.on("templates", onTemplates);
    socketClient.on("inferenceConfig", onInferenceConfig);

    const onError = (error: unknown) => {
      if (error instanceof ConnectError && error.code === Code.Canceled) {
        return;
      }
      console.error("Socket Error Event:", error);
    };
    socketClient.on("error", onError);

    // Initial Sync Request or URL change
    if (
      hasRequestedSync.current &&
      lastServerAddress.current !== serverAddress
    ) {
      hasRequestedSync.current = false;
    }

    if (!hasRequestedSync.current) {
      hasRequestedSync.current = true;
      lastServerAddress.current = serverAddress;
      void socketClient.send({
        payload: {
          case: "syncRequest",
          value: create(SyncRequestSchema, {
            graphId: "default",
          }),
        },
      });

      // Initial Template Discovery
      void socketClient.send({
        payload: {
          case: "templateDiscovery",
          value: create(TemplateDiscoveryRequestSchema, {}),
        },
      });

      // Initial Inference Config Discovery
      void socketClient.send({
        payload: {
          case: "inferenceDiscovery",
          value: create(InferenceConfigDiscoveryRequestSchema, {}),
        },
      });
    }

    return () => {
      socketClient.off("snapshot", onSnapshot);
      socketClient.off("yjsUpdate", onYjsUpdate);
      socketClient.off("mutations", onMutations);
      socketClient.off("taskUpdate", onTaskUpdate);
      socketClient.off("templates", onTemplates);
      socketClient.off("inferenceConfig", onInferenceConfig);
      socketClient.off("error", onError);
    };
  }, [applyMutations, setGraph, applyYjsUpdate, updateTask, serverAddress]);

  const executeTask = useCallback((action: ActionTemplate, nodeId?: string) => {
    void socketClient.send({
      payload: {
        case: "actionExecute",
        value: create(ActionExecutionRequestSchema, {
          actionId: action.id,
          contextNodeIds: [],
          params: {
            case: "paramsStruct",
            value: {} as JsonObject,
          },
          sourceNodeId: nodeId ?? "",
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

  const updateNodeData = useCallback(
    (nodeId: string, data: NodeData, width?: number, height?: number) => {
      const presentation =
        width || height
          ? create(PresentationSchema, { height, width })
          : undefined;

      void socketClient.send({
        payload: {
          case: "nodeUpdate",
          value: create(UpdateNodeRequestSchema, {
            data,
            nodeId,
            presentation,
          }),
        },
      });
    },
    [],
  );

  const updateWidget = useCallback(
    (nodeId: string, widgetId: string, value: unknown) => {
      void socketClient.send({
        payload: {
          case: "widgetUpdate",
          value: create(UpdateWidgetRequestSchema, {
            nodeId,
            value: fromJson(ValueSchema, value as JsonValue),
            widgetId,
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
          contextNodeIds: [],
          params: {
            case: "paramsStruct",
            value: { stream: true } as unknown as JsonObject,
          },
          sourceNodeId: nodeId,
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
            visibleBounds: { height, width, x, y },
          }),
        },
      });
    },
    [],
  );

  const restartTask = useCallback((nodeId: string) => {
    void socketClient.send({
      payload: {
        case: "nodeSignal",
        value: create(NodeSignalSchema, {
          nodeId,
          payload: {
            case: "restartInstance",
            value: create(RestartInstanceSchema, {}),
          },
        }),
      },
    });
  }, []);

  return useMemo(
    () => ({
      cancelTask,
      executeTask,
      inferenceConfig,
      restartTask,
      streamAction,
      templates,
      updateNodeData,
      updateViewport,
      updateWidget,
    }),
    [
      templates,
      inferenceConfig,
      executeTask,
      cancelTask,
      restartTask,
      updateNodeData,
      updateWidget,
      streamAction,
      updateViewport,
    ],
  );
};
