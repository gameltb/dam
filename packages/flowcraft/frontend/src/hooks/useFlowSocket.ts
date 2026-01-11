import { create, toJson } from "@bufbuild/protobuf";
import { Code, ConnectError } from "@connectrpc/connect";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useShallow } from "zustand/react/shallow";

import type { NodeTemplate } from "@/types";

import {
  type ActionTemplate,
} from "@/generated/flowcraft/v1/core/action_pb";
import {
  type NodeData,
  NodeDataSchema,
} from "@/generated/flowcraft/v1/core/node_pb";
// import {
//   type GraphMutation,
//   type GraphSnapshot,
// } from "@/generated/flowcraft/v1/core/service_pb";
import {
  InferenceConfigDiscoveryRequestSchema,
  type InferenceConfigDiscoveryResponse,
  TemplateDiscoveryRequestSchema,
  type TemplateDiscoveryResponse,
} from "@/generated/flowcraft/v1/core/service_pb";


import { useFlowStore } from "@/store/flowStore";
import { useTaskStore } from "@/store/taskStore";
import { useUiStore } from "@/store/uiStore";
import { TaskStatus } from "@/types";
// import { fromProtoGraph } from "@/utils/protoAdapter";
import { socketClient } from "@/utils/SocketClient";

export const useFlowSocket = (_config?: { disablePolling?: boolean }) => {
  const { spacetimeConn } = useFlowStore(
    useShallow((state) => ({
      // applyMutations: state.applyMutations,
      // applyYjsUpdate: state.applyYjsUpdate,
      // setGraph: state.setGraph,
      spacetimeConn: state.spacetimeConn,
    })),
  );

  const { serverAddress } = useUiStore(
    useShallow((state) => ({
      serverAddress: state.settings.serverAddress,
    })),
  );

  const [templates, setTemplates] = useState<NodeTemplate[]>([]);
  const [inferenceConfig, setInferenceConfig] =
    useState<InferenceConfigDiscoveryResponse | null>(null);
  const hasRequestedSync = useRef(false);
  const lastServerAddress = useRef(serverAddress);

  useEffect(() => {
    socketClient.updateBaseUrl(serverAddress);
  }, [serverAddress]);

  useEffect(() => {
    const onTemplates = (response: TemplateDiscoveryResponse) => {
      setTemplates(response.templates);
    };

    const onInferenceConfig = (response: InferenceConfigDiscoveryResponse) => {
      setInferenceConfig(response);
    };

    socketClient.on("templates", onTemplates);
    socketClient.on("inferenceConfig", onInferenceConfig);

    const onError = (error: unknown) => {
      if (error instanceof ConnectError && error.code === Code.Canceled) {
        return;
      }
      console.error("Socket Error Event:", error);
    };
    socketClient.on("error", onError);

    // Initial Discovery Requests
    if (
      hasRequestedSync.current &&
      lastServerAddress.current !== serverAddress
    ) {
      hasRequestedSync.current = false;
    }

    if (!hasRequestedSync.current) {
      hasRequestedSync.current = true;
      lastServerAddress.current = serverAddress;

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
      socketClient.off("templates", onTemplates);
      socketClient.off("inferenceConfig", onInferenceConfig);
      socketClient.off("error", onError);
    };
  }, [serverAddress]);

  const executeTask = useCallback(
    (action: ActionTemplate, nodeId?: string) => {
      if (!spacetimeConn) {
        console.warn("[SpacetimeDB] Cannot execute action: No connection");
        return "";
      }

      const taskId = crypto.randomUUID();

      useTaskStore.getState().registerTask({
        label: action.label,
        message: "Starting action...",
        nodeId,
        status: TaskStatus.TASK_PENDING,
        taskId,
      });

      try {
        spacetimeConn.reducers.executeAction({
          actionId: action.id,
          id: taskId,
          nodeId: nodeId ?? "",
          paramsJson: JSON.stringify({}),
        });
      } catch (err) {
        console.error("[SpacetimeDB] Failed to execute action:", err);
        useTaskStore.getState().updateTask(taskId, {
          message: String(err),
          status: TaskStatus.TASK_FAILED,
        });
      }
      return taskId;
    },
    [spacetimeConn],
  );

  const cancelTask = useCallback(
    (taskId: string) => {
      if (!spacetimeConn) return;
      // Note: We should ideally have a cancel reducer in STDB. 
      // For now, updating status to cancelled in STDB might trigger the backend.
      spacetimeConn.reducers.updateTaskStatus({
        id: taskId,
        resultJson: "Cancelled by user",
        status: "cancelled",
      });
    },
    [spacetimeConn],
  );

  const updateNodeData = useCallback(
    (nodeId: string, data: NodeData, width?: number, height?: number) => {
      const taskId = crypto.randomUUID();

      if (spacetimeConn) {
        try {
          if (width !== undefined || height !== undefined) {
            spacetimeConn.reducers.updateNodeLayout({
              height: height ?? 0,
              id: nodeId,
              width: width ?? 0,
              x: 0,
              y: 0,
            });
          }
          spacetimeConn.reducers.updateNodeData({
            dataJson: JSON.stringify(toJson(NodeDataSchema, data)),
            id: nodeId,
          });
          return taskId;
        } catch (err) {
          console.error("[SpacetimeDB] Failed to update node data:", err);
        }
      }
      return taskId;
    },
    [spacetimeConn],
  );

  const updateWidget = useCallback(
    (nodeId: string, widgetId: string, value: unknown) => {
      const taskId = crypto.randomUUID();
      if (spacetimeConn) {
        try {
          spacetimeConn.reducers.updateWidgetValue({
            nodeId,
            valueJson: JSON.stringify(value),
            widgetId,
          });
        } catch (err) {
          console.error("[SpacetimeDB] Failed to update widget value:", err);
        }
      }
      return taskId;
    },
    [spacetimeConn],
  );

  const streamAction = useCallback(
    (nodeId: string, actionId: string) => {
      if (!spacetimeConn) return "";
      const taskId = crypto.randomUUID();

      useTaskStore.getState().registerTask({
        label: `Streaming ${actionId}`,
        nodeId,
        status: TaskStatus.TASK_PENDING,
        taskId,
      });

      try {
        spacetimeConn.reducers.executeAction({
          actionId: actionId,
          id: taskId,
          nodeId: nodeId,
          paramsJson: JSON.stringify({ stream: true }),
        });
      } catch (err) {
        console.error("[SpacetimeDB] Failed to stream action:", err);
        useTaskStore
          .getState()
          .updateTask(taskId, { status: TaskStatus.TASK_FAILED });
      }
      return taskId;
    },
    [spacetimeConn],
  );

  const updateViewport = useCallback(
    (x: number, y: number, zoom: number) => {
      if (spacetimeConn) {
        try {
          spacetimeConn.reducers.updateViewport({
            id: "default",
            x,
            y,
            zoom,
          });
        } catch (err) {
          console.error("[SpacetimeDB] Failed to update viewport:", err);
        }
      }
    },
    [spacetimeConn],
  );

  const restartTask = useCallback(
    (nodeId: string) => {
      if (!spacetimeConn) return;
      spacetimeConn.reducers.sendNodeSignal({
        id: crypto.randomUUID(),
        nodeId,
        payloadJson: JSON.stringify({}),
        signalCase: "restartInstance",
      });
    },
    [spacetimeConn],
  );

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
