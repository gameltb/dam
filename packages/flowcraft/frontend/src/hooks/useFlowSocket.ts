import { create, type JsonObject } from "@bufbuild/protobuf";
import { useCallback, useMemo } from "react";
import { useTable } from "spacetimedb/react";

import { ActionExecutionRequestSchema } from "@/generated/flowcraft/v1/core/action_pb";
import { ViewportSchema } from "@/generated/flowcraft/v1/core/base_pb";
import { TaskStatus, TaskUpdateSchema } from "@/generated/flowcraft/v1/core/node_pb";
import { tables } from "@/generated/spacetime";
import { useFlowStore } from "@/store/flowStore";
import { convertStdbToPb } from "@/utils/pb-client";

export const useFlowSocket = (_args?: unknown) => {
  const { spacetimeConn } = useFlowStore();
  const [stTemplates] = useTable(tables.nodeTemplates);
  const [stInferenceConfig] = useTable(tables.inferenceConfig);

  const templates = useMemo(() => {
    return stTemplates.map((t: any) => {
      const pbTemplate = convertStdbToPb("nodeTemplates", t);
      return {
        ...pbTemplate,
        templateId: t.templateId, // Keep original ID if needed for indexing
      };
    });
  }, [stTemplates]);

  const inferenceConfig = useMemo(() => {
    if (stInferenceConfig.length === 0) return null;
    const firstConfig = stInferenceConfig[0];
    return convertStdbToPb("inferenceConfig", firstConfig);
  }, [stInferenceConfig]);

  const executeTask = useCallback(
    (action: { id: string; params?: unknown }, nodeId: string) => {
      if (spacetimeConn) {
        spacetimeConn.pbreducers.executeAction({
          id: crypto.randomUUID(),
          request: create(ActionExecutionRequestSchema, {
            actionId: action.id,
            contextNodeIds: [],
            params: {
              case: "paramsStruct",
              value: (action.params ?? {}) as JsonObject,
            },
            sourceNodeId: nodeId,
          }),
        });
      }
    },
    [spacetimeConn],
  );

  const cancelTask = useCallback(
    (taskId: string) => {
      if (spacetimeConn) {
        spacetimeConn.pbreducers.updateTaskStatus({
          update: create(TaskUpdateSchema, {
            displayLabel: "",
            message: "Cancelled by user",
            nodeId: "",
            progress: 0,
            result: undefined,
            status: TaskStatus.TASK_CANCELLED,
            taskId,
            type: "",
          }),
        });
      }
    },
    [spacetimeConn],
  );

  return {
    cancelTask,
    executeTask,
    inferenceConfig,
    restartTask: (taskId: string) => {
      console.log("Restarting task:", taskId);
    },
    streamAction: (nodeId: string, widgetId: string) => {
      console.log("streamAction", nodeId, widgetId);
    },
    templates,
    updateNodeData: (id: string, data: unknown) => {
      // Logic to merge data and call updateNodePb
      console.log("updateNodeData", id, data);
    },
    updateViewport: (x: number, y: number, zoom: number) => {
      spacetimeConn?.pbreducers.updateViewport({
        id: "default",
        viewport: create(ViewportSchema, { x, y, zoom }),
      } as any);
    },
    updateWidget: (nodeId: string, widgetId: string, value: string) => {
      spacetimeConn?.reducers.updateWidgetValue({ nodeId, value, widgetId });
    },
  };
};
